import logging
from collections import OrderedDict
from copy import deepcopy
import time
import functools

import pandas as pd
import torch as th
import numpy as np

from motor_braindecode.datautil.splitters import concatenate_sets
from motor_braindecode.experiments.loggers import Printer
from motor_braindecode.experiments.stopcriteria import MaxEpochs, ColumnBelow, Or
from motor_braindecode.torch_ext.util import np_to_var

from motor_braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F

from torch.func import functional_call



log = logging.getLogger(__name__)


def time_function(func):
    """Decorator to time function execution and log the duration."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        log.info(f"[TIMING] {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper


class RememberBest(object):
    """
    Class to remember and restore 
    the parameters of the model and the parameters of the
    optimizer at the epoch with the best performance.

    Parameters
    ----------
    column_name: str
        The lowest value in this column should indicate the epoch with the
        best performance (e.g. misclass might make sense).
        
    Attributes
    ----------
    best_epoch: int
        Index of best epoch
    """

    def __init__(self, column_name):
        self.column_name = column_name
        self.best_epoch = 0
        self.lowest_val = float("inf")
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self, epochs_df, model, optimizer):
        """
        Remember this epoch: Remember parameter values in case this epoch
        has the best performance so far.
        
        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
            Dataframe containing the column `column_name` with which performance
            is evaluated.
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info(
                "New best {:s}: {:5f}".format(self.column_name, current_val)
            )
            log.info("")

    def reset_to_best_model(self, epochs_df, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows 
        after best epoch from epochs dataframe.
        
        Modifies parameters of model and optimizer, changes epochs_df in-place.
        
        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`

        """
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch + 1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class Experiment(object):
    """
    Class that performs one experiment on training, validation and test set.

    It trains as follows:
    
    1. Train on training set until a given stop criterion is fulfilled
    2. Reset to the best epoch, i.e. reset parameters of the model and the 
       optimizer to the state at the best epoch ("best" according to a given
       criterion)
    3. Continue training on the combined training + validation set until the
       loss on the validation set is as low as it was on the best epoch for the
       training set. (or until the ConvNet was trained twice as many epochs as
       the best epoch to prevent infinite training)

    Parameters
    ----------
    model: `torch.nn.Module`
    train_set: :class:`.SignalAndTarget`
    valid_set: :class:`.SignalAndTarget`
    test_set: :class:`.SignalAndTarget`
    iterator: iterator object
    loss_function: function 
        Function mapping predictions and targets to a loss: 
        (predictions: `torch.autograd.Variable`, 
        targets:`torch.autograd.Variable`)
        -> loss: `torch.autograd.Variable`
    optimizer: `torch.optim.Optimizer`
    model_constraint: object
        Object with apply function that takes model and constraints its 
        parameters. `None` for no constraint.
    monitors: list of objects
        List of objects with monitor_epoch and monitor_set method, should
        monitor the traning progress.
    stop_criterion: object
        Object with `should_stop` method, that takes in monitoring dataframe
        and returns if training should stop:
    remember_best_column: str
        Name of column to use for storing parameters of best model. Lowest value
        should indicate best performance in this column.
    run_after_early_stop: bool
        Whether to continue running after early stop
    model_loss_function: function, optional
        Function (model -> loss) to add a model loss like L2 regularization.
        Note that this loss is not accounted for in monitoring at the moment.
    batch_modifier: object, optional
        Object with modify method, that can change the batch, e.g. for data
        augmentation
    cuda: bool, optional
        Whether to use cuda.
    pin_memory: bool, optional
        Whether to pin memory of inputs and targets of batch.
    do_early_stop: bool
        Whether to do an early stop at all. If true, reset to best model
        even in case experiment does not run after early stop.
    reset_after_second_run: bool
        If true, reset to best model when second run did not find a valid loss
        below or equal to the best train loss of first run.
    log_0_epoch: bool
        Whether to compute monitor values and log them before the
        start of training.
    loggers: list of :class:`.Logger`
        How to show computed metrics.
        
    Attributes
    ----------
    epochs_df: `pandas.DataFrame`
        Monitoring values for all epochs.
    """

    def __init__(
        self,
        model,
        train_set,
        valid_set,
        test_set,
        iterator,
        loss_function,
        optimizer,
        model_constraint,
        monitors,
        stop_criterion,
        remember_best_column,
        run_after_early_stop,
        model_loss_function=None,
        batch_modifier=None,
        cuda=True,
        pin_memory=False,
        do_early_stop=True,
        reset_after_second_run=False,
        log_0_epoch=True,
        loggers=("print",),
        meta = True,
        inner_lr = 1e-3
    ):
        if run_after_early_stop or reset_after_second_run:
            assert do_early_stop == True, (
                "Can only run after early stop or "
                "reset after second run if doing an early stop"
            )
        if do_early_stop:
            assert valid_set is not None
            assert remember_best_column is not None
        self.model = model
        self.valid_set_meta = valid_set
        self.datasets = OrderedDict(
            (("train", train_set), ("valid", valid_set), ("test", test_set))
        )
        if valid_set is None:
            self.datasets.pop("valid")
            assert run_after_early_stop == False
            assert do_early_stop == False
        if test_set is None:
            self.datasets.pop("test")

        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_constraint = model_constraint
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.remember_best_column = remember_best_column
        self.run_after_early_stop = run_after_early_stop
        self.model_loss_function = model_loss_function
        self.batch_modifier = batch_modifier
        self.cuda = cuda
        self.epochs_df = pd.DataFrame()
        self.before_stop_df = None
        self.rememberer = None
        self.pin_memory = pin_memory
        self.do_early_stop = do_early_stop
        self.reset_after_second_run = reset_after_second_run
        self.log_0_epoch = log_0_epoch
        self.loggers = loggers
        self.meta = meta

        self.n_tasks_per_meta_batch = 8 
        self.n_meta_batches_per_epoch =  max(1, 54 // self.n_tasks_per_meta_batch)
        self.inner_steps = 5
        self.inner_lr = inner_lr
        self.task_source = "train"    

    @time_function
    def run(self):
        """
        Run complete training.
        """
        self.setup_training()
        log.info("Run until first stop...")
        self.run_until_first_stop()
        if self.do_early_stop:
            # always setup for second stop, in order to get best model
            # even if not running after early stop...
            log.info("Setup for second stop...")
            self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            loss_to_reach = float(self.epochs_df["train_loss"].iloc[-1])
            self.run_until_second_stop()
            if (
                float(self.epochs_df["valid_loss"].iloc[-1]) > loss_to_reach
            ) and self.reset_after_second_run:
                # if no valid loss was found below the best train loss on 1st
                # run, reset model to the epoch with lowest valid_misclass
                log.info(
                    "Resetting to best epoch {:d}".format(
                        self.rememberer.best_epoch
                    )
                )
                self.rememberer.reset_to_best_model(
                    self.epochs_df, self.model, self.optimizer
                )

    def setup_training(self):
        """
        Setup training, i.e. transform model to cuda,
        initialize monitoring.
        """
        # reset remember best extension in case you rerun some experiment
        if self.do_early_stop:
            self.rememberer = RememberBest(self.remember_best_column)
        if self.loggers == ("print",):
            self.loggers = [Printer()]
        self.epochs_df = pd.DataFrame()
        if self.cuda:
            assert th.cuda.is_available(), "Cuda not available"
            self.model.cuda()

    def run_until_first_stop(self):
        """
        Run training and evaluation using only training set for training
        until stop criterion is fulfilled.
        """
        self.run_until_stop(self.datasets, remember_best=self.do_early_stop)

    def run_until_second_stop(self):
        """
        Run training and evaluation using combined training + validation set 
        for training. 
        
        Runs until loss on validation  set decreases below loss on training set 
        of best epoch or  until as many epochs trained after as before 
        first stop.
        """
        datasets = self.datasets
        datasets["train"] = concatenate_sets(
            [datasets["train"], datasets["valid"]]
        )

        self.run_until_stop(datasets, remember_best=True)

    @time_function
    def run_until_stop(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets until stop criterion is
        fulfilled.
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters at best epoch.
        """
        if self.log_0_epoch:
            self.monitor_epoch(datasets)
            self.log_epoch()
            if remember_best:
                self.rememberer.remember_epoch(
                    self.epochs_df, self.model, self.optimizer
                )

        self.iterator.reset_rng() # reset random number generator
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(datasets, remember_best)


    @time_function
    def run_one_epoch(self, datasets, remember_best):
        """
        Run training and evaluation on given datasets for one epoch.
        First, run normal training loop. Then, run meta-learning loop.
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.
        remember_best: bool
            Whether to remember parameters if this epoch is best epoch.
        """
        start_train_epoch_time = time.time()

        # ---- NORMAL TRAINING LOOP (unchanged)
        batch_generator = self.iterator.get_batches(datasets["train"], shuffle=True)
        for inputs, targets in batch_generator:
            if self.batch_modifier is not None:
                inputs, targets = self.batch_modifier.process(inputs, targets)
            if len(inputs) > 0:
                self.train_batch(inputs, targets)
        mid_time = time.time()
        log.info(f"Time for NORMAL training loop: {mid_time - start_train_epoch_time:.2f}s")

        # ---- META-LEARNING LOOP (FOMAML with stateless fast-weights)
        if self.meta:
            # micro-opt: reuse RNG, use running sums (not lists), cheap logging
            rng = getattr(self.iterator, "rng", None)
            if rng is None:
                rng = np.random

            meta_q_loss_sum = 0.0
            meta_s_loss_sum = 0.0

            for _ in range(self.n_meta_batches_per_epoch):
                stats = self.meta_fomaml_step(rng=rng)
                meta_q_loss_sum += stats["meta_query_loss"]
                meta_s_loss_sum += stats["meta_support_loss"]

            if self.n_meta_batches_per_epoch > 0:
                mq = meta_q_loss_sum / self.n_meta_batches_per_epoch
                ms = meta_s_loss_sum / self.n_meta_batches_per_epoch
                log.info(f"[FOMAML] meta_query_loss={mq:.4f} | meta_support_loss={ms:.4f}")

        

        end_train_epoch_time = time.time()
        log.info(f"Time for META-LEARNING loop: {end_train_epoch_time - mid_time:.2f}s")

        log.info("Time only for training updates: {:.2f}s".format(end_train_epoch_time - start_train_epoch_time))

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.rememberer.remember_epoch(self.epochs_df, self.model, self.optimizer)

    @time_function
    def meta_fomaml_step(self, rng=None):
        """
        FOMAML meta-episode using torch.func.functional_call (stateless).
        - Query = one whole 100-trial run from the subject (runs are contiguous).
        - Support = K samples drawn from the other three runs (300 trials).
        - Fast weights kept as leaf tensors; grads aligned by parameter NAME.
        - Robust to RandomState vs Generator RNGs and to unused params (LoRA etc.).
        """
        SAMPLES_PER_SUBJECT = 400
        RUN_SIZE = 100
        N_RUNS = 4
        K_SUPPORT = int(getattr(self, "k_support", 5))
        Q_QUERY = RUN_SIZE
        inner_steps = int(getattr(self, "inner_steps", 1))
        inner_lr = float(getattr(self, "inner_lr", 1e-3))

        if rng is None:
            rng = np.random

        src_ds = self.datasets["train"]
        N = len(src_ds.y)
        n_subjects = N // SAMPLES_PER_SUBJECT
        assert N % SAMPLES_PER_SUBJECT == 0, f"Dataset length {N} not divisible by 400"

        # sample task subjects without replacement
        task_ids = rng.choice(np.arange(n_subjects), size=self.n_tasks_per_meta_batch, replace=False)

        # Trainable (LoRA-safe) base parameters, aligned by NAME
        named_base_params = [(n, p) for (n, p) in self.model.named_parameters() if p.requires_grad]
        base_names = [n for n, _ in named_base_params]
        name_to_param = dict(self.model.named_parameters())

        # Accumulate averaged meta-grads by name, on the correct device
        meta_grads = {n: th.zeros_like(name_to_param[n], device=name_to_param[n].device)
                    for n, _ in named_base_params}

        # Light logging accumulators
        support_loss_sum = 0.0
        query_loss_sum = 0.0

        # Freeze BN/Dropout buffers during meta episodes so support batches don't pollute running stats
        self.model.eval()

        for task_num, sid in enumerate(task_ids, 1):
            subj_start = sid * SAMPLES_PER_SUBJECT
            # Four contiguous 100-trial runs
            run_ranges = [np.arange(subj_start + r * RUN_SIZE, subj_start + (r + 1) * RUN_SIZE) for r in range(N_RUNS)]

            # RandomState vs Generator: randint vs integers
            q_run = int(rng.integers(0, N_RUNS)) if hasattr(rng, "integers") else int(rng.randint(0, N_RUNS))
            query_idx = run_ranges[q_run]
            support_pool = np.concatenate([run_ranges[r] for r in range(N_RUNS) if r != q_run])  # 300 trials
            supp_idx = rng.choice(support_pool, size=K_SUPPORT, replace=False)

            # Fetch arrays and add channel-last singleton if needed
            Xs, Ys = src_ds.X[supp_idx], src_ds.y[supp_idx]
            Xq, Yq = src_ds.X[query_idx], src_ds.y[query_idx]
            if Xs.ndim == 3:
                Xs = Xs[:, :, :, np.newaxis]
                Xq = Xq[:, :, :, np.newaxis]

            xs_var = np_to_var(Xs, pin_memory=self.pin_memory)
            ys_var = np_to_var(Ys, pin_memory=self.pin_memory)
            xq_var = np_to_var(Xq, pin_memory=self.pin_memory)
            yq_var = np_to_var(Yq, pin_memory=self.pin_memory)
            if self.cuda:
                xs_var = xs_var.cuda(); ys_var = ys_var.cuda()
                xq_var = xq_var.cuda(); yq_var = yq_var.cuda()

            # ---- fast weights: LEAF tensors that require grad, aligned by NAME
            fast_params = OrderedDict(
                (n, p.detach().clone().requires_grad_(True))
                for n, p in named_base_params
            )

            # ---- INNER LOOP: take SGD steps on SUPPORT using fast params
            for step in range(inner_steps):
                yhat_s = functional_call(self.model, fast_params, (xs_var,))
                loss_s = self.loss_function(yhat_s, ys_var)
                if self.model_loss_function is not None:
                    loss_s = loss_s + self.model_loss_function(self.model)

                # grads wrt fast weights; allow_unused for params not used on this pass
                grads = th.autograd.grad(
                    loss_s,
                    tuple(fast_params[n] for n in base_names),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

                # SGD update: w' = w - alpha*g ; re-leaf to avoid graph growth
                for n, w, g in zip(base_names, (fast_params[n] for n in base_names), grads):
                    if g is None:
                        # param not used in this forward; treat grad as zero
                        fast_params[n] = w.detach().requires_grad_(True)
                    else:
                        fast_params[n] = (w - inner_lr * g).detach().requires_grad_(True)

                if step == inner_steps - 1:
                    support_loss_sum += float(loss_s.detach().item())

            # ---- QUERY with adapted fast weights
            yhat_q = functional_call(self.model, fast_params, (xq_var,))
            loss_q = self.loss_function(yhat_q, yq_var)
            if self.model_loss_function is not None:
                loss_q = loss_q + self.model_loss_function(self.model)

            query_loss_sum += float(loss_q.detach().item())

            # grads wrt fast weights for OUTER update; allow_unused again
            grads_q = th.autograd.grad(
                loss_q,
                tuple(fast_params[n] for n in base_names),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            scale = 1.0 / float(self.n_tasks_per_meta_batch)
            for n, g in zip(base_names, grads_q):
                if g is not None:
                    meta_grads[n].add_(g.detach() * scale)
                # if g is None, it's effectively a zero update for that param

        # ---- OUTER STEP: write averaged grads into base model and step
        self.optimizer.zero_grad()
        for n in base_names:
            p = name_to_param[n]
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = meta_grads[n]
            else:
                p.grad.copy_(meta_grads[n])
        self.optimizer.step()

        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

        mean_support = support_loss_sum / float(self.n_tasks_per_meta_batch) if self.n_tasks_per_meta_batch > 0 else 0.0
        mean_query = query_loss_sum / float(self.n_tasks_per_meta_batch) if self.n_tasks_per_meta_batch > 0 else 0.0
        return {
            "meta_support_loss": mean_support,
            "meta_query_loss": mean_query,
            "n_tasks": int(self.n_tasks_per_meta_batch),
            "k_support": int(K_SUPPORT),
            "q_query": int(Q_QUERY),
            "inner_steps": int(inner_steps),
        }


    def meta_loss(self, datasets):
        overall_loss = 0
        # batch_generator = self.iterator.get_batches(
        #     datasets["valid"], shuffle=False
        # )
        # for inputs, targets in batch_generator:
        #     print(inputs.shape)

        for i in range(0,int(len(self.valid_set_meta.X)/400)):
            inputs = self.valid_set_meta.X[i*400:i*400 + 5]
            targets =self.valid_set_meta.y[i*400:i*400 + 5]
            inputs = inputs[:,:,:,np.newaxis]
            self.model.train()
            # print(inputs.shape)
            input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
            target_vars = np_to_var(targets, pin_memory=self.pin_memory)
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            if self.model_loss_function is not None:
                loss = loss + self.model_loss_function(self.model)
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            inputs = self.valid_set_meta.X[i*400 + 300:(i+1)*400]
            targets =self.valid_set_meta.y[i*400 + 300:(i+1)*400]
            inputs = inputs[:,:,:,np.newaxis]
            input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
            target_vars = np_to_var(targets, pin_memory=self.pin_memory)
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            outputs = self.model(input_vars)
            meta_loss = self.loss_function(outputs, target_vars)
            if self.model_loss_function is not None:
                meta_loss = meta_loss + self.model_loss_function(self.model)

            overall_loss += meta_loss

        # Meta-Learning Loss
        # batch_generator = self.iterator.get_batches(
        #     datasets["valid"], shuffle=False
        # )
        # for inputs, targets in batch_generator:

        #     inputs , targets = inputs[:5] , targets[:5]

        #     self.model.train()
        #     # print(inputs.shape)
        #     input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
        #     target_vars = np_to_var(targets, pin_memory=self.pin_memory)
        #     if self.cuda:
        #         input_vars = input_vars.cuda()
        #         target_vars = target_vars.cuda()
        #     self.optimizer.zero_grad()
        #     outputs = self.model(input_vars)
        #     loss = self.loss_function(outputs, target_vars)
        #     if self.model_loss_function is not None:
        #         loss = loss + self.model_loss_function(self.model)
        #     loss.backward()
        #     self.optimizer.step()

        #     self.model.eval()
        #     input_vars = np_to_var(inputs[11:], pin_memory=self.pin_memory)
        #     target_vars = np_to_var(targets[11:], pin_memory=self.pin_memory)
        #     if self.cuda:
        #         input_vars = input_vars.cuda()
        #         target_vars = target_vars.cuda()
        #     outputs = self.model(input_vars)
        #     meta_loss = self.loss_function(outputs, target_vars)
        #     if self.model_loss_function is not None:
        #         meta_loss = meta_loss + self.model_loss_function(self.model)

        #     overall_loss += meta_loss
        
        self.model.train()
        overall_loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)        

    ## Change this for meta learning
    def train_batch(self, inputs, targets):
        """
        Train on given inputs and targets.
        
        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`
        """
        self.model.train()
        # print(inputs.shape)
        input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
        target_vars = np_to_var(targets, pin_memory=self.pin_memory)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        if self.model_loss_function is not None:
            loss = loss + self.model_loss_function(self.model)
        
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)
            
    def eval_on_batch(self, inputs, targets):
        """
        Evaluate given inputs and targets.
        
        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`

        Returns
        -------
        predictions: `torch.autograd.Variable`
        loss: `torch.autograd.Variable`

        """
        with th.no_grad():  # OPTIMIZATION: Ensure no gradients computed
            input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
            target_vars = np_to_var(targets, pin_memory=self.pin_memory)
            
            # OPTIMIZATION: Move to GPU only once
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            
            # Forward pass
            outputs = self.model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            
            # OPTIMIZATION: Convert to numpy efficiently
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().detach().numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().detach().numpy() for o in outputs]
            
            # OPTIMIZATION: Convert loss to numpy efficiently
            loss = loss.cpu().detach().numpy()
            
        return outputs, loss

    @time_function
    def monitor_epoch(self, datasets):
        """
        Evaluate one epoch for given datasets.
        
        Stores results in `epochs_df`
        
        Parameters
        ----------
        datasets: OrderedDict
            Dictionary with train, valid and test as str mapping to
            :class:`.SignalAndTarget` objects.

        """
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
        
        for setname in datasets:
            assert setname in ["train", "valid", "test"]
            dataset = datasets[setname]
            
            # OPTIMIZATION 1: Get batch generator once and check length efficiently
            batch_generator = self.iterator.get_batches(dataset, shuffle=False)
            
            # Try to get length without iteration
            if hasattr(batch_generator, "__len__"):
                n_batches = len(batch_generator)
            else:
                # Only iterate once if absolutely necessary
                n_batches = sum(1 for _ in batch_generator)
                batch_generator = self.iterator.get_batches(dataset, shuffle=False)
            
            # OPTIMIZATION 2: Use lists instead of pre-allocated arrays
            all_preds_list = []
            all_targets_list = []
            all_losses = []
            all_batch_sizes = []
            
            # OPTIMIZATION 3: Batch GPU operations and minimize data movement
            self.model.eval()  # Ensure model is in eval mode
            
            for inputs, targets in batch_generator:
                # Process batch
                preds, loss = self.eval_on_batch(inputs, targets)
                
                # Store results directly in lists (no pre-allocation)
                all_preds_list.append(preds)
                all_targets_list.append(targets)
                all_losses.append(loss)
                all_batch_sizes.append(len(targets))
            
            # OPTIMIZATION 4: Concatenate efficiently without nan handling
            if all_preds_list:
                # Concatenate all predictions and targets
                all_preds = np.concatenate(all_preds_list, axis=0)
                all_targets = np.concatenate(all_targets_list, axis=0)
                
                # Add batch dimension as expected by monitors
                all_preds = all_preds[np.newaxis, :]
                all_targets = all_targets[np.newaxis, :]
                all_batch_sizes = [sum(all_batch_sizes)]
                all_losses = [all_losses]
                
                # Pass to monitors
                for m in self.monitors:
                    result_dict = m.monitor_set(
                        setname,
                        all_preds,
                        all_losses,
                        all_batch_sizes,
                        all_targets,
                        dataset,
                    )
                    if result_dict is not None:
                        result_dicts_per_monitor[m].update(result_dict)
        
        # Update epochs dataframe
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        
        # Use concat instead of deprecated append
        if len(self.epochs_df) == 0:
            self.epochs_df = pd.DataFrame([row_dict])
        else:
            self.epochs_df = pd.concat([self.epochs_df, pd.DataFrame([row_dict])], ignore_index=True)
        
        # Ensure column consistency
        assert set(self.epochs_df.columns) == set(row_dict.keys()), (
            "Columns of dataframe: {:s}\n and keys of dict {:s} not same"
        ).format(str(set(self.epochs_df.columns)), str(set(row_dict.keys())))
        self.epochs_df = self.epochs_df[list(row_dict.keys())]


    def log_epoch(self):
        """
        Print monitoring values for this epoch.
        """
        for logger in self.loggers:
            logger.log_epoch(self.epochs_df)

    @time_function
    def setup_after_stop_training(self):
        """
        Setup training after first stop. 
        
        Resets parameters to best parameters and updates stop criterion.
        """
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.before_stop_df = deepcopy(self.epochs_df)
        self.rememberer.reset_to_best_model(
            self.epochs_df, self.model, self.optimizer
        )
        loss_to_reach = float(self.epochs_df["train_loss"].iloc[-1])
        self.stop_criterion = Or(
            stop_criteria=[
                MaxEpochs(max_epochs=self.rememberer.best_epoch * 2),
                ColumnBelow(
                    column_name="valid_loss", target_value=loss_to_reach
                ),
            ]
        )
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))
