import logging
from collections import OrderedDict
from copy import deepcopy
import time

import pandas as pd
import torch as th
import numpy as np

from motor_braindecode.datautil.splitters import concatenate_sets
from motor_braindecode.experiments.loggers import Printer
from motor_braindecode.experiments.stopcriteria import MaxEpochs, ColumnBelow, Or
from motor_braindecode.torch_ext.util import np_to_var

from motor_braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F

from torch.nn.utils.stateless import functional_call 


log = logging.getLogger(__name__)


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
        self.inner_lr = 1e-3
        self.task_source = "train"    

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
        log.info("Time only for training updates: {:.2f}s".format(end_train_epoch_time - start_train_epoch_time))

        self.monitor_epoch(datasets)
        self.log_epoch()
        if remember_best:
            self.rememberer.remember_epoch(self.epochs_df, self.model, self.optimizer)

    def meta_fomaml_step(self, rng=None):
        """
        First-Order MAML meta-episode with stateless fast-weights.

        Sampling per subject (Point 1):
        - 400 trials per subject in-order -> 4 runs of 100 trials.
        - Choose ONE run (0..3) as the QUERY (100 contiguous trials).
        - Choose K support trials from the remaining 3 runs (300 trials total), no replacement.

        Fast weights (Point 2):
        - Use torch>=2.0 stateless.functional_call to avoid deepcopy.
        - Keep all grads ON DEVICE (faster; we are not saving VRAM here).

        Micro-optimizations:
        - Filter to requires_grad params (LoRA-safe).
        - Accumulate averaged grads directly; avoid big Python lists.
        """
        # ---- constants (could be config)
        SAMPLES_PER_SUBJECT = 400
        RUN_SIZE = 100
        N_RUNS = 4
        K_SUPPORT = getattr(self, "k_support", 5)     # default 5-shot
        Q_QUERY = RUN_SIZE                            # whole run for query
        inner_steps = int(getattr(self, "inner_steps", 1))
        inner_lr = float(getattr(self, "inner_lr", 1e-3))

        if rng is None:
            rng = np.random

        src_ds = self.datasets["train"]
        N = len(src_ds.y)
        n_subjects = N // SAMPLES_PER_SUBJECT
        assert N % SAMPLES_PER_SUBJECT == 0, f"Dataset length {N} not divisible by 400"

        # sample tasks (subjects) without replacement
        task_ids = rng.choice(np.arange(n_subjects), size=self.n_tasks_per_meta_batch, replace=False)

        # Collect trainable params once (LoRA-safe); keep their names
        named_base_params = [(n, p) for (n, p) in self.model.named_parameters() if p.requires_grad]
        base_names = [n for n, _ in named_base_params]
        base_params = [p for _, p in named_base_params]

        # Accumulate averaged meta-grads by param name (kept on-device)
        meta_grads = {n: th.zeros_like(p, device=p.device) for n, p in named_base_params}

        # Running sums for logging
        support_loss_sum = 0.0
        query_loss_sum = 0.0

        # Freeze stochastic layers/buffers during support/query to avoid contaminating global BN
        # (stateless call overrides *parameters*; buffers live in the module; keep model in eval to freeze BN stats)
        self.model.eval()

        for task_num, sid in enumerate(task_ids, 1):
            subj_start = sid * SAMPLES_PER_SUBJECT

            # ---- build per-run index ranges for this subject
            run_ranges = [np.arange(subj_start + r * RUN_SIZE, subj_start + (r + 1) * RUN_SIZE) for r in range(N_RUNS)]

            # ---- choose query run; support pool is the other three runs
            q_run = int(rng.integers(0, N_RUNS))
            query_idx = run_ranges[q_run]  # 100 contiguous trials
            support_pool = np.concatenate([run_ranges[r] for r in range(N_RUNS) if r != q_run])
            assert support_pool.size == SAMPLES_PER_SUBJECT - RUN_SIZE  # 300

            # ---- sample K support (no replacement) from the 300
            supp_idx = rng.choice(support_pool, size=K_SUPPORT, replace=False)

            # ---- fetch arrays; add channel-last singleton axis if needed
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

            # ---- initialize fast-weights as a (detached) copy of base params (by name)
            fast_params = {n: p.detach().clone() for n, p in named_base_params}

            # ---- INNER LOOP: K steps of GD on SUPPORT using stateless functional forward
            for step in range(inner_steps):
                yhat_s = functional_call(self.model, fast_params, (xs_var,))
                loss_s = self.loss_function(yhat_s, ys_var)
                if self.model_loss_function is not None:
                    # regularization computed on fast weights -> use stateless forward/model_loss on a shadow module if needed
                    loss_s = loss_s + self.model_loss_function(self.model)  # lightweight; doesn’t touch weights

                # grads wrt fast (trainable) params; first-order (no create_graph)
                grads = th.autograd.grad(loss_s, tuple(fast_params.values()), retain_graph=False, create_graph=False)

                # SGD update: w' = w - alpha * g   (kept on-device; no deepcopy)
                for (n, w), g in zip(fast_params.items(), grads):
                    fast_params[n] = w - inner_lr * g

                if step == inner_steps - 1:
                    support_loss_sum += float(loss_s.detach().item())

            # ---- QUERY: compute loss using the adapted fast weights (same stateless call)
            yhat_q = functional_call(self.model, fast_params, (xq_var,))
            loss_q = self.loss_function(yhat_q, yq_var)
            if self.model_loss_function is not None:
                loss_q = loss_q + self.model_loss_function(self.model)

            query_loss_sum += float(loss_q.detach().item())

            # grads wrt fast weights; accumulate (average) into meta_grads mapped by name
            grads_q = th.autograd.grad(loss_q, tuple(fast_params.values()), retain_graph=False, create_graph=False)
            scale = 1.0 / float(self.n_tasks_per_meta_batch)
            for (n, _), g in zip(fast_params.items(), grads_q):
                meta_grads[n].add_(g.detach() * scale)

            # (no need to delete; locals go out of scope; keeping everything on GPU is faster)

        # ---- OUTER STEP: write the averaged grads into the base model and step
        self.optimizer.zero_grad()
        name_to_param = dict(self.model.named_parameters())
        for n in base_names:
            p = name_to_param[n]
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = meta_grads[n]  # already on the correct device
            else:
                p.grad.copy_(meta_grads[n])
        self.optimizer.step()

        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

        # return lightweight stats for the epoch logger
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
        self.model.eval()
        with th.no_grad():
            input_vars = np_to_var(inputs, pin_memory=self.pin_memory)
            target_vars = np_to_var(targets, pin_memory=self.pin_memory)
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            outputs = self.model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            if hasattr(outputs, "cpu"):
                outputs = outputs.cpu().detach().numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().detach().numpy() for o in outputs]
            loss = loss.cpu().detach().numpy()
        return outputs, loss

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
            batch_generator = self.iterator.get_batches(dataset, shuffle=False)
            if hasattr(batch_generator, "__len__"):
                # prevent loading of data to estimate number of batches when
                # using lazy iterators
                n_batches = len(batch_generator)
            else:
                # iterating through traditional iterators is cheap, since
                # nothing is loaded, recreate generator afterwards
                n_batches = sum(1 for i in batch_generator)
                batch_generator = self.iterator.get_batches(
                    dataset, shuffle=False
                )
            all_preds, all_targets = None, None
            all_losses, all_batch_sizes = [], []
            for inputs, targets in batch_generator:
                preds, loss = self.eval_on_batch(inputs, targets)
                all_losses.append(loss)
                all_batch_sizes.append(len(targets))
                if all_preds is None:
                    assert all_targets is None
                    if len(preds.shape) == 2:
                        # first batch size is largest
                        max_size, n_classes = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (n_batches * max_size, n_classes), dtype=np.float32
                        )
                    else:
                        assert len(preds.shape) == 3
                        # first batch size is largest
                        max_size, n_classes, n_preds_per_input = preds.shape
                        # pre-allocate memory for all predictions and targets
                        all_preds = np.nan * np.ones(
                            (
                                n_batches * max_size,
                                n_classes,
                                n_preds_per_input,
                            ),
                            dtype=np.float32,
                        )
                    all_preds[: len(preds)] = preds
                    all_targets = np.nan * np.ones((n_batches * max_size))
                    all_targets[: len(targets)] = targets
                else:
                    start_i = sum(all_batch_sizes[:-1])
                    stop_i = sum(all_batch_sizes)
                    all_preds[start_i:stop_i] = preds
                    all_targets[start_i:stop_i] = targets

            # check for unequal batches
            unequal_batches = len(set(all_batch_sizes)) > 1
            all_batch_sizes = sum(all_batch_sizes)
            # remove nan rows in case of unequal batch sizes
            if unequal_batches:
                assert np.sum(np.isnan(all_preds[: all_batch_sizes - 1])) == 0
                assert np.sum(np.isnan(all_preds[all_batch_sizes:])) > 0
                # TODO: is there a reason we dont just take
                # all_preds = all_preds[:all_batch_sizes] and
                # all_targets = all_targets[:all_batch_sizes] ?
                range_to_delete = range(all_batch_sizes, len(all_preds))
                all_preds = np.delete(all_preds, range_to_delete, axis=0)
                all_targets = np.delete(all_targets, range_to_delete, axis=0)
            assert (
                np.sum(np.isnan(all_preds)) == 0
            ), "There are still nans in predictions"
            assert (
                np.sum(np.isnan(all_targets)) == 0
            ), "There are still nans in targets"
            # add empty dimension
            # monitors expect n_batches x ...
            all_preds = all_preds[np.newaxis, :]
            all_targets = all_targets[np.newaxis, :]
            all_batch_sizes = [all_batch_sizes]
            all_losses = [all_losses]

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
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
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
