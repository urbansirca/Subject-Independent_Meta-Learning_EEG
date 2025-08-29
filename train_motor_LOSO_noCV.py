#!/usr/bin/env python
# coding: utf-8
"""Subject-adaptative classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
"""
import argparse
import json
import logging
import sys
from os.path import join as pjoin
import os
from types import new_class
import yaml
from datetime import datetime

import pandas as pd
from motor_braindecode.models.deep4 import Deep5Net

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from motor_braindecode.datautil.signal_target import SignalAndTarget

# from braindecode.models.deep4 import Deep4Net
from motor_braindecode.torch_ext.optimizers import AdamW
from motor_braindecode.torch_ext.util import set_random_seeds
from sklearn.model_selection import KFold
from torch import nn

## MAML EEG
# python custom.py D:/DeepConvNet/pre-processed/KU_mi_smt.h5 D:/braindecode/baseline_models D:/braindecode/results -scheme 5 -trfrate 10 -subj 1


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# formatter = logging.Formatter(log_fmt)
# logger.handlers[0].setFormatter(formatter)


# ---- Load config.yaml ----
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["experiment"]

# ---- Argparse setup ----
parser = argparse.ArgumentParser(
    description="Subject-adaptative classification with KU Data"
)

parser.add_argument("--meta", help="Training Mode", action="store_true")
parser.add_argument("--datapath", type=str, help="Path to the h5 data file")
parser.add_argument("--outpath", type=str, help="Path to the result folder")
parser.add_argument("--gpu", type=int, help="The gpu device to use")
parser.add_argument("--n_folds", type=int, help="n LOSO folds")
parser.add_argument("--kfold", type=int, help="Number of folds")
parser.add_argument(
    "--show_timing", help="Show timing of functions", action="store_true"
)
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--train_epoch", type=int, help="Training epochs")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--weight_decay", type=float, help="Weight decay")

parser.add_argument("--meta_lr", type=float, help="Meta learning rate")
parser.add_argument(
    "--n_tasks_per_meta_batch", type=int, help="Number of tasks per meta batch"
)
parser.add_argument("--inner_steps", type=int, help="Number of inner steps")

parser.add_argument("--log_timing", help="Log timing of functions", action="store_true")
parser.add_argument(
    "--run_after_early_stop", help="Run after early stop", action="store_true"
)


args = parser.parse_args()

# ---- Merge CLI args with config ----
# If CLI arg is None / False, fallback to config
final = {
    "meta": args.meta if args.meta else config["meta"],
    "datapath": args.datapath or config["datapath"],
    "outpath": args.outpath or config["outpath"],
    "gpu": args.gpu if args.gpu is not None else config["gpu"],
    "n_folds": args.n_folds or config["n_folds"],
    "batch_size": args.batch_size or config["batch_size"],
    "train_epoch": args.train_epoch or config["train_epoch"],
    "lr": args.lr or config["learning_rate"],
    "weight_decay": args.weight_decay or config["weight_decay"],
    "meta_lr": args.meta_lr or config["meta_learning_rate"],
    "n_tasks_per_meta_batch": args.n_tasks_per_meta_batch
    or config["n_tasks_per_meta_batch"],
    "inner_steps": args.inner_steps or config["inner_steps"],
    "kfold": args.kfold or config["kfold"],
    "log_timing": args.log_timing or config["log_timing"],
    "scheduler": config["scheduler"],
    "run_after_early_stop": args.run_after_early_stop or config["run_after_early_stop"],
}


BATCH_SIZE = final["batch_size"]
TRAIN_EPOCH = final["train_epoch"]
LR = final["lr"]
WEIGHT_DECAY = final["weight_decay"]
SCHEDULER = final["scheduler"]

META_LR = final["meta_lr"]
N_TASKS_PER_META_BATCH = final["n_tasks_per_meta_batch"]
N_INNER_STEPS = final["inner_steps"]

run_after_early_stop = final["run_after_early_stop"]
meta = final["meta"]
log_timing = final["log_timing"]
n_folds = final["n_folds"]  # number of subjects for training and testing LOSO
kfold = final["kfold"]  # number of folds for cross-validation

# Randomly shuffled subject.
subjs = [
    35,
    47,
    46,
    37,
    13,
    27,
    12,
    32,
    53,
    54,
    4,
    40,
    19,
    41,
    18,
    42,
    34,
    7,
    49,
    9,
    5,
    48,
    29,
    15,
    21,
    17,
    31,
    45,
    1,
    38,
    51,
    8,
    11,
    16,
    28,
    44,
    24,
    52,
    3,
    26,
    39,
    50,
    6,
    23,
    2,
    14,
    25,
    20,
    10,
    33,
    22,
    43,
    36,
    30,
]


# ---- Now use `final` dict everywhere ----
dfile = h5py.File(final["datapath"], "r")
experiment_dir = final["outpath"]
os.makedirs(experiment_dir, exist_ok=True)
# make a subfolder from hyperparameter values and date
subfolder = (
    f"experiment_epochs_NO_CV_{TRAIN_EPOCH}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)
experiment_dir = os.path.join(experiment_dir, subfolder)
os.makedirs(experiment_dir, exist_ok=True)


# save the config
with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
    yaml.dump(final, f)

torch.cuda.set_device(final["gpu"])
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True


# print all the arguments
print(final)


# Get data from single subject.
def get_data(subj):
    dpath = "s" + str(subj)
    X = dfile[dpath]["X"]
    Y = dfile[dpath]["Y"]
    return X, Y


def get_multi_data(subjs):
    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(s)
        Xs.append(x[:])
        Ys.append(y[:])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


for loso_fold in range(n_folds):

    outpath = os.path.join(experiment_dir, f"fold_{loso_fold}")
    os.makedirs(outpath, exist_ok=True)
    print(f"outpath to save: {outpath}")

    test_subj = subjs[loso_fold]
    cv_set = np.array(subjs[loso_fold + 1 :] + subjs[:loso_fold])

    # get 90% of the data for training and 10% for validation
    train_subjs = cv_set[: int(len(cv_set) * 0.9)]
    valid_subjs = cv_set[int(len(cv_set) * 0.9) :]
    X_train, Y_train = get_multi_data(train_subjs)
    X_val, Y_val = get_multi_data(valid_subjs)
    X_test, Y_test = get_data(test_subj)
    train_set = SignalAndTarget(X_train, y=Y_train)
    valid_set = SignalAndTarget(X_val, y=Y_val)
    test_set = SignalAndTarget(X_test[300:], y=Y_test[300:])  # TODO: why 300?

    print("=" * 40)
    print(f"LOSO Fold {loso_fold+1} of {n_folds}")
    print(f"Test subject: {test_subj}")
    print(f"Train subjects: {train_subjs}")
    print(f"Validation subjects: {valid_subjs}")
    print("=" * 40)

    n_classes = 2
    in_chans = train_set.X.shape[1]

    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Deep5Net(
        in_chans=in_chans,
        n_classes=n_classes,
        input_time_length=train_set.X.shape[2],
        final_conv_length="auto",
    ).cuda()

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.compile(
        loss=F.nll_loss,
        optimizer=optimizer,
        iterator_seed=1,
    )

    # Fit the base model for transfer learning, use early stopping as a hack to remember the model
    exp = model.fit(
        train_set.X,
        train_set.y,
        epochs=TRAIN_EPOCH,
        batch_size=BATCH_SIZE,
        scheduler=SCHEDULER,
        validation_data=(valid_set.X, valid_set.y),
        remember_best_column="valid_loss",
        meta=meta,
        inner_lr=META_LR,
        log_timing=log_timing,
        n_tasks_per_meta_batch=N_TASKS_PER_META_BATCH,
        inner_steps=N_INNER_STEPS,
        fold_info={"loso_fold": loso_fold, "cv_fold": 0},
        run_after_early_stop=run_after_early_stop,
    )

    rememberer = exp.rememberer
    base_model_param = {
        "epoch": rememberer.best_epoch,
        "model_state_dict": rememberer.model_state_dict,
        "optimizer_state_dict": rememberer.optimizer_state_dict,
        "loss": rememberer.lowest_val,
    }
    torch.save(
        base_model_param,
        pjoin(outpath, "model_f{}_cv{}.pt".format(loso_fold, 0)),
    )
    model.epochs_df.to_csv(
        pjoin(outpath, "original_epochs_f{}_cv{}.csv".format(loso_fold, 0))
    )

    test_loss = model.evaluate(test_set.X, test_set.y)
    with open(
        pjoin(
            outpath,
            "original_test_base_s{}_f{}_no_CV.json".format(test_subj, loso_fold),
        ),
        "w",
    ) as f:
        json.dump(test_loss, f)
