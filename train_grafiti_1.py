import argparse
import sys
import time
from random import SystemRandom

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=300,    type=int,   help="maximum epochs")
parser.add_argument("-es",  "--early-stop",  default=30,    type=int,   help="early stop patience")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-bs", "--batch-size",   default=32,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size",  default=32,    type=int,   help="hidden-size")
parser.add_argument("-ki", "--kernel-init",  default="skew-symmetric",   help="kernel-inititialization")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=1,   type=int,   help="Set the random seed.")
parser.add_argument("-nl",  "--nlayers", default=2,   type=int,   help="")
parser.add_argument("-ahd",  "--attn-head", default=2,   type=int,   help="")
parser.add_argument("-ldim",  "--latent-dim", default=128,   type=int,   help="")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-fh", "--forc-time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ot", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-ec", "--extra_channels", default=1, type=int, help="#additional fully observed channels")


import pdb

# fmt: on


ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
model_path = "saved_models/" + ARGS.dataset + "_" + str(experiment_id) + ".h5"

print(ARGS, experiment_id)

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

import logging
import os
import pdb
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    # "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

if ARGS.dataset == "ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019

    TASK = USHCN_DeBrouwer2019(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    TASK = MIMIC_III_DeBrouwer2019(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )

elif ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    TASK = MIMIC_IV_Bilos2021(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "p12":
    from tsdm.tasks.physionet2012 import Physionet2012

    TASK = Physionet2012(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )

from gratif.gratif import tsdm_collate

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": tsdm_collate,
}

dloader_config_infer = {
    "batch_size": 64,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate,
}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask]) ** 2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask * torch.abs(y - yhat), 1) / (torch.sum(mask, 1))
    return torch.mean(err)


def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gratif.gratif import GrATiF

MODEL_CONFIG = {
    "input_dim": TASK.dataset.shape[-1],
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "extra_channels":ARGS.extra_channels,
    "device": DEVICE,
}

MODEL = GrATiF(**MODEL_CONFIG).to(DEVICE)
torchinfo.summary(MODEL)


def predict_fn(model, batch) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY)
    return target_U_, output.squeeze(), target_mask_


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT, MASK = predict_fn(MODEL, batch)
# Backward
R = LOSS(Y, YHAT, MASK)
assert torch.isfinite(R).item(), "Model Collapsed!"
# R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)

# ## Initialize Optimizer

from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=10, factor=0.5, min_lr=0.00001, verbose=True
)
ovr_start_time = time.time()
es = False
best_val_loss = 10e8
total_num_batches = 0
epoch_times = []
N_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print(N_params)
for epoch in range(1, ARGS.epochs + 1):
    loss_list = []
    start_time = time.time()
    for batch in TRAIN_LOADER:
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()
    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))
    loss_list = []
    count = 0
    with torch.no_grad():
        for batch in VALID_LOADER:
            total_num_batches += 1
            # Forward
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            if R.isnan():
                pdb.set_trace()
            loss_list.append([R * MASK.sum()])
            count += MASK.sum()
    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
    epoch_times.append(epoch_time - start_time)
    print(
        f"{epoch:3.0f} Train:{train_loss.item():4.4f}  VAL: {val_loss.item():4.4f}   EPOCH_TIME: {epoch_times[-1]:3.2f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "args": ARGS,
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": train_loss,
            },
            "saved_models/" + ARGS.dataset + "_" + str(experiment_id) + ".h5",
        )
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == ARGS.early_stop:
        print(
            f"Early stopping because of no improvement in val. metric for {ARGS.early_stop} epochs"
        )
        es = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (es == True):
        print(f"tot_train_time: {time.time()- ovr_start_time}")
        chp = torch.load(model_path)
        MODEL.load_state_dict(chp["state_dict"])
        loss_list = []
        count = 0
        with torch.no_grad():
            inf_start = time.time()
            for batch in TEST_LOADER:
                total_num_batches += 1
                # Forward
                Y, YHAT, MASK = predict_fn(MODEL, batch)
                R = LOSS(Y, YHAT, MASK)
                assert torch.isfinite(R).item(), "Model Collapsed!"
                # loss_list.append([R*Y.shape[0]])
                loss_list.append([R * MASK.sum()])
                count += MASK.sum()
        print(f"inference_time: {time.time() - inf_start}")
        print(f"avg_epoch_time: {np.mean(epoch_times)}")
        test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
        print(f"best_val_loss: {best_val_loss.item()},  test_loss: {test_loss.item()}")
        break