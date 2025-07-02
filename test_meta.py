#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger

from models.model_stdc     import STDC
from models.model_bisenetv2 import BiSeNetv2
from models.model_bisenet   import BiSeNet

from face_dataset_with_h import FaceMask
from loss       import *
from evaluate   import evaluate
from optimizer  import Optimizer

import cv2, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, os.path as osp, logging, time, datetime, argparse, csv, sys   # ← sys added
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


# --------------------------------------------------------------------------- #
# argument parser
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--local_rank',    type=int,   default=-1)
    p.add_argument('--save_interval', type=int,   default=5_000)
    p.add_argument('--eval_interval', type=int,   default=1_000)
    p.add_argument('--model',         type=str,   required=True,
                   choices=['stdc', 'bisenetv2'],
                   help="Backbone to train")
    p.add_argument('--experiment',    type=str,   default='default_exp',
                   help='Short tag used for result directory')
    p.add_argument('-m', '--message', type=str,   default='',
                   help='Optional description of this run (shows up in run_info.txt)')
    p.add_argument('--resume',        type=str,   default='',
                   help='Resume / fine-tune from checkpoint')
    p.add_argument('--data', choices=['infolks', 'labellerr'],
                   default='labellerr',
                   help="'infolks' → 60 k iters • 'labellerr' → 100 k iters")
    return p.parse_args()



ERROR_LIST: list[str] = []
def log_bad_batch(names, exc, log_dir) -> None:
    """Append failing-batch info to <exp_root>/logs/bad_batches.log"""
    ERROR_LIST.extend(names)
    path = os.path.join(log_dir, "bad_batches.log")
    with open(path, "a") as f:
        f.write(f"{datetime.datetime.now()}: {', '.join(names)}\n")
        f.write(f"    {repr(exc)}\n")


# --------------------------------------------------------------------------- #
# Helper functions (unchanged except for path handling)
# --------------------------------------------------------------------------- #
def configure_logging(log_root: str) -> str:
    """Create <root>/logs & return CSV-metrics file path"""
    logs_dir = osp.join(log_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_fp = osp.join(logs_dir,
                      f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    fh = logging.FileHandler(log_fp)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(fh)

    csv_fp = osp.join(logs_dir,
                      f"metrics_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(csv_fp, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['Iteration', 'Epoch', 'Loss', 'LR', 'mAP', 'mIoU'])
    return csv_fp


def log_metrics(csv_fp, *row) -> None:
    with open(csv_fp, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def save_checkpoint(net, optim, ckpt_dir,
                    epoch, it, loss, is_best=False) -> None:
    payload = {
        'epoch': epoch,
        'iteration': it,
        'model': net.module.state_dict()
                 if hasattr(net, 'module') else net.state_dict(),
        'optim': optim.optim.state_dict(),
        'optim_it': optim.it,
        'loss': loss,
    }
    ckpt_fp = osp.join(ckpt_dir, f'ckpt_{it + 1}.pth')
    torch.save(payload, ckpt_fp)
    if is_best:
        torch.save(payload, osp.join(ckpt_dir, 'best_model.pth'))


# ---------------------------------------------------------------------------
# load_checkpoint – works for wrapped & bare .pth files        

def load_checkpoint(net, optim, ckpt_fp):
    """
    Load *both* training checkpoints and plain state_dict files.

    Returns
    -------
    epoch : int
    iteration : int
    best_loss : float
    """
    ckpt = torch.load(ckpt_fp, map_location="cpu")

    # ---------- 1) try the usual wrapped formats ----------------------------
    for key in ("model", "state_dict", "net", "model_state", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            epoch      = ckpt.get("epoch", 0)
            iteration  = ckpt.get("iteration", 0)
            best_loss  = ckpt.get("loss", float("inf"))
            break
    else:
        # ---------- 2) assume the whole file *is* a state_dict ---------------
        if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
            state_dict = ckpt
            epoch = iteration = 0
            best_loss = float("inf")
            logging.info(
                f"Loaded bare state_dict from {ckpt_fp}; "
                f"starting fine-tune at epoch 0 / iter 0."
            )
        else:                              # anything else → give a clear error
            raise KeyError(
                f"Could not find a model state-dict in {ckpt_fp}. "
                f"Top-level keys are: {list(ckpt.keys())}"
            )

    # ---------- 3) fix DataParallel ↔ single-GPU prefix ---------------------
    has_module = any(k.startswith("module.") for k in state_dict)
    wants_module = hasattr(net, "module")
    if has_module and not wants_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module and wants_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    # ---------- 4) load into the network ------------------------------------
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing weights: {missing[:5]}…")
    if unexpected:
        logging.warning(f"Unexpected weights: {unexpected[:5]}…")

    # ---------- 5) restore optimiser (if present) ---------------------------
    if optim is not None and isinstance(ckpt, dict) and "optim" in ckpt:
        optim.optim.load_state_dict(ckpt["optim"])
        optim.it = ckpt.get("optim_it", 0)

    return epoch, iteration, best_loss



# --------------------------------------------------------------------------- #
# main training routine
# --------------------------------------------------------------------------- #
def train() -> None:
    args = parse_args()
    torch.cuda.set_device(args.local_rank)

    run_start = datetime.datetime.now()                     # ← start stamp

    # --------------------------------------------------------------------- #
    # per-experiment folder hierarchy
    # --------------------------------------------------------------------- #
    res_root = osp.join('./res_distributed', args.experiment)
    ckpt_dir = osp.join(res_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    setup_logger(res_root)
    metrics_csv = configure_logging(res_root)
    logs_dir    = osp.join(res_root, 'logs')

    # --------------------- dataset / dataloader -------------------------- #
    n_classes      = 11
    n_img_per_gpu  = 16
    n_workers      = 8
    cropsize       = [512, 512]
    data_root      = '/workspace/bisenet_training/nilay_png_annotation_data/labellerr_dataset'
    aux_w          = 0.4
    n_iter         = 100_000

    if args.data == 'infolks':
        n_iter    = 60_000
        data_root = '/workspace/bisenet_training/nilay_png_annotation_data/infolks_dataset'

    ds   = FaceMask(data_root, cropsize=cropsize, mode='train')
    dl   = DataLoader(ds, batch_size=n_img_per_gpu, shuffle=True,
                      num_workers=n_workers, pin_memory=True, drop_last=True)
    val_ds = FaceMask(data_root, cropsize=cropsize, mode='val')
    val_dl = DataLoader(val_ds, batch_size=n_img_per_gpu, shuffle=False,
                        num_workers=n_workers, pin_memory=True, drop_last=False)

    # --------------------------- model ---------------------------------- #
    if args.model == "stdc":
        net = STDC(n_classes=n_classes).cuda().train()
    elif args.model == "bisenetv2":
        net = BiSeNetv2(n_classes=n_classes).cuda().train()

    ignore_idx  = -100
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    crit_main = SoftmaxFocalLoss(gamma=0.3)
    crit_aux  = SoftmaxFocalLoss(gamma=0.3)

    # --------------------------- optimiser ------------------------------ #
    optim = Optimizer(model=net, lr0=1e-2, momentum=0.9, wd=5e-4,
                      warmup_steps=1_000, warmup_start_lr=1e-5,
                      max_iter=n_iter, power=0.9)

    # ---------------------- resume / fine-tune -------------------------- #
    start_epoch = start_it = 0
    best_loss   = float('inf')
    if args.resume:
        if not osp.isfile(args.resume):
            raise FileNotFoundError(f'Checkpoint {args.resume} not found')
        start_epoch, start_it, best_loss = load_checkpoint(
            net, optim, args.resume)
        logging.info(f'Resumed from {args.resume} at '
                     f'epoch {start_epoch}, iter {start_it}')

    # ------------------------- training loop --------------------------- #
    loss_buf, pbar = [], tqdm(total=n_iter - start_it, initial=start_it,
                              desc=f'[Exp: {args.experiment}]')
    diter = iter(dl)
    epoch = start_epoch

    for it in range(start_it, n_iter):
        try:
            im, lb, names = next(diter)
            if im.size(0) != n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb, names = next(diter)

        im, lb = im.cuda(), lb.cuda().squeeze(1)
        optim.zero_grad()
        try:
            main, auxs = net(im)
            loss = crit_main(main, lb)
            for aux_pred in auxs:
                loss = loss + aux_w * crit_aux(aux_pred, lb)
            loss.backward()
            optim.step()
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            log_bad_batch(names, exc, logs_dir)
            pbar.update(1)
            continue

        # progress / logging ---------------------------------------------
        loss_val = loss.item()
        loss_buf.append(loss_val)
        pbar.update(1);   pbar.set_postfix(loss=f'{loss_val:.3f}', epoch=epoch)
        log_metrics(metrics_csv, it + 1, epoch, loss_val, optim.lr)

        # checkpointing --------------------------------------------------
        if (it + 1) % args.save_interval == 0:
            save_checkpoint(net, optim, ckpt_dir,
                            epoch, it, loss_val,
                            is_best=loss_val < best_loss)
            best_loss = min(best_loss, loss_val)

    pbar.close()
    logger.info(f"training done – skipped {len(ERROR_LIST)} batches "
                f"({len(set(ERROR_LIST))} unique files). "
                "See logs/bad_batches.log for details.")

    # final save ------------------------------------------------------------
    final_pth = osp.join(res_root, f'{args.experiment}_final.pth')
    torch.save(net.state_dict(), final_pth)
    logging.info(f'Training complete – model saved to {final_pth}.')

    # --------------------------------------------------------------------- #
    # write metadata file for the run
    # --------------------------------------------------------------------- #
    run_end = datetime.datetime.now()
    meta_fp = osp.join(res_root, 'run_info.txt')
    with open(meta_fp, 'w', encoding='utf-8') as f:
        f.write(f"Experiment : {args.experiment}\n")
        f.write(f"Description: {args.message or '(none given)'}\n")
        f.write(f"Command    : {' '.join(sys.argv)}\n")
        f.write(f"Started    : {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Finished   : {run_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logging.info(f'Metadata written to {meta_fp}')


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    train()
