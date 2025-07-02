#!/usr/bin/python
# -*- coding: utf-8 -*-

from logger import setup_logger

from models.model_bisenet import BiSeNet
from models.model_bisenetv2 import BiSeNetv2
from models.model_stdc import STDC

import os
import os.path as osp
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from ultralytics.utils.plotting import Colors

# --------------------------------------------------------------------------- #
# -------------------------  VISUALISATION UTILITIES  ----------------------- #
# --------------------------------------------------------------------------- #
COLORS = Colors()                                     # pretty  palette

def vis_parsing_maps(
    im: Image.Image,
    parsing_anno: np.ndarray,
    stride: int,
    save_path: Path,
) -> None:
    """
    Overlay *parsing_anno* on *im* and save as JPEG to *save_path*.
    """
    part_colors = [list(COLORS(i)) for i in range(11)]

    im_arr = np.array(im)
    vis_map = cv2.resize(
        parsing_anno.astype(np.uint8), None,
        fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST,
    )
    colour_map = np.full((*vis_map.shape, 3), 255, dtype=np.uint8)

    for lbl in range(1, vis_map.max() + 1):
        idx = np.where(vis_map == lbl)
        colour_map[idx[0], idx[1], :] = part_colors[lbl]

    blended = cv2.addWeighted(
        cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR), 0.4, colour_map, 0.6, 0
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path),
                blended,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# --------------------------------------------------------------------------- #
# ------------------------------  CORE PIPELINE  ---------------------------- #
# --------------------------------------------------------------------------- #



# def load_model(checkpoint: Path, n_classes: int = 10) -> BiSeNet:
#     """
#     Instantiate BiSeNet and load weights.
#     """
#     net = BiSeNet(n_classes=n_classes).cuda()
#     net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
#     net.eval()
#     return net

# --------------------------------------------------------------------------- #
# -----------------------  smarter checkpoint loader  ----------------------- #
# --------------------------------------------------------------------------- #
def load_model(checkpoint: Path, model_cls, n_classes: int = 11) -> BiSeNet:
    """
    Instantiate BiSeNet and load weights from *checkpoint*.

    The file may be
    1) a wrapped training checkpoint
       (dict with 'model' or 'state_dict' plus metadata), or
    2) a bare state-dict produced by `torch.save(net.state_dict(), …)`.
    """
    net = model_cls(n_classes=n_classes).cuda()

    ckpt = torch.load(checkpoint, map_location="cpu")

    # --- unwrap if necessary -------------------------------------------------
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            # assume the dict itself is already the state_dict
            state_dict = ckpt
    else:  # unlikely, but handle torch.save(..., _use_new_zipfile_serialization=False)
        state_dict = ckpt

    # --- deal with DataParallel / DDP prefixes ------------------------------
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    wants_module      = hasattr(net, "module")          # False here

    if has_module_prefix and not wants_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and wants_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    # --- load (allowing size-mismatched aux heads, etc.) ---------------------
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] ⚠️  Missing {len(missing)} params (aux heads?): "
              f"{missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[load_model] ⚠️  Unexpected {len(unexpected)} params: "
              f"{unexpected[:5]}{'…' if len(unexpected)>5 else ''}")

    net.eval()
    return net



def valid_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def process_image(
    img_path: Path,
    save_path: Path,
    net: BiSeNet,
    tf,
    img_size: int = 512,
) -> None:
    """
    Run BiSeNet on a single image and write colour overlay.
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_res = img_pil.resize((img_size, img_size), Image.BILINEAR)

    with torch.no_grad():
        inp = tf(img_res).unsqueeze(0).cuda()
        parsing = net(inp)[0].squeeze(0).cpu().numpy().argmax(0)

    vis_parsing_maps(img_res, parsing, stride=1, save_path=save_path)


def walk_and_process(
    input_root: Path,
    output_root: Path,
    net: BiSeNet,
    tf,
) -> None:
    """
    Recursively traverse *input_root*,
    skipping directories prefixed with 'body_parsed',
    and process each image while preserving the directory layout
    under *output_root*.
    """
    out_abs = output_root.resolve()          # compute once

    for dirpath, dirnames, filenames in os.walk(input_root):
        # Prune dirnames *in place* so os.walk doesn’t descend into them
        pruned = []
        for d in dirnames:
            # Skip unwanted prefixes
            if d.startswith(("body_parsed", ".ipynb_checkpoints")):
                continue

            full = (Path(dirpath) / d)       # ← build from *dirpath*, not input_root
            try:
                if full.resolve().samefile(out_abs):
                    # This is the output directory – don’t walk into it
                    continue
            except FileNotFoundError:
                # Path vanished or is a broken symlink – just skip it
                continue

            pruned.append(d)

        dirnames[:] = pruned                 # modify in place

        # ---- Process files --------------------------------------------------
        for fname in filenames:
            if not valid_image(fname):
                continue
            src  = Path(dirpath) / fname
            dest = output_root / src.relative_to(input_root)
            process_image(src, dest, net, tf)


"""if __name__ == "__main__":
    
    setup_logger("./")

    INPUT_DIR  = Path("/workspace/bisenet_training/sp_small_bp_xl/retouching-data")
    OUTPUT_DIR = Path("/workspace/bisenet_training/sp_small_bp_xl/bisenet_results_without_h_14_May")
    CKPT       = Path("res_distributed/isolated_dilated_without_h_flip_model.pth")   # checkpoint

    # --- model & preprocessing ------------------------------------------------
    net = load_model(CKPT)
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # --- run ------------------------------------------------------------------
    print(f"Scanning {INPUT_DIR} …")
    walk_and_process(INPUT_DIR, OUTPUT_DIR, net, tf)
    print(f"✅ Done! Results saved to: {OUTPUT_DIR}")"""


# --------------------------------------------------------------------------- #
# ----------------------------------  MAIN  --------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Run BiSeNet inference on every image under INPUT_DIR and write colour
            overlays to an OUTPUT_DIR that mirrors the same folder structure.

            Examples
            --------
            ● With explicit paths
                python test_directory.py \
                    --input  /data/images \
                    --output /data/bisenet_vis \
                    --ckpt   weights/bisenet_faces.pth

            ● Let output default to a sibling folder called <input>_vis
                python infer_bisenet.py --input /data/images --ckpt weights/best.pth
        """),
    )
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Root directory to recursively scan for images")
    parser.add_argument("--output", "-o", type=Path,
                        help="Destination root for visualisations "
                             "(defaults to <input>_vis)")
    parser.add_argument("--ckpt",   "-c", type=Path, required=True,
                        help="Path to the BiSeNet checkpoint (.pth)")
    parser.add_argument("--model", type=str, required=True, help="Model class to test")

    args = parser.parse_args()
    input_dir  = args.input.expanduser().resolve()
    output_dir = (args.output or (input_dir.parent / f"{input_dir.name}_vis")) \
                    .expanduser().resolve()
    ckpt_path  = args.ckpt.expanduser().resolve()
    
    os.makedirs(args.output, exist_ok=True)

    # ---------- logging ------------------------------------------------------
    setup_logger(str(output_dir))
    print(f"Model checkpoint : {ckpt_path}")
    print(f"Input directory  : {input_dir}")
    print(f"Output directory : {output_dir}\n")

    # ---------- model & preprocessing ---------------------------------------
    model_cls = STDC
    if args.model == "bisenet":
        model_cls = BiSeNet
    elif args.model == "bisenetv2":
        model_cls = BiSeNetv2
    elif args.model == "stdc":
        model_cls = STDC
        
    net = load_model(ckpt_path, model_cls)
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # ---------- run ----------------------------------------------------------
    print(f"Scanning {input_dir} … (this may take a while)")
    walk_and_process(input_dir, output_dir, net, tf)
    print(f"\n✅  Done! Results saved to: {output_dir}")
