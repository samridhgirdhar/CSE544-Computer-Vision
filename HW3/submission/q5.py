
import os, sys, glob, argparse, shutil, cv2, numpy as np, torch
from types import SimpleNamespace

# ------- user editable paths --------------------------------------------------
ROOT_IMG_DIR = "datasets/my_two_image_folders"   # <‑‑ EDIT
OUTPUT_DIR   = "outputs/matcher_vis"             # <‑‑ EDIT
# ------------------------------------------------------------------------------

# 1) minimal set of Matcher imports (assumes script lives in repo root)
sys.path.append("./")
from matcher.Matcher import build_matcher_oss

def overlay_mask(img, mask, alpha=0.6):
    """RGBA overlay for pretty visualisation (green mask)."""
    overlay        = img.copy()
    overlay[mask]  = (0,255,0)                   # green
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# 2) build a ready‑to‑use matcher instance (no training!)
def get_matcher(device="cuda"):
    args = SimpleNamespace(
        device            = torch.device(device if torch.cuda.is_available() else "cpu"),
        # weight locations — keep default filenames
        dinov2_weights    = "models/dinov2_vitl14_pretrain.pth",
        sam_weights       = "models/sam_vit_h_4b8939.pth",
        dinov2_size       = "vit_large",
        sam_size          = "vit_h",
        use_semantic_sam  = False,               # flip to True if you want part masks
        # run‑time knobs (leave as defaults for simple images)
        points_per_side   = 64,
        pred_iou_thresh   = 0.88,
        sel_stability_score_thresh = 0.90,
        iou_filter        = 0.0,
        box_nms_thresh    = 1.0,
        output_layer      = 3,
        use_dense_mask    = 0,
        multimask_output  = 0,
        num_centers       = 8,
        use_box           = False,
        use_points_or_centers = True,
        sample_range      = (1,6),
        max_sample_iterations = 64,
        alpha=1.0, beta=0.0, exp=0.0,
        emd_filter=0.0, purity_filter=0.0, coverage_filter=0.0,
        use_score_filter=False, deep_score_filter=0.33,
        deep_score_norm_filter=0.10, topk_scores_threshold=0.0,
        num_merging_mask = 9,
    )
    return build_matcher_oss(args)

matcher = get_matcher()

# 3) walk through folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
for folder in sorted(os.listdir(ROOT_IMG_DIR)):
    fpath = os.path.join(ROOT_IMG_DIR, folder)
    if not os.path.isdir(fpath): continue

    imgs = sorted(glob.glob(os.path.join(fpath, "*.*g")))   # jpg / png
    assert len(imgs) == 2, f"Folder {folder} must contain exactly two images"
    ref_img_p, tgt_img_p = imgs                              # (A,B)

    for direction, (ref_p, tgt_p) in enumerate([(ref_img_p, tgt_img_p),
                                                (tgt_img_p, ref_img_p)]):
        ref = cv2.cvtColor(cv2.imread(ref_p), cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(cv2.imread(tgt_p), cv2.COLOR_BGR2RGB)

        # ---- 1. build masks for the reference image automatically ------------
        # SamAutomaticMaskGenerator (inside Matcher) returns K candidate masks.
        # We use the **largest** area mask as reference; this works well for the
        # “simple” images described in the assignment.
        masks_ref = matcher.generator.generate(ref)
        ref_mask  = max(masks_ref, key=lambda m: m["area"])["segmentation"]

        # ---- 2. feed into Matcher -------------------------------------------
        matcher.set_reference(torch.from_numpy(ref).permute(2,0,1).unsqueeze(0),
                              torch.from_numpy(ref_mask[None, None, ...]))  # 1x1xH xW
        matcher.set_target(torch.from_numpy(tgt).permute(2,0,1).unsqueeze(0))
        with torch.no_grad():
            pred_mask = matcher.predict().squeeze(0).bool().cpu().numpy()

        matcher.clear()     # free CUDA memory for next pair

        # ---- 3. save visualisations & raw mask ------------------------------
        out_sub   = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(out_sub, exist_ok=True)

        # 3.a pretty overlay
        vis = overlay_mask(cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR), pred_mask)
        cv2.imwrite(os.path.join(out_sub,
                    f"seg_{direction+1}_{os.path.basename(tgt_p)}"), vis)

        # 3.b binary mask for quantitative use
        np.save(os.path.join(out_sub,
                 f"mask_{direction+1}_{os.path.splitext(os.path.basename(tgt_p))[0]}.npy"),
                pred_mask.astype(np.uint8))

        print(f"[✓] {folder}: {os.path.basename(ref_p)} → {os.path.basename(tgt_p)}")

print(f"\nAll done!  Results live under:  {OUTPUT_DIR}")



# #!/usr/bin/env python3
# """
# one_shot_box_segment.py

# For each two‑image folder under ROOT:
#   - A→B: use A’s largest SAM mask to prompt a box on B → save seg_1_B, mask_1_B
#   - B→A: same in reverse         → save seg_2_A, mask_2_A
# """

# import os
# import cv2
# import numpy as np
# import torch
# from pathlib import Path
# from tqdm import tqdm

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# # ── USER CONFIG ───────────────────────────────────
# ROOT     = Path("Images")               # parent folder of subfolders
# OUTDIR   = Path("outputs/sam_box")      # where to save results
# SAM_W    = "models/sam_vit_h_4b8939.pth"       # SAM weights
# IMG_SIDE = 518                          # SAM’s fixed input size
# # ──────────────────────────────────────────────────

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# # 1) load SAM
# sam = sam_model_registry["vit_h"](checkpoint=SAM_W).to(device)
# sam.eval()

# # Automatic mask generator (for reference)
# auto_gen = SamAutomaticMaskGenerator(sam,
#     points_per_side=64,
#     pred_iou_thresh=0.88,
#     stability_score_thresh=0.95
# )

# # Predictor (for box prompt on target)
# predictor = SamPredictor(sam)

# def get_largest_box(rgb_uint8: np.ndarray):
#     """Return bounding box [x0,y0,x1,y1] of SAM’s largest mask."""
#     masks = auto_gen.generate(rgb_uint8)
#     seg  = max(masks, key=lambda m: m["area"])["segmentation"]  # H×W bool
#     ys, xs = np.where(seg)
#     return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

# def segment_with_box(rgb_uint8: np.ndarray, box: list):
#     """
#     Prompt SAM with box→ returns a boolean mask H×W.
#     rgb_uint8 must be original image resized to IMG_SIDE.
#     """
#     predictor.set_image(rgb_uint8)  
#     masks, _, _ = predictor.predict(
#         box=np.array(box)[None, :],       # 1×4
#         multimask_output=False
#     )
#     return masks[0]  # H×W bool

# def overlay(bgr, mask, alpha=0.6):
#     o = bgr.copy()
#     o[mask] = (0,255,0)
#     return cv2.addWeighted(o, alpha, bgr, 1-alpha, 0)

# # make output dirs
# OUTDIR.mkdir(parents=True, exist_ok=True)



# mask_gen = SamAutomaticMaskGenerator(
#     sam,
#     points_per_side=16,        # speed vs quality trade‑off
#     pred_iou_thresh=0.88,
#     stability_score_thresh=0.95
# )


# for sub in tqdm(sorted(p for p in ROOT.iterdir() if p.is_dir())):
#     imgs = sorted(sub.glob("*.*g"))
#     assert len(imgs) == 2, f"{sub} must have exactly two images"
#     A, B = imgs
#     dest = OUTDIR / sub.name
#     dest.mkdir(exist_ok=True)

#     for idx, (ref_p, tgt_p) in enumerate([(A,B),(B,A)], start=1):
#         # load & resize (uint8)
#         ref0 = cv2.cvtColor(cv2.imread(str(ref_p)), cv2.COLOR_BGR2RGB)
#         tgt0 = cv2.cvtColor(cv2.imread(str(tgt_p)), cv2.COLOR_BGR2RGB)
#         ref  = cv2.resize(ref0, (IMG_SIDE, IMG_SIDE), interpolation=cv2.INTER_AREA)
#         tgt  = cv2.resize(tgt0, (IMG_SIDE, IMG_SIDE), interpolation=cv2.INTER_AREA)

#         # 1) box from ref’s largest mask
#         box = get_largest_box(ref)

#         # 2) segment on target with that box
#         mask = segment_with_box(tgt, box)

#         # 3) save overlay + raw mask
#         vis = overlay(
#             cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR),
#             mask
#         )
#         seg_name  = dest / f"seg_{idx}_{tgt_p.name}"
#         mask_name = dest / f"mask_{idx}_{tgt_p.stem}.npy"
#         cv2.imwrite(str(seg_name), vis)
#         np.save(str(mask_name), mask.astype(np.uint8))

#     print(f"[✓] {sub.name}")

# print(f"\nDone! Results in: {OUTDIR}")
