import json, numpy as np, pathlib as pl

# ── Dateien angeben ───────────────────────────────────────────
GT_FILE  = pl.Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
                   "projects/BEVFusion/demo/Ivo/ground_truth_EgoFzg/"
                   "ground_truth_scene-0161_1.json")

DET_FILE = pl.Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
                   "projects/BEVFusion/demo/Ivo/ResultsMain/Results4_mainPkl/"
                   "results_scene-0161_1.json")

# ── JSON laden ────────────────────────────────────────────────
gt  = json.load(GT_FILE.open())
det = json.load(DET_FILE.open())

# ── Mittelpunkt der ersten Box vergleichen ───────────────────
g = np.array(gt[0]["position"])          # Ground-Truth
d = np.array(det["bboxes"][0][:3])       # Detection

dist = np.linalg.norm(g - d)
print(f"Entfernung erster GT ↔ Det: {dist:.2f} m")