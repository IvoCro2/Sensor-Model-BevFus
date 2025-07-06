#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hungarian-Matching ▸ Ground-Truth + Detektionen → Matches

Erwartete Dateinamen
────────────────────
GT :  ground_truth_scene-XXXX_Y.json          (LiDAR-Frame)
DET:  results_scene-XXXX_Y.json               (bereits umbenannt)
OUT:  matches_scene-XXXX_Y.json

Das Skript durchläuft zwei Detection-Ordner (train + val),
matched jede vorhandene results_scene-Datei mit der passenden
Ground-Truth-Datei und speichert eine Matches-Datei
mit demselben Scene-Part in OUT_DIR.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# ───────────── Pfade anpassen ─────────────
GT_DIR = Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
              "projects/BEVFusion/demo/Ivo/ground_truth_EgoFzg")

DET_DIRS = [
    Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
         "projects/BEVFusion/demo/Ivo/ResultsMain/Results4_mainPkl"),
    Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
         "projects/BEVFusion/demo/Ivo/ResultsVal/Results5_valPkl"),
]

OUT_DIR = Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
               "projects/BEVFusion/demo/Ivo/Matches_EgoFzg")

IOU_THR  = 0.10            # Mindest-IoU
VERBOSE  = False           # True → Debug-Ausgabe
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Label-ID → nuScenes-Kategorie
CLASS_MAP = {
    0: "vehicle.car", 1: "vehicle.truck", 2: "vehicle.bus",
    3: "vehicle.trailer", 4: "vehicle.construction_vehicle",
    5: "human.pedestrian.adult", 6: "vehicle.motorcycle",
    7: "vehicle.bicycle", 8: "movable_object.debris", 9: "barrier"
}

# ───────────── vereinfachte 3-D-IoU ─────────────
def iou_3d(gt, det):
    # LiDAR-Frame: x = vorne, y = links ⇢ hier x = rechts, y = vorne
    p1 = np.array([gt["position"][1], gt["position"][0], gt["position"][2]])
    p2 = np.array([det["position"][1], det["position"][0], det["position"][2]])
    s1, s2 = np.array(gt["size"]), np.array(det["size"])

    def bounds(c, s):
        return (c[0]-s[0]/2, c[0]+s[0]/2,
                c[1]-s[1]/2, c[1]+s[1]/2,
                c[2]-s[2]/2, c[2]+s[2]/2)

    x1min,x1max,y1min,y1max,z1min,z1max = bounds(p1, s1)
    x2min,x2max,y2min,y2max,z2min,z2max = bounds(p2, s2)

    xo = max(0, min(x1max,x2max)-max(x1min,x2min))
    yo = max(0, min(y1max,y2max)-max(y1min,y2min))
    zo = max(0, min(z1max,z2max)-max(z1min,z2min))
    inter = xo * yo * zo
    union = np.prod(s1) + np.prod(s2) - inter
    return inter / union if union else 0.0

# ───────────── Hungarian-Matcher ─────────────
def match(gt_list, det_list):
    m, n = len(gt_list), len(det_list)
    cost = np.full((m, n), np.inf)

    # Kostenmatrix füllen
    for i, gt in enumerate(gt_list):
        for j, det in enumerate(det_list):
            if gt["category"] == CLASS_MAP.get(det["class"], "unknown"):
                cost[i, j] = -iou_3d(gt, det)

    if VERBOSE:
        print(f"      ├ Klassen-kompatible Paare : {np.isfinite(cost).sum()}")

    cost[np.isinf(cost)] = 1000.0
    gi, dj = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(gi, dj):
        iou = -cost[i, j]
        if iou >= IOU_THR:
            matches.append({
                "ground_truth_instance": gt_list[i]["instance_token"],
                "detection_id":          det_list[j]["id"],
                "iou":                   iou,
                "category":              gt_list[i]["category"]
            })

    if VERBOSE:
        print(f"      └ Paare nach IoU ≥ {IOU_THR:.2f}: {len(matches)}")
    return matches

# ───────────── Haupt-Loop ─────────────
for det_dir in DET_DIRS:
    print(f"\n🔍  Suche in {det_dir} …")
    for det_path in det_dir.glob("results_scene-*.json"):

        scene_part = det_path.stem.replace("results_scene-", "")   # z. B. 0161_1
        gt_path  = GT_DIR  / f"ground_truth_scene-{scene_part}.json"
        out_path = OUT_DIR / f"matches_scene-{scene_part}.json"

        if not gt_path.exists():
            print(f"⚠️  GT fehlt: scene-{scene_part}")
            continue

        # 1) Detektionen laden & umwandeln
        det_raw = json.load(det_path.open())
        detections = []
        for idx, (bbox, label) in enumerate(zip(
                det_raw["bboxes"],
                det_raw.get("labels", [0]*len(det_raw["bboxes"])))):
            x, y, z, l, w, h, yaw, *_ = bbox   # length ↔ width tauschen
            detections.append({
                "id":       idx,
                "class":    int(label),
                "position": [x, y, z],
                "size":     [w, l, h],
                "yaw":      float(yaw)
            })

        # 2) Ground-Truth laden
        ground_truth = json.load(gt_path.open())

        # 3) Matching
        matches = match(ground_truth, detections)

        # 4) Speichern
        json.dump(matches, out_path.open("w"), indent=4)
        print(f"✅  scene-{scene_part}: {len(matches):3d} Matches  →  {out_path.name}")
