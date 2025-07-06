#!/usr/bin/env python
# rename_results_by_lidarfile.py
# ------------------------------------------------------------
# Benennt   result*.json  ➜  results_scene-XXXX_Y.json
#  ▸ nutzt den Dateinamen aus  "lidar_file"  jeder Result-Datei
#  ▸ ermittelt dazu die Szene & Frame-Index aus den nuScenes-Metadaten
# ------------------------------------------------------------
from nuscenes.nuscenes import NuScenes
from pathlib import Path
import json, os, tqdm

# ───────────────────────── Pfade anpassen ─────────────────────────
DATAROOT = (
    "/home/mobilitylabextreme002/Documents/mmdetection3d/"
    "data/nuscenes"                               # nuScenes-Root
)
VERSION = "v1.0-trainval"                         # passender Split

DET_DIRS = [
    Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
         "projects/BEVFusion/demo/Ivo/ResultsMain/Results4_mainPkl"),
    Path("/home/mobilitylabextreme002/Documents/mmdetection3d/"
         "projects/BEVFusion/demo/Ivo/ResultsVal/Results5_valPkl"),
]
# ──────────────────────────────────────────────────────────────────

print("⊳ Lade nuScenes-Metadaten …")
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# ───────── 1. Mapping  (LIDAR-Dateiname → scene_part) ────────────
lookup = {}
for sd in nusc.sample_data:                       # alle Sensor-Frames
    if sd["channel"] != "LIDAR_TOP":
        continue

    fname = Path(sd["filename"]).name             # nur Dateiname

    sample = nusc.get("sample", sd["sample_token"])
    scene  = nusc.get("scene",  sample["scene_token"])
    scene_name = scene["name"]                    # z. B. scene-0161

    # Rückwärts über prev-Kette zählen → exakter 1-basierter Frame-Index
    idx, tmp = 1, sd
    while tmp["prev"] != "":
        tmp = nusc.get("sample_data", tmp["prev"])
        idx += 1

    lookup[fname] = f"{scene_name}_{idx}"

print(f"⊳ Mapping-Einträge: {len(lookup):,}")

# ───────── 2. Result-Dateien umbenennen ──────────────────────────
renamed, missing = 0, 0
for det_dir in DET_DIRS:
    print(f"\n⊳ Verarbeite {det_dir} …")
    for det_path in tqdm.tqdm(sorted(det_dir.glob("result*.json"))):
        det = json.load(det_path.open())
        lid_fn = Path(det["lidar_file"]).name      # nur Dateiname

        scene_part = lookup.get(lid_fn)
        if scene_part is None:
            missing += 1
            continue

        new_path = det_path.with_name(f"results_scene-{scene_part}.json")
        if not new_path.exists():                  # nicht überschreiben
            os.rename(det_path, new_path)
            renamed += 1

print(f"\n✅  {renamed:,} Dateien umbenannt, {missing} ohne Mapping.")
