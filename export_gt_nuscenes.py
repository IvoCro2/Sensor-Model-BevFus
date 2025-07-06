from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion          # ‼️ neu
from pathlib import Path
import json, numpy as np

# -------------------- Einstellungen --------------------
DATAROOT   = "/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes"     
VERSION    = "v1.0-trainval"            
OUT_DIR    = Path("data/ground_truth")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

# -------------------------------------------------------
# Hilfs-Funktion: global → ego → LiDAR
# -------------------------------------------------------
def ann_to_lidar_dict(ann_token, lidar_token):
    ann  = nusc.get('sample_annotation', ann_token)   # Metadaten
    box  = nusc.get_box(ann_token)                    # Box im GLOBAL-Frame

    # --- ① global → ego ---
    sd_rec   = nusc.get('sample_data', lidar_token)
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    box.translate(-np.array(pose_rec['translation']))
    box.rotate(Quaternion(pose_rec['rotation']).inverse)

    # --- ② ego → LiDAR-sensor ---
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    box.translate(-np.array(cs_rec['translation']))
    box.rotate(Quaternion(cs_rec['rotation']).inverse)

    return {
        "instance_token": ann["instance_token"],
        "position":  list(box.center),        # jetzt LiDAR-Koordinaten
        "size":      list(box.wlh),           # [w, l, h]
        "yaw":       float(box.orientation.yaw_pitch_roll[0]),
        "visibility": ann["visibility_token"],
        "category":  ann["category_name"]
    }

# -------------------------------------------------------
# Export pro Szene / Keyframe
# -------------------------------------------------------
for scene in nusc.scene:
    scene_name = scene["name"]               # z.B. scene-0161
    sample     = nusc.get("sample", scene["first_sample_token"])
    part_idx   = 1

    while True:
        lidar_token = sample["data"]["LIDAR_TOP"]
        gts = [ann_to_lidar_dict(t, lidar_token) for t in sample["anns"]]

        out_file = OUT_DIR / f"ground_truth_{scene_name}_{part_idx}.json"
        with open(out_file, "w") as f:
            json.dump(gts, f, indent=4)
        print(f"✅  {out_file} ({len(gts)} Objekte)")

        if sample["next"] == "":
            break
        sample   = nusc.get("sample", sample["next"])
        part_idx += 1



































# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from pathlib import Path
# import json
# import numpy as np

# # -------------------- Einstellungen --------------------
# DATAROOT   = "/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes"     
# VERSION    = "v1.0-trainval"            # oder "v1.0-trainval"
# OUT_DIR    = Path("data/ground_truth")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# # -------------------------------------------------------

# nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

# def box_to_dict(box, ann):
#     # nuScenes: box.wlh = [width, length, height]  (passt!)
#     return {
#         "instance_token": ann["instance_token"],
#         "position":  list(box.center),           # [x, y, z]
#         "size":      list(box.wlh),              # [w, l, h]
#         "yaw":       float(box.orientation.yaw_pitch_roll[0]),
#         "visibility": ann["visibility_token"],
#         "category":  ann["category_name"]
#     }

# for scene in nusc.scene:

#     scene_name = scene["name"]            # z.B. "scene-0161"
#     first_sample = nusc.get("sample", scene["first_sample_token"])
#     sample = first_sample
#     part_idx = 1

#     while True:
#         # ---------- alle Annotationen in diesem Keyframe ----------
#         gts = []
#         for ann_token in sample["anns"]:
#             ann = nusc.get("sample_annotation", ann_token)
#             box = nusc.get_box(ann_token)
#             gts.append(box_to_dict(box, ann))

#         # ---------- Datei speichern ----------
#         out_file = OUT_DIR / f"ground_truth_{scene_name}_{part_idx}.json"
#         with open(out_file, "w") as f:
#             json.dump(gts, f, indent=4)
#         print(f"✅  {out_file} ({len(gts)} Objekte)")

#         # ---------- nächstes Sample ----------
#         if sample["next"] == "":
#             break
#         sample = nusc.get("sample", sample["next"])
#         part_idx += 1
