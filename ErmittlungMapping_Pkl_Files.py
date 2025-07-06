# import pickle
# import os

# # === INPUT ===
# # Pfad zu deinem Mapping-File (nuscenes_infos_xxx.pkl)
# mapping_pkl_path = '/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl'

# # Name der LiDAR-Datei, die du prüfen willst
# lidar_filename = '0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin'

# # === LOGIK ===
# # Timestamp aus dem Dateinamen extrahieren
# timestamp_str = lidar_filename.split('__')[-1].replace('.bin', '')
# timestamp = int(timestamp_str)

# # Mapping-Datei laden
# with open(mapping_pkl_path, 'rb') as f:
#     infos = pickle.load(f)

# # Alle Lidar-Timestamps aus der Mapping-Datei holen
# all_timestamps = []
# for info in infos:
#     if 'lidar_path' in info and info['lidar_path']:
#         lidar_path = info['lidar_path']
#         lidar_file = os.path.basename(lidar_path)
#         lidar_timestamp_str = lidar_file.split('__')[-1].replace('.bin', '')
#         lidar_timestamp = int(lidar_timestamp_str)
#         all_timestamps.append(lidar_timestamp)

# # Prüfen, ob Timestamp existiert
# if timestamp in all_timestamps:
#     print(f"✅ Timestamp {timestamp} gefunden!")
# else:
#     print(f"❌ Timestamp {timestamp} NICHT gefunden!")

import os, pickle
from pathlib import Path

# ── INPUT anpassen ────────────────────────────────
PKL_PATH  = Path('data/nuscenes/nuscenes_infos_train.pkl')
MASK_FILE = '0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin'
# ─────────────────────────────────────────────────

# PKL laden (neuere MMDet3D-Versionen => dict mit 'data_list')
infos_obj = pickle.load(PKL_PATH.open('rb'))
infos     = infos_obj['data_list'] if isinstance(infos_obj, dict) else infos_obj

# herausfinden, wie das Maskenfeld heißt
mask_key = next(k for k in ('pts_semantic_mask_path', 'lidarseg_path',
                            'semantic_mask_path') if k in infos[0])

# nur Dateiname vergleichen (Verzeichnisse ignorieren)
basename = os.path.basename(MASK_FILE)
hits = [i for i, entry in enumerate(infos)
        if mask_key in entry and os.path.basename(entry[mask_key]) == basename]

if hits:
    print(f"✅  '{basename}' im PKL gefunden (Sample-Index {hits[0]})")
else:
    print(f"❌  '{basename}' taucht im PKL NICHT auf")
