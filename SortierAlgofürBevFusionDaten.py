import os
import shutil
from glob import glob
import sys

# Basisordner
source_base = "/media/mobilitylabextreme002/Data1/DatenBevFusion"
target_base = "/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes"
# Datenordner
dirs = [
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "LIDAR_TOP",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"
]

# Zielordner anlegen
for datatype in ["samples", "sweeps"]:
    for dir_name in dirs:
        os.makedirs(os.path.join(target_base, datatype, dir_name), exist_ok=True)

# Dateien kopieren
blobs = sorted(glob(os.path.join(source_base, "v1.0-trainval*_blobs")))

total_blobs = len(blobs)  # Gesamtanzahl der Blobs

for idx, blob in enumerate(blobs, start=1):
    blob_name = os.path.basename(blob)  # z.B. "v1.0-trainval01_blobs"

    # 🗂️ Fortschrittsanzeige
    print(f"\n\033[1;34m➡️  Bearbeite Blob {idx}/{total_blobs}: {blob_name}\033[0m")

    for datatype in ["samples", "sweeps"]:
        for dir_name in dirs:
            src_dir = os.path.join(blob, datatype, dir_name)
            dst_dir = os.path.join(target_base, datatype, dir_name)

            if os.path.isdir(src_dir):
                for file_path in glob(os.path.join(src_dir, "*")):
                    file_name = os.path.basename(file_path)
                    dst_file = os.path.join(dst_dir, file_name)
                    if not os.path.exists(dst_file):
                        shutil.copy2(file_path, dst_file)
                        print(f"Kopiert {file_name} {dir_name} nach {dst_dir}")
                    else:
                        print(f"Überspringe {file_name} {dir_name}, existiert schon.")

    # ❗ Nach dem letzten Ordner sofort abbrechen
    if "v1.0-trainval10_blobs" in blob:
        print(f"\n\033[1;32m✅ Letzter Ordner {blob_name} fertig bearbeitet. Skript wird beendet.\033[0m")
        sys.exit(0)