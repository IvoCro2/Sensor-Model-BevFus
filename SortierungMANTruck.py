import os
import shutil
import sys
import time
from glob import glob
from datetime import timedelta
from tqdm import tqdm

"""
Dieses Skript zeigt einen **live**‑Fortschritts­balken *pro* Blob **und** einen äußeren Balken, der die bereits abgearbeiteten Blobs anzeigt.

• Äußerer Balken   = wie viele Blobs (Ordner) schon erledigt sind
• Innerer Balken   = Kopierfortschritt aller Dateien (samples + sweeps) im aktuellen Blob


"""

# ----------------------------- Hilfsfunktionen -----------------------------

def format_time(seconds: float) -> str:
    """Formatiere Sekunden in HH:MM:SS"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

# ----------------------------- Pfade anpassen -----------------------------
source_base = "/media/mobilitylabextreme002/Extreme SSD/Trainval"
target_base = "/media/mobilitylabextreme002/Extreme SSD/data/man-truckscenes"

# Zielgrundstruktur
sensor_dirs = [
    "CAMERA_LEFT_BACK", "CAMERA_LEFT_FRONT", "CAMERA_RIGHT_BACK", "CAMERA_RIGHT_FRONT",
    "LIDAR_LEFT", "LIDAR_REAR", "LIDAR_RIGHT", "LIDAR_TOP_FRONT", "LIDAR_TOP_LEFT", "LIDAR_TOP_RIGHT",
    "RADAR_LEFT_BACK", "RADAR_LEFT_FRONT", "RADAR_LEFT_SIDE",
    "RADAR_RIGHT_BACK", "RADAR_RIGHT_FRONT", "RADAR_RIGHT_SIDE",
]
for datatype in ("samples", "sweeps"):
    for sensor in sensor_dirs:
        os.makedirs(os.path.join(target_base, datatype, sensor), exist_ok=True)

# ----------------------------- Blobs finden -----------------------------
blob_pattern = os.path.join(source_base, "man-truckscenes_sensordata*_v1.0-trainval")
blobs = sorted(p for p in glob(blob_pattern) if os.path.isdir(p))
if not blobs:
    print(f"❌ Keine Blobs gefunden unter: {blob_pattern}")
    sys.exit(1)

# ----------------------------- Kopiervorgang -----------------------------
start = time.time()
outer_bar = tqdm(blobs, desc="Blobs (Ordner)", unit="Blob")

for blob in outer_bar:
    blob_name = os.path.basename(blob)
    inner_root = os.path.join(blob, "man-truckscenes")
    if not os.path.isdir(inner_root):
        outer_bar.write(f"⚠️  Ordner fehlt: {inner_root}")
        continue

    # ------- Alle zu kopierenden Dateien im aktuellen Blob sammeln -------
    tasks = []  # Liste (src_file, dst_file)
    for datatype in ("samples", "sweeps"):
        src_type_dir = os.path.join(inner_root, datatype)
        if not os.path.isdir(src_type_dir):
            outer_bar.write(f"⚠️  {datatype} fehlt in {blob_name}")
            continue
        for sensor in os.listdir(src_type_dir):
            src_sensor_dir = os.path.join(src_type_dir, sensor)
            if not os.path.isdir(src_sensor_dir):
                continue
            dst_sensor_dir = os.path.join(target_base, datatype, sensor)
            os.makedirs(dst_sensor_dir, exist_ok=True)
            for fname in os.listdir(src_sensor_dir):
                src_file = os.path.join(src_sensor_dir, fname)
                dst_file = os.path.join(dst_sensor_dir, fname)
                # wirD kopierT nur, wenn Datei noch nicht existiert
                if not os.path.exists(dst_file):
                    tasks.append((src_file, dst_file))

    total_files = len(tasks)
    if total_files == 0:
        outer_bar.write(f"ℹ️  {blob_name}: Alle Dateien schon vorhanden – übersprungen.")
        continue

    # ---------------- Innerer Fortschrittsbalken für diesen Blob ----------------
    inner_bar = tqdm(total=total_files,
                     desc=f"{blob_name} (samples+sweeps)",
                     unit="file", leave=False, position=1)

    blob_start = time.time()
    for src_file, dst_file in tasks:
        shutil.copy2(src_file, dst_file)
        inner_bar.update(1)
    inner_bar.close()
    blob_time = format_time(time.time() - blob_start)
    outer_bar.write(f"✅ {blob_name} fertig in {blob_time}")

outer_bar.close()
print(f"\n✅ Alle Blobs abgeschlossen in {format_time(time.time()-start)}")
