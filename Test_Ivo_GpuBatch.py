import os
import glob
import json
import mmengine
from mmdet3d.apis import init_model, inference_multi_modality_detector

# === Modell laden ===
config_path = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
checkpoint_path = "projects/BEVFusion/pretrained_models/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
device = 'cuda:0'
model = init_model(config_path, checkpoint_path, device=device)

# === Datenpfade und Mapping-Datei ===
root_dir = 'data/nuscenes/samples'
lidar_dir = os.path.join(root_dir, 'LIDAR_TOP')
ann_file = 'data/nuscenes/nuscenes_infos_val.pkl'
img_base = root_dir  # Annahme: In den Metadaten sind relative Bildpfade hinterlegt

# === Kameras definieren ===
cam_types = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# === Output-Ordner ===
output_dir = '/home/mobilitylabextreme002/Documents/mmdetection3d/projects/BEVFusion/demo/Ivo/ResultsVal/Results5_valPkl'
os.makedirs(output_dir, exist_ok=True)

# === Metadaten laden (Mapping) ===
ann_data = mmengine.load(ann_file)
data_list = ann_data['data_list']

# === LiDAR-Dateien finden (alle .pcd.bin) ===
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd.bin')))

# === Gültige Szenen sammeln ===
valid_scenes = []  # Hier sammeln wir alle Szenen, die alle Voraussetzungen erfüllen

for lidar_path in lidar_files:
    pcd_filename = os.path.basename(lidar_path)
    # Finde das Mapping-Dictionary, das genau diesen LiDAR-Dateinamen hat
    data_info = next((d for d in data_list if os.path.basename(d['lidar_points']['lidar_path']) == pcd_filename), None)
    if data_info is None:
        print(f"❌ Kein Mapping gefunden für {pcd_filename}")
        continue

    # Extrahiere Token und Timestamp (angenommen: scene_token__LIDAR_TOP__timestamp.pcd.bin)
    parts = pcd_filename.replace('.pcd.bin', '').split('__')
    if len(parts) != 3:
        print(f"❌ Ungültiges Dateiformat: {pcd_filename}")
        continue
    scene_token = parts[0]
    timestamp = parts[2]

    # Überprüfe, ob für alle Kameras Bildinformationen und Dateien vorhanden sind
    all_found = True
    for cam in cam_types:
        if cam not in data_info['images']:
            print(f"❌ Keine Bildinfo für Kamera {cam} in Mapping für {pcd_filename}")
            all_found = False
            break
        rel_img_path = data_info['images'][cam]['img_path']
        # Falls der Pfad nicht absolut ist, ergänzen wir ihn mit dem Kameraordner
        if not os.path.isabs(rel_img_path):
            img_path = os.path.join(img_base, cam, os.path.basename(rel_img_path))
        else:
            img_path = rel_img_path
        if not os.path.isfile(img_path):
            print(f"❌ Fehlendes Bild: {img_path}")
            all_found = False
            break
    if not all_found:
        continue

    valid_scenes.append({
        'lidar_path': lidar_path,
        'pcd_filename': pcd_filename,
        'scene_token': scene_token,
        'timestamp': timestamp,
        'data_info': data_info
    })

print(f"Gefundene gültige Szenen: {len(valid_scenes)}")

# === Batch-Verarbeitung definieren ===
batch_size = 8  # Batch-Größe auf 8 gesetzt, höhher würde eventuell den VRAM überlasten
scene_count = 0

for i in range(0, len(valid_scenes), batch_size):
    batch_scenes = valid_scenes[i:i + batch_size]
    batch_lidar_paths = [scene['lidar_path'] for scene in batch_scenes]

    try:
        # Batch-Inferenz starten: Hier wird eine Liste von LiDAR-Dateien übergeben.
        results, data = inference_multi_modality_detector(
            model, batch_lidar_paths, img_base, ann_file, cam_type='all'
        )
    except Exception as e:
        print(f"❌ Fehler bei der Inferenz im Batch {i // batch_size}: {e}")
        continue

    # Ergebnisse aus dem Batch extrahieren und speichern
    for idx, res in enumerate(results):
        try:
            bboxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy().tolist()
            scores = res.pred_instances_3d.scores_3d.cpu().numpy().tolist()
            labels = res.pred_instances_3d.labels_3d.cpu().numpy().tolist()
        except Exception as e:
            print(f"❌ Fehler beim Extrahieren der Ergebnisse für {batch_scenes[idx]['pcd_filename']}: {e}")
            continue

        result_dict = {
            'lidar_file': batch_scenes[idx]['pcd_filename'],
            'bboxes': bboxes,
            'scores': scores,
            'labels': labels
        }

        scene_count += 1
        output_filename = f'result{scene_count:02d}.json'
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"✅ Ergebnis gespeichert: {output_path}")

# Notiz: Prüft ob passendes Mapping in der Mapping-Datei vorhanden ist. (Nach der Prüfung wurde festgestellt, das der Datensatz nicht vollständig ist ) 
# Außerdem wird für jeden definierten Kameratyp geprüft, ob der entsprechende Bildpfad existiert und die Bilddatei vorhanden ist. 
# Falls ein Mapping oder ein Bild fehlt, wird diese Szene übersprungen. 
# Diese Vorab-Prüfungen sorgen dafür, dass nur vollständige und korrekte Daten in den Inferenzprozess gelangen und Fehler während der Verarbeitung vermieden werden.