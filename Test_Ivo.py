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
img_base = root_dir  # Annahme: in den Metadaten sind relative Bildpfade hinterlegt

# === Kameras definieren ===
cam_types = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

# === Output-Ordner ===
output_dir = '/home/mobilitylabextreme002/Documents/mmdetection3d/projects/BEVFusion/demo/Ivo/Results5'
os.makedirs(output_dir, exist_ok=True)

# === Metadaten laden (Mapping) ===
ann_data = mmengine.load(ann_file)
data_list = ann_data['data_list']

# === Lidar-Dateien finden (alle .pcd.bin) ===
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd.bin')))
scene_count = 0

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

    # === Kamera-Bilder zusammensuchen (über Mapping) ===
    img_files = []
    all_found = True
    for cam in cam_types:
        if cam not in data_info['images']:
            print(f"❌ Keine Bildinfo für Kamera {cam} in Mapping für {pcd_filename}")
            all_found = False
            break
        rel_img_path = data_info['images'][cam]['img_path']
        # Falls der Pfad nicht absolut ist, ergänzen  ihn mit dem entsprechenden Kameraordner
        if not os.path.isabs(rel_img_path):
            img_path = os.path.join(img_base, cam, os.path.basename(rel_img_path))
        else:
            img_path = rel_img_path
        if not os.path.isfile(img_path):
            print(f"❌ Fehlendes Bild: {img_path}")
            all_found = False
            break
        img_files.append(img_path)
        
    if not all_found:
        continue

    print(f"🚗 Verarbeite Szene: {pcd_filename}")

    # === Inferenz starten ===
    try:
        # Hier übergeben wir den LiDAR-Pfad als Liste, aber den Bildordner als String,
        # da bei cam_type='all' intern die Mapping-Daten genutzt werden.
        result, data = inference_multi_modality_detector(
            model, [lidar_path], img_base, ann_file, cam_type='all'
        )
    except Exception as e:
        print(f"❌ Fehler bei Inferenz für {pcd_filename}: {e}")
        continue

    # Falls als Batch (Liste) zurückgegeben, das erste Element extrahieren
    if isinstance(result, list):
        result = result[0]
    if isinstance(data, list):
        data = data[0]

    # === Ergebnisse extrahieren ===
    try:
        bboxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy().tolist()
        scores = result.pred_instances_3d.scores_3d.cpu().numpy().tolist()
        labels = result.pred_instances_3d.labels_3d.cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ Fehler beim Extrahieren der Ergebnisse für {pcd_filename}: {e}")
        continue

    result_dict = {
        'lidar_file': pcd_filename,
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







# import os
# import mmcv
# import json
# from mmdet3d.apis import inference_multi_modality_detector, init_model
# from mmdet3d.registry import VISUALIZERS

# # Pfad zum LiDAR-Verzeichnis
# lidar_dir = "/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/"
# lidar_files = os.listdir(lidar_dir)

# # Modell einmal laden
# config_path = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
# checkpoint_path = "projects/BEVFusion/pretrained_models/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
# device = 'cuda:0'
# model = init_model(config_path, checkpoint_path, device=device)

# # Visualisierer initialisieren (falls du die Ergebnisse visualisieren möchtest)
# visualizer = VISUALIZERS.build(model.cfg.visualizer)
# visualizer.dataset_meta = model.dataset_meta

# def convert_result_to_dict(result):
#     result_dict = {
#         'bboxes': result.pred_instances_3d.bboxes_3d.tensor.tolist(),
#         'scores': result.pred_instances_3d.scores_3d.tolist(),
#         'labels': result.pred_instances_3d.labels_3d.tolist(),
#     }
#     return result_dict

# # Beispielhafte statische Pfade für Bild- und Annotationsdaten (Gegebenheiten anpassen)

# img_file = 'data/nuscenes/samples/'
# ann_file = 'data/nuscenes/nuscenes_infos_train.pkl'
# cam_type = 'all'         #'CAM_FRONT'

# # Alle .bin-Dateien aus dem Verzeichnis durchlaufen
# for file in lidar_files:
#     # Vollständigen Pfad zur LiDAR-Datei erstellen
#     pcd_file = os.path.join(lidar_dir, file)
    
#     # Inferenz durchführen
#     result, data = inference_multi_modality_detector(model, pcd_file, img_file, ann_file, cam_type)
    
#     # Ergebnis in ein Dictionary konvertieren
#     result_dict = convert_result_to_dict(result)
    
#     # Ergebnis als JSON abspeichern
#     output_filename = f'results_{os.path.splitext(file)[0]}.json'
#     with open(output_filename, 'w') as f:
#         json.dump(result_dict, f, indent=4)
    
#     print(f"Ergebnis für {file} gespeichert in {output_filename}")
