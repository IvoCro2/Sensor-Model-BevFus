import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import glob

# Dateien laden
with open("data/ground_truth/ground_truth_scene-0161_1.json", "r") as f:
    ground_truth = json.load(f)

with open("data/bevfusion_detection/bevfusion_detection_scene-0161_1.json", "r") as f:
    detections = json.load(f)

def iou_3d(gt, det):
    """
    Berechnet eine vereinfachte 3D-IoU für zwei Boxen.
    gt und det sind Dictionaries mit folgenden Feldern:
      - "position": [x, y, z]  (hier: x: Vorwärtsachse, y: Seitenachse)
      - "size": [w, l, h] (Annahme: Reihenfolge [width, length, height])
      - "yaw": Rotation (wird hier ignoriert, kann erweitert werden)
    Hier erfolgt eine vereinfachte Berechnung, die in der XY-Ebene die Überschneidung der Flächen
    und in Z die Überschneidung der Höhen berechnet.

    ACHTUNG: Für das Ego-Koordinatensystem (x: rechts, y: vorne) werden hier x und y vertauscht.
    """
    # Extrahiere Parameter und tausche x und y (new_x = old_y, new_y = old_x)
    pos1_orig = np.array(gt["position"])
    pos1 = np.array([pos1_orig[1], pos1_orig[0], pos1_orig[2]])
    size1 = np.array(gt["size"])
    
    pos2_orig = np.array(det["position"])
    pos2 = np.array([pos2_orig[1], pos2_orig[0], pos2_orig[2]])
    size2 = np.array(det["size"])
    
    # Annahme: Boxen sind achs-aligned (Rotation wird hier nicht berücksichtigt)
    # Berechne Grenzen in XY und Z
    # Hier nehmen wir an, dass size = [w, l, h] und pos = [x, y, z] ist der Boxmittelpunkt.
    def get_bounds(pos, size):
        # in x: pos[0] ± w/2, in y: pos[1] ± l/2, in z: pos[2] ± h/2
        return (pos[0] - size[0]/2, pos[0] + size[0]/2,
                pos[1] - size[1]/2, pos[1] + size[1]/2,
                pos[2] - size[2]/2, pos[2] + size[2]/2)
    
    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = get_bounds(pos1, size1)
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = get_bounds(pos2, size2)
    
    # Berechne Überschneidungsbereich in jeder Dimension
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    
    intersection = x_overlap * y_overlap * z_overlap
    vol1 = np.prod(size1)
    vol2 = np.prod(size2)
    union = vol1 + vol2 - intersection
    return intersection / union if union > 0 else 0

# Optional: Wenn deine Detection-Objekte die Klasse als Zahl liefern (z.B. 0) und
# die Ground Truth als String (z.B. "vehicle.car"), kannst du hier eine Mapping-Tabelle definieren.
bevfusion_class_map = {
    0: "vehicle.car",
    1: "vehicle.truck",
    2: "vehicle.bus",
    3: "vehicle.trailer",
    4: "vehicle.construction_vehicle",
    5: "human.pedestrian.adult",
    6: "vehicle.motorcycle",   # Beispiel, anpassen falls nötig
    7: "vehicle.bicycle",       # Beispiel
    8: "movable_object.debris", # Beispiel
    9: "barrier"                # Beispiel
}

folder_path_detections = "data/bevfusion_detection"
detection_files = glob.glob(os.path.join(folder_path_detections, "bevfusion_detection_scene-0161_*"))

for i, file in enumerate(detection_files):
    # 📥 Daten laden
    with open(file, "r") as f:
        detections = json.load(f)

    file_name = os.path.basename(file)
    prefix = "bevfusion_detection_"
    suffix = ".json"

    if file_name.startswith(prefix) and file_name.endswith(suffix):
        part_name = file_name[len(prefix):-len(suffix)]
    with open(("data/ground_truth/ground_truth_" + part_name + ".json"), "r") as f:
        ground_truth = json.load(f)
    

    # Erstelle die Kostenmatrix (Zeilen: Ground Truth, Spalten: Detections)
    num_gt = len(ground_truth)
    num_det = len(detections)
    cost_matrix = np.full((num_gt, num_det), np.inf)

    for i, gt in enumerate(ground_truth):
        for j, det in enumerate(detections):
            # Vergleiche nur, wenn die Klassen übereinstimmen.
            # Hier wandeln wir die numerische Klasse der Detection in einen String um:
            det_class_str = bevfusion_class_map.get(det["class"], "unknown")
            if gt["category"] == det_class_str:
                iou_val = iou_3d(gt, det)
                # Negative IoU als Kosten, damit höhere IoU einen niedrigeren "Cost" hat.
                cost_matrix[i, j] = -iou_val

    # Optional: Falls in einzelnen Reihen oder Spalten nur Inf-Werte stehen,
    # könntest du einen hohen, aber endlichen Wert einsetzen, z.B.:
    cost_matrix[np.isinf(cost_matrix)] = 1000

    # Führe den Hungarian Algorithmus aus
    gt_indices, det_indices = linear_sum_assignment(cost_matrix)

    # Erstelle das Matching-Ergebnis
    matches = []
    for i, j in zip(gt_indices, det_indices):
        # Wir können hier auch einen Mindest-IoU-Schwellenwert prüfen
        matched_iou = -cost_matrix[i, j]  # Da wir negative Werte benutzt haben
        if matched_iou < 0.1:  # Beispiel-Schwelle; anpassen falls nötig
            continue
        match = {
            "ground_truth_instance": ground_truth[i]["instance_token"],
            "detection_id": detections[j]["id"],
            "iou": matched_iou,
            "category": ground_truth[i]["category"]
        }
        matches.append(match)

    # Speichere das Matching-Ergebnis in einer JSON-Datei
    output_filename = "data/matches/matches_" + part_name + ".json"
    with open(output_filename, "w") as f:
        json.dump(matches, f, indent=4)

    print(f"Matching abgeschlossen. Ergebnisse gespeichert in {output_filename}.")
