# Copyright (c) OpenMMLab. All rights reserved.

# Importieren notwendiger Module und Bibliotheken
from argparse import ArgumentParser  # Für die Kommandozeilenargumentverarbeitung, damit man von der Konsole aus Argumente übergeben kann
import mmcv                        # OpenMMLab-Toolbox, nützlich für Bildverarbeitung und diverse Hilfsfunktionen
import json                        # Für den JSON-Datenaustausch (Lesen/Schreiben von JSON-Dateien)
import numpy as np                 # Numerische Berechnungen, Array-Operationen
from pathlib import Path           # Arbeiten mit Dateipfaden (plattformunabhängig)
from typing import Sequence      # Typannotationen für sequentielle Daten (wie Listen oder Tupel)

# Importieren von Funktionen und Klassen aus dem mmdet3d-Framework
from mmdet3d.apis import inference_multi_modality_detector, init_model  
#  -> inference_multi_modality_detector: führt die Inferenz (Erkennung) mit multimodalen Eingabedaten durch (z. B. Bild + LiDAR)
#  -> init_model: initialisiert ein Modell anhand einer Konfigurationsdatei und eines Checkpoints

from mmdet3d.registry import VISUALIZERS  
#  -> VISUALIZERS: eine Registry, die Visualisierer für die Darstellung von Ergebnissen verwaltet

from mmdet3d.structures import LiDARInstance3DBoxes, BaseInstance3DBoxes  
#  -> Klassen, die 3D-Box-Strukturen repräsentieren, etwa für LiDAR-Daten


############################Auslesen eines Verzeichnisinhalts############################################
import os
Path = '/home/mobilitylabextreme002/Documents/mmdetecion3d/data/nuscenes/samples/LIDAR_TOP/'
lidar_top = os.listdir(Path) # Liest den Pfad mit den ganzen Dateien aus !

# Modell einmal laden
config_path = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
# Gewichte des NN
checkpoint_path = "projects/BEVFusion/pretrained_models/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
device = 'cuda:0'
model = init_model(config_path, checkpoint_path, device=device)

#Optional
# print (lidar_top) # Gibt den Inhalt des Lidard-Verzeichnisses aus (wird vorest nicht benötigt).


#lidar_file_name = 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392'
#pcd = 'data/nuscenes/samples/LIDAR_TOP/' + lidar_file_name + '.pcd.bin'
#pcd_files = [
    #"data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin",]


# Beispielhafte Listen mit Dateipfaden für die Eingabe-Daten (LiDAR und Bilddaten)
#pcd_list = ['demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin']
#  -> pcd_list enthält Pfade zu LiDAR-Dateien im .bin-Format

##############################Auslesn der Annotationsdatei und Kamerapfade################################################################

# Pfad zur Annotationsdatei im Pickle-Format, die Informationen zu den Objekten enthält
ann = 'data/nuscenes/nuscenes_infos_train.pkl'

# Pfad zu den Kamera bildern
cam = 'data/nuscenes/samples'  #  -> Hier werden die Bilddateien für verschiedene Kameras (Hinter-, Seiten- und Frontansicht) angegeben


#########################################################################################################################################

#img_list = ['demo/data/nuscenes/']  
#  -> img_list verweist auf ein Verzeichnis, in dem Demo-Bilder (z. B. aus dem nuScenes-Datensatz) gespeichert sind

# Einzelner LiDAR-Pfad (konkrete Datei)
#pcd = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927612460.pcd.bin'

# Pfad zur Annotationsdatei im Pickle-Format, die Informationen zu den Objekten enthält
#ann = 'data/nuscenes/nuscenes_infos_train.pkl'


def parse_args():
    """
    Diese Funktion definiert und parst die Kommandozeilenargumente,
    die beim Aufruf des Skripts übergeben werden.
    """
    parser = ArgumentParser() # erzeugt einen Übersetzer („Parser“), der deine Eingabe in nutzbare Variablen verwandelt.
    # Erforderliche Argumente
    parser.add_argument('pcd', help='Point cloud file')       # Pfad zur LiDAR-Punktwolke
    parser.add_argument('img', help='image file')              # Pfad zur Bilddatei
    parser.add_argument('ann', help='ann file')                # Pfad zur Annotationsdatei
    parser.add_argument('config', help='Config file')          # Pfad zur Konfigurationsdatei des Modells
    parser.add_argument('checkpoint', help='Checkpoint file')  # Pfad zur Checkpoint-Datei des Modells

###################################Optionale Argumente################################################################
    # Optionale Argumente mit Standardwerten
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')  #default='cuda:0' (0 bedeutet die erste Grafikkarte) bedeutet, standardmäßig wird deine Grafikkarte verwendet (CUDA).
    
    #  -> Gibt das verwendete Gerät an (Standard: CUDA-GPU 0)    
    parser.add_argument('--cam-type', type=str, default='CAM_FRONT', help='choose camera type to inference')
    
    #  -> Bestimmt den Kameratyp, der für die Inferenz genutzt werden soll (z. B. Frontkamera) 0.0 bedeutet hier: Es wird erstmal alles angezeigt, auch wenn das Modell wenig sicher ist. WIE SICHER ES IST
    parser.add_argument('--score-thr', type=float, default=0.0, help='bbox score threshold')
    
    #  -> Schwellwert für die Konfidenzwerte (nur Boxen mit einem Score über diesem Wert werden berücksichtigt)
    parser.add_argument('--out-dir', type=str, default='demo', help='dir to save results')
    
    #  -> Ausgabeordner, in dem die Ergebnisse gespeichert werden sollen
    parser.add_argument('--show', action='store_true', help='show online visualization results')
    
    #  -> Flag, um die Ergebnisse direkt anzuzeigen (visuelle Ausgabe)
    parser.add_argument('--snapshot', action='store_true', help='whether to save online visualization results')
    #  -> Flag, um die Visualisierungen als Bilddateien zu speichern

    args = parser.parse_args()  # Parst die übergebenen Argumente
    return args
##############################################################################################################################


def convert_result_to_dict(result):
    """
    Wandelt das Ergebnisobjekt (typischerweise vom Typ Det3DDataSample) in ein Dictionary um,
    sodass es in ein JSON-Format serialisiert werden kann.
    """
    # Beispielhafter Ausdruck: Anzahl der Ecken der 3D-Bounding-Boxen wird ausgegeben
    print(len(result.pred_instances_3d.bboxes_3d.corners))
    result_dict = {
        'bboxes': result.pred_instances_3d.bboxes_3d.tensor.tolist(),  # Wandelt die Tensoren der Bounding Boxes in eine Liste um
        'scores': result.pred_instances_3d.scores_3d.tolist(),            # Wandelt die Konfidenzwerte in eine Liste um
        'labels': result.pred_instances_3d.labels_3d.tolist(),             # Wandelt die Labels in eine Liste um
        # Weitere Felder können hier hinzugefügt werden, falls nötig
    }
    return result_dict



###############################################################################################################################################
# Die Funktion main() ist die Hauptfunktion, die das Modell initialisiert, Inferenz durchführt,
# Ergebnisse verarbeitet, visualisiert und speichert.
def main(args):
    """
    Hauptfunktion, die das Modell initialisiert, Inferenz durchführt,
    Ergebnisse verarbeitet, visualisiert und speichert.
    """
    # Modell anhand der Konfigurations- und Checkpoint-Dateien initialisieren
    model = init_model(args.config, args.checkpoint, device=args.device)

    #Erstellt ein Visualisierungs-Objekt, um Vorhersagen später bildlich darzustellen.
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta  # Metadaten des Datensatzes (z. B. Klassenbezeichnungen) werden übergeben

    # Inferenz mit multimodalen Daten (Punktwolke und Bild) durchführen.
    # args.pcd: Pfad zur LiDAR-Punktwolke, args.img: Pfad zum Bild, args.ann: Pfad zur Annotationsdatei,
    # args.cam_type: Kameraansicht (z. B. Frontkamera)
    result, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                     args.ann, args.cam_type)
    # Alternative Variante (auskommentiert): Mehrere Kamerabilder können verarbeitet werden
    # result, data = inference_multi_modality_detector(model, pcd, cam_folders, ann, args.cam_type) 
###############################################################################################################################################
 

# Das Ergebnis wird in ein Dictionary umgewandelt, um es später als JSON speichern zu können
    converted_results = convert_result_to_dict(result)

    # Speichern der Ergebnisse in einer JSON-Datei
    output_filename = 'projects/BEVFusion/demo/felix/results.json'
    with open(output_filename, 'w') as f:
        json.dump(converted_results, f, indent=4)
    print(f"Ergebnisse wurden erfolgreich in {output_filename} gespeichert.")

    # Laden und Konvertieren des Bildes für die Visualisierung:
    points = data['inputs']['points']  # Extrahiert die Punktwolke aus den Eingabedaten

    # Überprüfen, ob der Bildpfad eine Liste ist (mehrere Bilder) oder nur ein einzelner Pfad
    if isinstance(result.img_path, list):
        img = []
        for img_path in result.img_path:
            single_img = mmcv.imread(img_path)  # Bild einlesen
            single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')  # Farbkanäle konvertieren (BGR zu RGB)
            img.append(single_img)
    else:
        img = mmcv.imread(result.img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

    # Zusammenstellen der Eingabedaten für die Visualisierung
    data_input = dict(points=points, img=img)

    # Visualisierung der Ergebnisse:
    # Hier wird eine Datenprobe (data sample) dem Visualisierer hinzugefügt, der das Bild, die Punktwolke und die Vorhersagen kombiniert.
    visualizer.add_datasample(
        'result',         # Name des Datensatzes für die Visualisierung
        data_input,       # Die Eingabedaten (Punktwolke und Bild)
        data_sample=result,   # Das Ergebnisobjekt, das Vorhersagen wie Bounding Boxes, Scores und Labels enthält
        draw_gt=False,        # Gibt an, dass keine Ground-Truth (wahre Beschriftungen) gezeichnet werden sollen
        show=args.show,       # Ob die Visualisierung direkt angezeigt wird (abhängig vom Kommandozeilenargument)
        wait_time=-1,         # Wartezeit für die Anzeige (hier unendlich, bis ein Tastendruck erfolgt)
        out_file=args.out_dir,# Verzeichnis, in dem das Visualisierungsbild gespeichert wird
        pred_score_thr=args.score_thr,  # Schwellwert für die Anzeige von Vorhersagen (nur Boxen mit Score über diesem Wert werden gezeigt)
        vis_task='multi-modality_det'     # Bestimmt die Art der Visualisierungsaufgabe (hier multimodale Detektion)
    )

# Wenn das Skript direkt ausgeführt wird, werden hier die Argumente geparst und die main()-Funktion aufgerufen
if __name__ == '__main__':
    args = parse_args()
    main(args)



















































# # Copyright (c) OpenMMLab. All rights reserved.
# from argparse import ArgumentParser # Für die Kommandozeilenargumentverarbeitung

# import mmcv     # OpenMMLab-Toolbox, nützlich für Bildverarbeitung und diverse Hilfsfunktionen
# import json  #Für den JSON-Datenaustausch (Lesen/Schreiben von JSON-Dateien)
# import numpy as np # Numerische Berechnungen, Array-Operationen

# from pathlib import Path
# from typing import Sequence

# from mmdet3d.apis import inference_multi_modality_detector, init_model
# from mmdet3d.registry import VISUALIZERS
# from mmdet3d.structures import LiDARInstance3DBoxes, BaseInstance3DBoxes

# pcd_list = ['demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin'] #Pfade zu LiDAR-Dateien (.bin)

# img_list = ['demo/data/nuscenes/'] #Verzeichnis für nuscenes Demo

# pcd = 'data/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927612460.pcd.bin'  # Konkrete Pfade zu Lidat_Top Daten

# ann = 'data/nuscenes/nuscenes_infos_train.pkl' #Annotationen (.pkl) File

# cam_folders: Sequence[str] = ['data/nuscenes/samples/CAM_BACK',
#                 'data/nuscenes/samples/CAM_BACK_LEFT',
#                 'data/nuscenes/samples/CAM_BACK_RIGHT',
#                 'data/nuscenes/samples/CAM_FRONT',
#                 'data/nuscenes/samples/CAM_FRONT_LEFT',
#                 'data/nuscenes/samples/CAM_FRONT_RIGHT']

# cam_folders: Sequence[str] = [
#                 'data/nuscenes/samples/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927612460.jpg',
#                 'data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927612460.jpg',
#                 'data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927612460.jpg',
#                 'data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg',
#                 'data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927612460.jpg',
#                 'data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927612460.jpg']




# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('pcd', help='Point cloud file')
#     parser.add_argument('img', help='image file')
#     parser.add_argument('ann', help='ann file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--cam-type',
#         type=str,
#         default='CAM_FRONT',
#         help='choose camera type to inference')
#     parser.add_argument(
#         '--score-thr', type=float, default=0.0, help='bbox score threshold')
#     parser.add_argument(
#         '--out-dir', type=str, default='demo', help='dir to save results')
#     parser.add_argument(
#         '--show',
#         action='store_true',
#         help='show online visualization results')
#     parser.add_argument(
#         '--snapshot',
#         action='store_true',
#         help='whether to save online visualization results')
#     args = parser.parse_args()
    
    
#     return args

# def convert_result_to_dict(result):
#     # Hier gehen wir davon aus, dass `result` ein Objekt von Det3DDataSample ist
#     # und wir es in ein Dictionary umwandeln
#     print(len(result.pred_instances_3d.bboxes_3d.corners))
#     result_dict = {
#         'bboxes': result.pred_instances_3d.bboxes_3d.tensor.tolist(), 
#         'scores': result.pred_instances_3d.scores_3d.tolist(),  
#         'labels' : result.pred_instances_3d.labels_3d.tolist(), 
#         # Weitere Felder können hier hinzugefügt werden
#     }
#     return result_dict


# def main(args):
#     # build the model from a config file and a checkpoint file
#     model = init_model(args.config, args.checkpoint, device=args.device)

#     # init visualizer
#     visualizer = VISUALIZERS.build(model.cfg.visualizer)
#     visualizer.dataset_meta = model.dataset_meta


#     # test a single image and point cloud sample
#     result, data = inference_multi_modality_detector(model, args.pcd, args.img,
#                                                      args.ann, args.cam_type)   
#     #result, data = inference_multi_modality_detector(model, pcd, cam_folders,
#     #                                                 ann, args.cam_type) 
    
#     #print(result)    
                        
#     # Verarbeite alle Ergebnisse und wandle sie in ein serialisierbares Format um
#     #converted_results = [convert_result_to_dict(res) for res in result]
#     converted_results = convert_result_to_dict(result)

#     # Speichern als JSON-Datei
#     output_filename = 'projects/BEVFusion/demo/felix/results.json'
#     with open(output_filename, 'w') as f:
#         json.dump(converted_results, f, indent=4)

#     print(f"Ergebnisse wurden erfolgreich in {output_filename} gespeichert.")
    
#     # Speichere das Dictionary als JSON-Datei
#     #output_path = "result.json"
#     #with open(output_path, "w") as f:
#     #    json.dump(result_dict, f, indent=4, default=convert_ndarray)

#     #print(f"Results saved to {output_path}")

#     points = data['inputs']['points']
#     if isinstance(result.img_path, list):
#         img = []
#         for img_path in result.img_path:
#             single_img = mmcv.imread(img_path)
#             single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
#             img.append(single_img)
#     else:
#         img = mmcv.imread(result.img_path)
#         img = mmcv.imconvert(img, 'bgr', 'rgb')
#     data_input = dict(points=points, img=img)

#     # show the results
#     visualizer.add_datasample(
#         'result',
#         data_input,
#         data_sample=result,
#         draw_gt=False,
#         show=args.show,
#         wait_time=-1,
#         out_file=args.out_dir,
#         pred_score_thr=args.score_thr,
#         vis_task='multi-modality_det')


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

