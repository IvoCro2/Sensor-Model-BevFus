from nuscenes.nuscenes import NuScenes

# Gib hier den Pfad zum nuScenes-Datensatz an (z.B. "v1.0-trainval")
data_path = '/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

# Anzahl der Szenen ermitteln
num_scenes = len(nusc.scene)
print(f"Anzahl der Szenen im Datensatz: {num_scenes}")
