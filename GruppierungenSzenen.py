from nuscenes.nuscenes import NuScenes

# nuScenes-Datensatz an (übergeordnetes Verzeichnis, in dem v1.0-trainval liegt)
data_path = '/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

# Alle Szenen aus dem Datensatz
scenes = nusc.scene
print("Anzahl der Szenen:", len(scenes))  # sollte z.B. 850 ausgeben

# Gruppiere alle Samples (ein vollständiger Sample enthält alle Sensor-Daten) pro Szene
scene_samples = {}
for scene in scenes:
    scene_token = scene['token']
    sample_token = scene['first_sample_token']
    samples_in_scene = []
    # Iteriere über alle Samples einer Szene mittels des "next"-Felds
    while sample_token:
        samples_in_scene.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
    scene_samples[scene_token] = samples_in_scene

# Beispielausgabe: Anzahl Samples pro Szene
for scene_token, samples in scene_samples.items():
    print(f"Szene {scene_token} hat {len(samples)} Samples.")

print(f"Insgesamt verteilte Samples: {sum(len(s) for s in scene_samples.values())}")


