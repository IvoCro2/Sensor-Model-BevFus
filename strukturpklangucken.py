import pickle, pprint

pkl = "/home/mobilitylabextreme002/Documents/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl"
data = pickle.load(open(pkl, "rb"))

print("Erster Eintrag in data_list:")
pprint.pp(data["data_list"][0].keys())
