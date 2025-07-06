import pickle, os, glob, pathlib
basename = '0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin'
root = pathlib.Path('data/nuscenes')
for pkl in glob.glob(str(root/'nuscenes_infos_*pkl')):
    infos = pickle.load(open(pkl, 'rb'))
    infos = infos.get('data_list', infos) if isinstance(infos, dict) else infos
    key   = next(k for k in ('pts_semantic_mask_path','lidarseg_path','semantic_mask_path') if k in infos[0])
    found = any(os.path.basename(d[key]) == basename for d in infos if key in d)
    print(f"{pkl:45}  {'✅' if found else '❌'}")
