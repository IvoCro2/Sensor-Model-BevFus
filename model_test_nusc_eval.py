import os
from torchpack.utils.config import configs
import mmcv
from mmcv import Config, DictAction
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.apis import single_gpu_test
import torch

CONFIG_FILE = './configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser/var_head.yaml'

configs.load(CONFIG_FILE, recursive=True)
cfg = Config(recursive_eval(configs), filename=CONFIG_FILE)

samples_per_gpu = 1
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    samples_per_gpu = max(
        [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
    )
    if samples_per_gpu > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
			
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
torch.cuda.set_device(1)
torch.cuda.empty_cache()
model = MMDataParallel(model, device_ids=[1])
model.eval()
with torch.no_grad():
	for i in data_loader:
		res = model(**i)
		break
print(res[0].keys())
print(res[0]['layer_-1']['boxes_3d'].tensor.shape, res[0]['layer_-1'].keys()) 

outputs = res*6019
assert len(outputs) == 6019
assert len(outputs[0]) == 1
print("Evaluate Output", outputs[0]['layer_-1'].keys())
print(dataset.evaluate(outputs))
