# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# transform as coco data
import os
import pickle
import numpy as np
import json
import re

import cv2
import glob
import shutil
import sys, stat
import pycocotools.mask as maskUtils
from detectron2.structures import BoxMode
# val imports
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
import torch


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger('a.log', sys.stdout)
sys.stderr = Logger('a.log_file', sys.stderr)

def get_fish_dicts(file_obj):
    if file_obj == 'train':
        file_dir = '/data/project/annotations/niap_fix_maskrcnn_train0_30.pik'
    elif file_obj == 'val':
        file_dir = '/data/project/annotations/niap_fix_maskrcnn_val_30_40.pik'
    elif file_obj == 'test':
        file_dir = '/data/project/annotations/niap_fix_maskrcnn_test_40_45.pik'
    else:
        print('Cant find your file !!!')
    with open(file_dir, 'rb') as f:# ------------change here
        train_data = pickle.load(f)
    total_obj = max(train_data.keys())+1 # objs amount
    #total_obj = 2
    i = -1
     dataset_dicts = []
    for v in range(total_obj):
        #print("write:")
        #print(v)

        record = {}

        # image details
        file_path = list(list(train_data[v].items())[1])[1]
        # print(file_path)
        filename = file_path.replace('/data/NIAP/Data/frames','/data/project/frame')

        height, width = cv2.imread(filename).shape[:2]
        really_filename = file_path.replace('/data/NIAP/Data/frames/', '')

        record["file_name"] = filename
        record["image_id"] = v
        record["height"] = height
        record["width"] = width
        
                annos = train_data[v]['points']

        objs = []
        # print(annos)
        for point in annos:
        #annotation points
            # points = list(list(train_data[v].items())[0])[1]
            str_a = ','.join(str(x) for x in point)
            str_a1 = str_a.replace('[','')
            str_a2 = str_a1.replace(']','')
            clean_str = ' '.join(str_a2.split())
            str_a3 = re.sub('[\s+]',',',clean_str)
            points_list = str_a3.split(',')
            points_list = filter(None, points_list)#remove ''
            a_float_m = map(float, points_list)
            # print('::',points_list,'::')
            a_float_m = list(a_float_m)

            px = a_float_m[::2]
            py = a_float_m[1::2]


            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            # print(":",len(poly),":")

            if len(poly) <= 4:
                continue
            else:
                # calculate area
                rles = maskUtils.frPyObjects([poly], height, width)
                rle = maskUtils.merge(rles)
                area = float(maskUtils.area(rle))

                i = i+1

                anno = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
            objs.append(anno)
        record["annotations"] = objs

        dataset_dicts.append(record)
        
            print('Finished')
    return dataset_dicts

for d in ["train", "val" , "test"]:
    DatasetCatalog.register("fish_" + d, lambda d=d: get_fish_dicts(d))
    MetadataCatalog.get("fish_" + d).set(thing_classes=["fish"])
fish_metadata = MetadataCatalog.get("fish_train")

dataset_dicts = get_fish_dicts('train')

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fish_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = '/data/project/mask_cascade_output/cascade_mask_rcnn_R_50_FPN_3x_out_put/model_final.pth'

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00001  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#cfg.SOLVER.STEPS = [ 72000, 81000]

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (fish)

#set val func
cfg.DATASETS.VAL = ("fish_val",)
#validation function:
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                                **loss_dict_reduced)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])
# swap the order of PeriodicWriter and ValidationLoss
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=False)
trainer.train()

#validation
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("fish_val", )
predictor = DefaultPredictor(cfg)
#validation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("fish_val", cfg, False, output_dir="./output/cascade/val/")
val_loader = build_detection_test_loader(cfg, "fish_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# test
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("fish_test", )
predictor = DefaultPredictor(cfg)

#Test
evaluator_test = COCOEvaluator("fish_test", cfg, False, output_dir="./output/cascade/test/")
test_loader = build_detection_test_loader(cfg, "fish_test")
print(inference_on_dataset(trainer.model, test_loader, evaluator_test))
