# Setup detectron2 logger
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch
import json


import time
import tqdm
# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor

from detectron2.data.datasets import register_coco_instances
from demo.visualizer import Visualizer,ColorMode
from detectron2.data.detection_utils import read_image

import atexit
import bisect
import multiprocessing as mp

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
cpu_device = torch.device("cpu")


def setup_cfg(dataset, model_path):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    
    ########################################update path#######################################################
    cfg_path = '.../configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml'
    #########################################################################################################
  
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ONE_FORMER.HIDDEN_DIM = 128
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES = 60
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_CTX = 8
    cfg.MODEL.TEXT_ENCODER.WIDTH = 128
    cfg.INPUT.MIN_SIZE_TRAIN = 128  # You can specify a list of sizes or a single value
    cfg.INPUT.MAX_SIZE_TRAIN = 128
    cfg.freeze()
    return cfg

########################################update path#######################################################
# Define paths
annotation_file_path_train = '.../annotations/instances_train2017.json'
image_root_train = '.../final-1/train'
#########################################################################################################


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_category_info(annotation_file_path):
    data = load_json(annotation_file_path)
    categories = data['categories']
    thing_classes = [category['name'] for category in categories]
    thing_dataset_id_to_contiguous_id = {category['id']: idx for idx, category in enumerate(categories)}
    return thing_classes, thing_dataset_id_to_contiguous_id

thing_classes, thing_dataset_id_to_contiguous_id = get_category_info(annotation_file_path_train)

register_coco_instances("custom_train", {}, annotation_file_path_train, image_root_train)
metadata_train = MetadataCatalog.get("custom_train")
metadata_train.set(thing_classes=thing_classes, thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
dataset = 'custom_train'

########################################update path#######################################################
model_path = '.../OneFormer/output_saved/model_final.pth'
#########################################################################################################

# print('metadata ')
# print(metadata_train.get("thing_classes", None))
cfg = setup_cfg(dataset, model_path)
num_gpu = torch.cuda.device_count()
predictor = DefaultPredictor(cfg)


def instance_prediction(image):
    image = image[:, :, ::-1]
    visualizer = Visualizer(image, metadata= metadata_train, instance_mode=ColorMode.IMAGE_BW)
    predictions = predictor(image,task='instances')
    instances = predictions["instances"].to(cpu_device)
    instances = instances[instances.scores>0.1]  # mention the threshold for instance score
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out


########################################update path#######################################################
# path to test image folder
image_path = '.../final-1/test'
image_title = os.listdir(image_path)
# path the save the generated results 
output = '.../final-1/results'
#########################################################################################################


# print(image_title) 
# # Initialize tqdm with the list of files
k = 0
for file_name in tqdm.tqdm(image_title):
    file_path = os.path.join(image_path, file_name)
    img = read_image(file_path, format="BGR")
    out = instance_prediction(img)
    
    opath = output 
    os.makedirs(opath, exist_ok=True)
    out_filename = os.path.join(opath,file_name)
    out.save(out_filename)   
