# point to the compatible cuda packages
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-11.3/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-11.3/nvvm/lib64/libnvvm.so"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:200'

#import dependent files
import logging
import json
import cv2
import random
from collections import OrderedDict

import torch
import detectron2
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch,AMPTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config,build_lr_scheduler
import copy
import itertools

from oneformer import (
    COCOUnifiedNewBaselineDatasetMapper,
    OneFormerUnifiedDatasetMapper,
    SemanticSegmentorWithTTA,
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from oneformer.config import add_oneformer_config
from detectron2.config import CfgNode
from detectron2.data import build_detection_train_loader
from oneformer.data.dataset_mappers.dataset_mapper import DatasetMapper

from oneformer.evaluation import (
    COCOEvaluator,
    DetectionCOCOEvaluator,
    CityscapesInstanceEvaluator,
)


# source the instance_coco_custom_dataset_mapper file, for instance segmentation on coco dataset
import sys
########################################update path#######################################################
sys.path.insert(0, '.../OneFormer/datasets/custom_datasets/')
##########################################################################################################
from instance_coco_custom_dataset_mapper import  InstanceCOCOCustomNewBaselineDatasetMapper

from detectron2.evaluation import (
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from oneformer.evaluation.instance_evaluation import InstanceSegEvaluator

import detectron2.utils.comm as comm
from time import sleep
from detectron2.solver.build import maybe_add_gradient_clipping

from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from oneformer.utils.events import WandbWriter, setup_wandb

from detectron2.data import build_detection_test_loader
from typing import List, Dict, Any, Set


# Setup detectron2 logger
setup_logger()
########################################update path#######################################################
# Define paths
annotation_file_path_train = '.../annotations/instances_train2017.json'
image_root_train = '.../final-1/train'

annotation_file_path_test ='.../annotations/instances_val2017.json'
image_root_test = '.../final-1/valid'

# config file 
cfg_path = '.../OneFormer/configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml'
########################################################################################################


# read the json file
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# fetch metadata 
def get_category_info(annotation_file_path):
    data = load_json(annotation_file_path)
    categories = data['categories']
    thing_classes = [category['name'] for category in categories]
    thing_dataset_id_to_contiguous_id = {category['id']: idx for idx, category in enumerate(categories)}
    return thing_classes, thing_dataset_id_to_contiguous_id

# Load the annotation file and extract category information
thing_classes, thing_dataset_id_to_contiguous_id = get_category_info(annotation_file_path_train)

# Register custom datasets for coco data format 
def register_datasets():
    if "custom_train" in DatasetCatalog:
        DatasetCatalog.remove("custom_train")
        MetadataCatalog.remove("custom_train")
    if "custom_test" in DatasetCatalog:
        DatasetCatalog.remove("custom_test")
        MetadataCatalog.remove("custom_test")

    register_coco_instances("custom_train", {}, annotation_file_path_train, image_root_train)
    metadata_train = MetadataCatalog.get("custom_train")
    metadata_train.set(thing_classes=thing_classes, thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)

    
    register_coco_instances("custom_test", {}, annotation_file_path_test, image_root_test)
    metadata_test = MetadataCatalog.get("custom_test")
    metadata_test.set(thing_classes=thing_classes, thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
    


# training block tailored coco dataset format

class Trainer(DefaultTrainer):
    torch.cuda.empty_cache()
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            if cfg.MODEL.TEST.DETECTION_ON:
                evaluator_list.append(DetectionCOCOEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # custom data mapper for instance segmentation
        mapper = InstanceCOCOCustomNewBaselineDatasetMapper(cfg, True)
        # mapper = COCOUnifiedNewBaselineDatasetMapper(cfg, True)
        loader = build_detection_train_loader(cfg, mapper=mapper)
        return loader

    def build_writers(self):
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter(),
        ]

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST_SEMANTIC
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg,task= 'instance')
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, InstanceSegEvaluator):
            evaluators = [evaluators]
        
        if cfg.MODEL.TEST.TASK == "panoptic":
            test_dataset = cfg.DATASETS.TEST_PANOPTIC
        elif cfg.MODEL.TEST.TASK == "instance":
            test_dataset = cfg.DATASETS.TEST_INSTANCE
        elif cfg.MODEL.TEST.TASK == "semantic":
            test_dataset = cfg.DATASETS.TEST_SEMANTIC

        if evaluators is not None:
            assert len(test_dataset) == len(evaluators), "{} != {}".format(
                len(test_dataset), len(evaluators)
            )
        results = OrderedDict

        results = OrderedDict()
        for idx, dataset_name in enumerate(test_dataset):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use DefaultTrainer.test(evaluators=), "
                        "or implement its build_evaluator method."
                    )
                    results[dataset_name] = {}
                    continue
            
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        torch.cuda.empty_cache()
        return results
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.CLIP_GRADIENTS = CfgNode()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    # cfg.MODEL.WEIGHTS = model_weights  # Specify the model weights
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_test",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-4

    # cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 128
    # cfg.MODEL.SWIN.PATCH_SIZE = 2
    # cfg.MODEL.SWIN.EMBED_DIM = 58

    # reduce the structure and image size 
    cfg.MODEL.ONE_FORMER.HIDDEN_DIM = 128
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES = 60
    cfg.MODEL.ONE_FORMER.NUM_OBJECT_CTX = 8
    cfg.MODEL.TEXT_ENCODER.WIDTH = 128
    # Image resizing configuration
    cfg.INPUT.MIN_SIZE_TRAIN = 128  # You can specify a list of sizes or a single value
    cfg.INPUT.MAX_SIZE_TRAIN = 128
    cfg.INPUT.MIN_SIZE_TEST = 128
    cfg.INPUT.MAX_SIZE_TEST = 128
    ########################################update path#######################################################
    cfg.OUTPUT_DIR = '.../output_saved'  # folder to save the output files 
    ##########################################################################################################
    # cfg.SOLVER.STEPS = []
    # cfg.SOLVER.AMP.ENABLED = True  # Enable automatic mixed precision
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.freeze()
    setup_wandb(cfg, args)
    
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="oneformer")
    return cfg

def main(args):
    cfg = setup(args)
    register_datasets()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        net_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Params: {} M".format(net_params/1e6))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.machine_rank == 0:
        net_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print("Total Params: {} M".format(net_params/1e6))
    sleep(3)
    torch.cuda.empty_cache() 
    return trainer.train()

def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

if __name__ == "__main__":
    torch.cuda.init()  # Initialize CUDA
    torch.cuda.empty_cache()  # Clear cache if needed
    # Load the allocator
    invoke_main()
