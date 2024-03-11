# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import wandb
import logging
import numpy as np
import copy
import contextlib
import shutil
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators
from detectron2.structures import BoxMode
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from .distributed_trainer import DistributedTrainer
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable
from datasets.registration.register_refcoco_dataset import register_refcoco
from datasets.registration.register_coco_panoptic_annos_caption_grounding import \
    register_coco_panoptic_annos_caption_grounding_sem_seg

logger = logging.getLogger(__name__)


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path = os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        new_origin = base_path
        new_search_locations = ['./']
        spec.name = base_name
        spec.origin = new_origin
        spec.submodule_search_locations = [self.opt['base_path']]
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        # pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        # pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        # logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        # self.pipeline = pipeline_class(self.opt)

        # ref_coco
        # register_coco_panoptic_annos_caption_grounding_sem_seg("coco_2017_train",
        #                                                        self.get_metadata(),
        #                                                        '/content/datasets/my_data_set/coco_train2017',
        #                                                        '/content/datasets/my_data_set/pan_seg/',
        #                                                        '/content/datasets/my_data_set/panoptic_train2017.json',
        #                                                        '/content/datasets/my_data_set/sem_seg/',
        #                                                        '/content/datasets/my_data_set/captions_train2017.json/',
        #                                                        '/content/datasets/my_data_set/grounding_train2017.json/',
        #                                                        '/content/datasets/xdecoder_data/coco/annotations/caption_class_similarity.pth',
        #                                                        '/content/datasets/my_data_set/coco_train2017.json/',
        #                                                        )

        # leaves_coco
        register_coco_panoptic_annos_caption_grounding_sem_seg("coco_2017_train",
                                                               self.get_metadata(),
                                                               '/content/datasets/DARE/train_images/',
                                                               '/content/datasets/DARE/panoptic/',
                                                               '/content/datasets/DARE/panoptic/panoptic_train_gt_DARE.json',
                                                               '/content/datasets/DARE/semantic/',
                                                               '/content/datasets/DARE/DARE_captions_train2024.json',
                                                               '/content/datasets/DARE/DARE_groundings_train2024.json',
                                                               '/content/datasets/xdecoder_data/coco/annotations/caption_class_similarity.pth',
                                                               '/content/datasets/DARE/instances.json',
                                                               )
        # register_refcoco("coco_g_train_umd", {}, '/content/', 'grounding_my_val.json')
        # register_refcoco("refcocog_val_leaves_umd", self.get_metadata(), '/content/datasets/leaves_dataset/coco_train2017', '/content/datasets/leaves_dataset/coco_train2017_grounding_gt_json_leaves.json')
        register_refcoco("refcocog_val_leaves_umd", self.get_metadata(), '/content/datasets/DARE/validation',
                         '/content/datasets/DARE/validation/DARE_groundings_validation2024_1.json')

        from pipeline.XDecoderPipeline import XDecoderPipeline
        self.pipeline = XDecoderPipeline(self.opt)
        # DatasetCatalog.register("coco_my_dataset_val", self.get_dataset)
        # MetadataCatalog.get("coco_my_dataset_val").set(thing_classes=COCO_PANOPTIC_CLASSES)
        # MetadataCatalog.get("coco_my_dataset_val").set(stuff_classes=COCO_PANOPTIC_CLASSES)
        # MetadataCatalog.get("coco_my_dataset_val").set(evaluator_type="coco_panoptic_seg")
        # MetadataCatalog.get("coco_my_dataset_val").set(thing_dataset_id_to_contiguous_id=self.to_contiguouse_id())
        # MetadataCatalog.get("coco_my_dataset_val").set(stuff_dataset_id_to_contiguous_id=self.to_contiguouse_id())
        # MetadataCatalog.get("coco_my_dataset_val").set(ignore_label=[255])
        # MetadataCatalog.get("coco_my_dataset_val").set(panoptic_json='/content/pan_gt_val.json')
        # MetadataCatalog.get("coco_my_dataset_val").set(panoptic_root='/content/')

    def get_metadata(self, ):
        meta = {}
        __COCO_CATEGORIES = [{'color': [224, 12, 193], 'isthing': 0, 'id': 12, 'name': 'light mold'},
                             {'color': [34, 11, 243], 'isthing': 0, 'id': 26, 'name': 'dark mold'},
                             {'color': [50, 10, 180], 'isthing': 0, 'id': 29, 'name': 'light stain'},
                             {'color': [150, 60, 80], 'isthing': 0, 'id': 30, 'name': 'dark stain'},
                             {'color': [53, 90, 180], 'isthing': 0, 'id': 45, 'name': 'liquid spillage'},
                             {'color': [183, 190, 0], 'isthing': 0, 'id': 66, 'name': 'peeling'},
                             {'color': [69, 169, 69], 'isthing': 1, 'id': 68, 'name': 'Broken pipe'},
                             {'color': [68, 11, 12], 'isthing': 0, 'id': 69, 'name': 'raising'},
                             {'color': [0, 165, 120], 'isthing': 1, 'id': 70, 'name': 'toilet'},
                             {'color': [173, 5, 142], 'isthing': 1, 'id': 71, 'name': 'pipe'},
                             {'color': [127, 167, 115], 'isthing': 1, 'id': 81, 'name': 'sink'},
                             {'color': [142, 108, 45], 'isthing': 1, 'id': 84, 'name': 'book'},
                             {'color': [196, 172, 0], 'isthing': 1, 'id': 85, 'name': 'clock'},
                             {'color': [78, 233, 129], 'isthing': 1, 'id': 94, 'name': 'other'},
                             {'color': [200, 40, 101], 'isthing': 0, 'id': 97, 'name': 'wet gateway'},
                             {'color': [9, 59, 100], 'isthing': 0, 'id': 96, 'name': 'drip'},
                             {'color': [210, 170, 100], 'isthing': 0, 'id': 109, 'name': 'curtain'},
                             {'color': [92, 136, 89], 'isthing': 0, 'id': 112, 'name': 'door-stuff'},
                             {'color': [218, 88, 184], 'isthing': 0, 'id': 118, 'name': 'floor-wood'},
                             {'color': [255, 160, 98], 'isthing': 0, 'id': 156, 'name': 'shelf'},
                             {'color': [137, 54, 74], 'isthing': 0, 'id': 171, 'name': 'wall-brick'},
                             {'color': [7, 246, 231], 'isthing': 0, 'id': 176, 'name': 'wall-tile'},
                             {'color': [255, 73, 97], 'isthing': 0, 'id': 181, 'name': 'window-other'},
                             {'color': [146, 139, 141], 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'},
                             {'color': [70, 130, 180], 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'},
                             {'color': [134, 199, 156], 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'},
                             {'color': [96, 36, 108], 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'},
                             {'color': [152, 251, 152], 'isthing': 0, 'id': 193, 'name': 'grass-merged'},
                             {'color': [102, 102, 156], 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'}]
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in __COCO_CATEGORIES if k["isthing"] == 1]
        thing_colors = [k["color"] for k in __COCO_CATEGORIES if k["isthing"] == 1]
        stuff_classes = [k["name"] for k in __COCO_CATEGORIES]
        stuff_colors = [k["color"] for k in __COCO_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(__COCO_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            # else:
            #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

            # in order to use sem_seg evaluator
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

        return meta

    def get_dataset(self, ):
        data_set = []
        panoptic_gt = 'pan_gt_val.json'  # ['images' --> ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
        #                                   'annotations' --> ['segments_info' --> ['id', 'category_id', 'iscrowd', 'bbox', 'area'],
        #                                                      'file_name',
        #                                                      'image_id']
        #                                   'categories'] --> ['supercategory', 'isthing', 'id', 'name']

        instances_gt = 'ins_gt_val.json'  # ['images' --> ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
        # 'annotations' --> ['bbox', 'category_id', 'image_id', 'id', 'segmentation', 'area']
        # 'categories' --> ['name', 'instance_count', 'def', 'synonyms', 'image_count', 'id', 'frequency', 'synset']

        with open(instances_gt, 'r') as file_ins:
            json_file_ins = json.load(file_ins)

        with open(panoptic_gt, 'r') as file_pan:
            json_file_pan = json.load(file_pan)

        for img in json_file_pan['images']:
            an_img = dict()
            an_img['file_name'] = img['file_name']
            an_img['width'] = img['width']
            an_img['height'] = img['height']
            an_img['image_id'] = img['id']
            an_img[
                'annotations'] = None  # list[dict] --> [{bbox: ?, bbox_mode: ?, category_id:[0, num_categories-1] ?, segmentation: ?, iscrowd: ?}]
            an_img['sem_seg_file_name'] = None  # str
            an_img['pan_seg_file_name'] = None  # str
            an_img['segments_info'] = None  # list[dict] --> [{id: ?, category_id: ?, iscrowd: ?}]

            the_img_ins_annots = list()
            the_img_ins_annotations_id_counter = 0
            for ins_annot in json_file_ins['annotations']:
                if ins_annot['image_id'] == img['id']:
                    an_ins_annot_for_the_img = dict()
                    an_ins_annot_for_the_img['bbox'] = ins_annot['bbox']
                    an_ins_annot_for_the_img['bbox_mode'] = 1
                    an_ins_annot_for_the_img['category_id'] = the_img_ins_annotations_id_counter
                    an_ins_annot_for_the_img['segmentation'] = ins_annot['segmentation']
                    an_ins_annot_for_the_img['iscrowd'] = 0
                    the_img_ins_annotations_id_counter += 1
                    the_img_ins_annots.append(an_ins_annot_for_the_img)

            an_img['annotations'] = the_img_ins_annots
            an_img['sem_seg_file_name'] = f'sem_seg_{img["file_name"].split(".")[0]}.png'
            an_img['pan_seg_file_name'] = f'{img["file_name"].split(".")[0]}.png'

            the_img_pan_segments_info = list()
            the_img_pan_annotations_id_counter = 0
            for annot_pan in json_file_pan['annotations']:
                if annot_pan['image_id'] == img['id']:
                    for seg_info in annot_pan['segments_info']:
                        an_pan_annot_for_the_img = dict()
                        an_pan_annot_for_the_img['id'] = seg_info['id']
                        an_pan_annot_for_the_img['category_id'] = the_img_pan_annotations_id_counter
                        an_pan_annot_for_the_img['iscrowd'] = seg_info['iscrowd']
                        the_img_pan_annotations_id_counter += 1
                        the_img_pan_segments_info.append(an_pan_annot_for_the_img)
            an_img['segments_info'] = the_img_pan_segments_info

            assert not any([information is None for information in
                            [an_img['segments_info'], an_img['pan_seg_file_name'], an_img['sem_seg_file_name'],
                             an_img['annotations']]])
            data_set.append(an_img)

        return data_set

    def to_contiguouse_id(self, ):
        panoptic_gt = 'pan_gt_val.json'
        with open(panoptic_gt, 'r') as file_pan:
            json_file_pan = json.load(file_pan)

        # ['categories'] --> ['supercategory', 'isthing', 'id', 'name']
        return {cat['id']: idx for idx, cat in enumerate(json_file_pan['categories'])}

    def eval(self, ):
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.mode = "eval"

        # self.model_names, self.raw_models, self.criteria = self.pipeline.set_up_model()
        self.raw_models = self.pipeline.initialize_model()
        self.model_names = self.raw_models.keys()

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        # load model during evaluation
        if self.opt['WEIGHT'] and os.path.isfile(self.opt['RESUME_FROM']):
            model_path = self.opt['RESUME_FROM']
            self.load_model(model_path)
        else:
            raise ValueError(f"Model not found: {model_path}")

        results = self._eval_on_set(self.save_folder)
        return results

    def _eval_on_set(self, save_folder):
        logger.info(f"Evaluation start ...")
        if self.opt['FP16']:
            from torch.cuda.amp import autocast
            with autocast():
                results = self.pipeline.evaluate_model(self, save_folder)
        else:
            results = self.pipeline.evaluate_model(self, save_folder)
        if self.opt['rank'] == 0:
            logger.info(results)
        return results

    def compute_loss(self, forward_func, batch):

        def forward(func, trainer, batch):
            if self.opt['FP16']:
                from torch.cuda.amp import autocast
                with autocast():
                    loss = func(trainer, batch)
            else:
                loss = func(trainer, batch)
            return loss

        loss = forward(forward_func, self, batch)
        return loss

    def backward_loss(self, loss, model_names=['default']):  # noqa: E252

        def backward(loss_tensor):
            if self.opt['FP16']:
                self.grad_scaler.scale(loss_tensor).backward()
            else:
                loss_tensor.backward()

        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps

        backward(loss)
        return loss

    def update_model(self, model_name='default'):
        if self.opt['FP16']:
            self.grad_scaler.unscale_(self.optimizers[model_name])
            self.grad_scaler.step(self.optimizers[model_name])
        else:
            self.optimizers[model_name].step()

        self.optimizers[model_name].zero_grad()
        self.train_params['optim_steps'][model_name] += 1
        self.lr_schedulers[model_name].step()

    def train_step(self, batch):
        self.grad_acc_batches.append(batch)  # support batch accumulation

        if self.is_gradient_accumulation_boundary():
            # set all modules and criteria into training mode
            for model_name in self.model_names:
                self.models[model_name].train()

            for name, param in self.models['default'].named_parameters():
                if ("transformer_ffn_layers_adapters" in name or
                        "transformer_self_attention_layers_adapters" in name or
                        "transformer_cross_attention_layers_adapters" in name or
                        "predictions_heads_mask_embs_adapters" in name):
                    continue

                if not ("sem_seg_head.predictor.transformer" in name and "norm" in name):
                    param.requires_grad = False

            assert len(self.grad_acc_batches) == self.grad_acc_steps

            total_batch_sample = 0
            for batch_index, batch in enumerate(self.grad_acc_batches):
                loss_info, sample_size_info, extra_info = \
                    self.pipeline.forward_step(self,
                                               batch,
                                               self.grad_acc_batches,
                                               batch_index,
                                               is_distributed=(self.opt['world_size'] > 1))

                self.train_loss.update_iter(loss_info)
                total_batch_sample += sample_size_info['num_samples']

            if self.opt['FP16']:
                # Update GradScaler after an effective batch
                self.grad_scaler.update()

            # update losses and item counts of an effective batch to the AverageMeters
            if self.opt['world_size'] > 1:
                total_batch_sample = torch.tensor(total_batch_sample).to(self.opt['device'])
                torch.distributed.all_reduce(total_batch_sample, torch.distributed.ReduceOp.SUM)
                total_batch_sample = total_batch_sample.item()

            self.train_params['total_batch_size'] += total_batch_sample
            self.grad_acc_batches = []

        self.train_params['num_updates'] += 1

    def init_train(self):
        self.mode = "train"
        logger.info('-------------------------------------------------------')
        logger.info("Training on rank: {}".format(self.opt['rank']))

        self.raw_models = self.pipeline.initialize_model()
        self.model_names = list(self.raw_models.keys())

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        self.train_dataloaders = self.pipeline.get_dataloaders(self, 'train', is_evaluation=False)
        self.train_params = {
            "updates_per_epoch": len(self.train_dataloaders),
            "total_batch_size": 0,
            "num_updates": 0,
            "optim_steps": {module_name: 0 for module_name in self.model_names},
            "start_epoch_idx": 0,
            "start_batch_idx": 0,
            "current_epoch_idx": 0,
            "current_batch_idx": 0,
            "resume_epoch_idx": 0,
        }

        self.train_loss = LossMeter()
        self.grad_acc_batches = []

        if self.opt['CUDA']:
            torch.cuda.empty_cache()

        self.create_optimizer_and_scheduler()
        self.models = {model_name: self.raw_models[model_name] for model_name in self.model_names}
        self._initialize_ddp()

        if self.opt.get('WEIGHT', False):
            self.load_weight(self.opt['RESUME_FROM'], must_exist=True)
        if self.opt.get('RESUME', False):
            self.load_checkpoint(self.opt['RESUME_FROM'], must_exist=True)

        ######################
        # Start the main loop
        ######################
        if self.opt['rank'] == 0:
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num of GPUs = {self.opt['world_size']}")
            logger.info(f"  Num Epochs = {self.opt['SOLVER']['MAX_NUM_EPOCHS']}")
            logger.info(f"  Num of Mini Batches per Epoch = {self.train_params['updates_per_epoch']}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch']}")
            logger.info(f"  Gradient Accumulation steps = {self.grad_acc_steps}")
            logger.info(
                f"  Total optimization steps = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch'] // self.grad_acc_steps}")

    def train(self):
        """
        Training
        """
        self.init_train()
        current_optim_steps = self._get_and_validate_current_optim_steps()
        num_epochs = self.opt['SOLVER']['MAX_NUM_EPOCHS']

        if self.opt.get('EVAL_AT_START', False):
            results = self._eval_on_set(self.save_folder)
            if self.opt['rank'] == 0 and self.opt['WANDB']:
                wandb.log(results)

        train_prev_logged_time = datetime.now()
        for epoch in range(self.train_params['start_epoch_idx'], num_epochs):
            self.train_params['current_epoch_idx'] = epoch
            logger.info(f"Start epoch: {epoch} training.")

            epoch_start_time = datetime.now()
            for batch_idx, batch in enumerate(self.train_dataloaders):
                if self.train_params['current_epoch_idx'] == self.train_params['start_epoch_idx']:
                    if batch_idx < self.train_params['start_batch_idx']:  # skip the first few batches for resuming
                        continue

                self.train_params['current_batch_idx'] = batch_idx
                prev_optim_steps = current_optim_steps
                prev_total_batch_size = self.train_params['total_batch_size']

                # update
                self.prev_optim_steps = prev_optim_steps
                self.train_step(batch)

                current_optim_steps = self._get_and_validate_current_optim_steps()

                # logging
                if prev_optim_steps != current_optim_steps:  # an optimizer update was made
                    # log_first = self.opt.get("LOG_FIRST", 10)
                    log_first = 2
                    log_every = 1
                    # log_every = self.opt.get("LOG_EVERY", 100)
                    if (current_optim_steps % log_every == 0) or (
                            epoch == 0 and current_optim_steps <= log_first):  # print logging

                        last_lr = {}
                        for module_name in self.model_names:
                            last_lr[module_name] = self.lr_schedulers[module_name].get_last_lr()[0]

                        train_time_delta = (datetime.now() - train_prev_logged_time).total_seconds()
                        train_prev_logged_time = datetime.now()
                        MB = 1024.0 * 1024.0
                        memory = torch.cuda.max_memory_allocated() / MB

                        if self.opt['rank'] == 0:
                            if self.opt['WANDB']:
                                # log for wandb
                                wb_loss_info = {key: obj.val for key, obj in self.train_loss.losses.items()}
                                wandb.log(wb_loss_info, step=self.prev_optim_steps)

                            # log for terminal
                            logger.info(f"epochs[{epoch:6}] optim steps[{current_optim_steps:.0f}] "
                                        f"learning rate[{', '.join([f'{key}: {val:.5e}' for key, val in last_lr.items()])}] "
                                        f"train loss[{', '.join([f'{key}: {obj.val:.5f}/{obj.avg:.5f}' for key, obj in self.train_loss.losses.items()])}] "
                                        # f"total_loss[{total_loss:.5f}/{total_loss_avg:.5f} "
                                        f"items per batch[{self.train_params['total_batch_size'] - prev_total_batch_size}] "
                                        f"items per second[{(self.train_params['total_batch_size'] - prev_total_batch_size) / train_time_delta:.2f}] "
                                        f"total items[{self.train_params['total_batch_size']}] "
                                        f"mini batches[{self.train_params['num_updates']:6}] "
                                        f"memory[{memory:.0f}] "
                                        f"epoch remaining[{str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (self.train_params['updates_per_epoch'] - batch_idx - 1)).split('.')[0]}]")

                # evaluate and save ckpt every epoch
                if batch_idx + 1 == self.train_params['updates_per_epoch']:
                    # self.save_checkpoint(self.train_params['num_updates'])
                    # results = self._eval_on_set(self.save_folder)
                    # by me
                    self.models['default'].save_pretrained('/content/data/output/test/')

                    if self.opt['rank'] == 0 and self.opt['WANDB']:
                        wandb.log(results)
                    break

            logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
            logger.info(f"PROGRESS: {100.0 * (epoch + 1) / num_epochs:.2f}%")
            logger.info(f"Config files are at {self.opt['conf_files']}")
