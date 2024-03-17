# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import numpy as np
import os
import glob
from typing import List, Tuple, Union
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from utils.constants import BDD_SEM, COCO_PANOPTIC_CLASSES

__all__ = ["load_scannet_instances", "register_scannet_context"]


def load_bdd_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], annots):
    """
    Load BDD annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
        annots: to get id of image
    """
    # img_folder = os.path.join(dirname, 'images', '10k', split)
    img_folder = os.path.join(dirname, 'images')
    img_pths = sorted(glob.glob(os.path.join(img_folder, '*')))
    
    # sem_folder = os.path.join(dirname, 'labels', 'sem_seg', 'masks', split)
    sem_folder = os.path.join(dirname, 'semantic')
    sem_pths = sorted(glob.glob(os.path.join(sem_folder, '*.png')))

    assert len(img_pths) == len(sem_pths)

    with PathManager.open(annots) as f:
        json_info = json.load(f)

    images = json_info['images']
        
    dicts = []
    for img_pth, sem_pth in zip(img_pths, sem_pths):

        file_name = img_pth.split('/')[-1]
        file_name_WO_ext = file_name.rsplit(".", 1)[0]

        the_img = list(filter(lambda imgs: imgs['file_name'].rsplit(".", 1)[0] == file_name_WO_ext, images))
        _id = the_img[0]['id']

        r = {
            "file_name": img_pth,
            "sem_seg_file_name": sem_pth,
            "image_id": _id,
        }
        dicts.append(r)
    return dicts


def register_bdd_context(name, dirname, split, annots, class_names=COCO_PANOPTIC_CLASSES):
    DatasetCatalog.register(name, lambda: load_bdd_instances(name, dirname, split, class_names, annots))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        dirname=dirname,
        split=split,
        ignore_label=[255],
        thing_dataset_id_to_contiguous_id={},
        class_offset=0,
        keep_sem_bgd=False
    )


def register_all_sunrgbd_seg(root, annots):
    SPLITS = [
            ("bdd10k_val_sem_seg", "validation", "val"),
        ]
        
    for name, dirname, split in SPLITS:
        register_bdd_context(name, os.path.join(root, dirname), split, annots)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


# _root = os.getenv("DATASET", "datasets")
# register_all_sunrgbd_seg(_root)