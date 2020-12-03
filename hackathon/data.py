import os
import glob

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

_DATASETS = {
    # VIAME
    "ehu": {
        "imgs_loc": "Ehu",
        "labels_loc": "Ehu/Ehu_coco.json"
    },
    "gindai2": {
        "imgs_loc": "Gindai2",
        "labels_loc": "Gindai2/Gindai2_coco.json"
    },
    "gindai3": {
        "imgs_loc": "Gindai3",
        "labels_loc": "Gindai3/Gindai3_coco.json"
    },
    "lehi": {
        "imgs_loc": "Lehi",
        "labels_loc": "Lehi/Lehi_coco.json"
    },
    "lehi1": {
        "imgs_loc": "Lehi1",
        "labels_loc": "Lehi1/Lehi1_coco.json"
    },
    "lehi2": {
        "imgs_loc": "Lehi2",
        "labels_loc": "Lehi2/Lehi2_coco.json"
    },
    "onaga1": {
        "imgs_loc": "Onaga1",
        "labels_loc": "Onaga1/Onaga1_coco.json"
    },
    "onaga2": {
        "imgs_loc": "Onaga2",
        "labels_loc": "Onaga2/Onaga2_coco.json"
    },
    "onaga3": {
        "imgs_loc": "Onaga3",
        "labels_loc": "Onaga3/Onaga3_coco.json"
    },
    
    # data-challenge-training
    "mouss_seq0": {
        "imgs_loc": "Girder/data-challenge-training-imagery/mouss_seq0",
        "labels_loc": "Girder/data-challenge-training-annotations/mouss_seq0_training.mscoco.oneclass.json"
    },
    "mouss_seq1": {
        "imgs_loc": "Girder/data-challenge-training-imagery/mouss_seq1",
        "labels_loc": "Girder/data-challenge-training-annotations/mouss_seq1_training.mscoco.oneclass.json"
    },
    "afsc_seq0": {
        "imgs_loc": "Girder/data-challenge-training-imagery/afsc_seq0",
        "labels_loc": "Girder/data-challenge-training-annotations/afsc_seq0.mscoco.oneclass.json"
    },
    "mbari_seq0": {
        "imgs_loc": "Girder/data-challenge-training-imagery/mbari_seq0",
        "labels_loc": "Girder/data-challenge-training-annotations/mbari_seq0_training.mscoco.oneclass.json"
    },
        
    # SE Quadcam
    "JRS_1": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_1",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_1/JRS_1_coco.json"
    },
    "JRS_2": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_2",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_2/JRS_2_coco.json"
    },
    "JRS_3": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_3",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_3/JRS_3_coco.json"
    },
    "JRS_4": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_4",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_4/JRS_4_coco.json"
    },
    "JRS_5": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_5",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_5/JRS_5_coco.json"
    },
    "JRS_6": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_6",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_6/JRS_6_coco.json"
    },
    "JRS_7": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_7",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_7/JRS_7_coco.json"
    },
    "JRS_8": {
        "imgs_loc": "US_SE_Quadcam_Sampler/JRS_8",
        "labels_loc": "US_SE_Quadcam_Sampler/JRS_8/JRS_8_coco.json"
    },
}

def register_all(data_dir="../data"):
    for k in _DATASETS.keys():
        register_dataset(k, data_dir)
    return tuple(_DATASETS.keys())

def register_mouss(data_dir="../data"):
    mouss_sets = ["ehu", "gindai2", "gindai3", "lehi", "lehi1", "lehi2", "onaga1", "onaga2", "onaga3",
                 "mouss_seq0", "mouss_seq1"]
    for d in mouss_sets:
        register_dataset(d, data_dir)
    return tuple(mouss_sets)

def register_dataset(dataset_name, data_dir="../data"):
    """Register a dataset in _DATASETS, or re-register it if it already exists."""
    dataset = _DATASETS[dataset_name]
    imgs_path = os.path.join(data_dir, dataset["imgs_loc"])
    labels_path = os.path.join(data_dir, dataset["labels_loc"])

    try:
        del DatasetCatalog._REGISTERED[dataset_name]
    except KeyError:
        pass
    register_coco_instances(dataset_name, {}, labels_path, imgs_path)
    
    return dataset_name

def register_image_dataset(dataset_name, img_dir, img_extension=".jpg", height=None, width=None):
    """
    Register a Dataset in Detectron2 for a directory of images with no annotations.
    
    For your own good, provide a height and width...
    """
    try:
        del DatasetCatalog._REGISTERED[dataset_name]
    except KeyError:
        pass
    DatasetCatalog.register(dataset_name, lambda: _get_simple_dataset_dicts(img_dir, img_extension, height, width))
    
def _get_simple_dataset_dicts(img_dir, img_extension=".jpg", height=None, width=None):
    """Create a simple dataset_dict from a directory of images.
    """
    dataset_dicts = []
    
    files = sorted([f for f in glob.glob(img_dir + "/*" + img_extension)])
    
    for i, f in enumerate(files):
        record = {}
        record['file_name'] = f
        
        if not height and not width:
            h, w = cv2.imread(f).shape[:2]
        else:
            h = height
            w = width
        record["height"] = h
        record["width"] = w
        record["image_id"] = i
        
        dataset_dicts.append(record)
        
    return dataset_dicts
