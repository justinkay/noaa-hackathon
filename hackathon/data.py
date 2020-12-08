import os
import glob

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

_DATASETS = {
    # use this as the final validation set for all experiments for consistency !
    "val": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_lynker_val_5k_25k_v2.json"
    },
    "train_pct10": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_lynker_train_5k_25k_10pct_v2.json"
    },
    "train_pct25": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_lynker_train_5k_25k_25pct_v2.json"
    },
    "train_pct50": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_lynker_train_5k_25k_25pct_v2.json"
    },
    "train_pct100": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_lynker_train_5k_25k_v2.json"
    },
}

def register_all(data_dir="../data"):
    for k in _DATASETS.keys():
        register_dataset(k, data_dir)
    return tuple(_DATASETS.keys())

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

def register_image_dataset(dataset_name, img_dir, img_extensions=[".jpg", ".png", ".tif"], height=None, width=None):
    """
    Register a Dataset in Detectron2 for a directory of images with no annotations.
    For your own good, provide a height and width...
    """
    try:
        del DatasetCatalog._REGISTERED[dataset_name]
    except KeyError:
        pass
    DatasetCatalog.register(dataset_name, lambda: _get_simple_dataset_dicts(img_dir, img_extensions, height, width))
    
def _get_simple_dataset_dicts(img_dir, img_extensions=[".jpg", ".png", ".tif"], height=None, width=None):
    """Create a simple dataset_dict from a directory of images."""
    dataset_dicts = []
    
    files = []
    for img_extension in img_extensions:
        files += [f for f in glob.glob(img_dir + "/*" + img_extension)]
    files = sorted(files)
    
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
