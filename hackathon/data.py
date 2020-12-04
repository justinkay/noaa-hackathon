import os
import glob

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

_DATASETS = {
    "rockfish_train": {
        "imgs_loc": "rockfish/OCNMS_port/",
        "labels_loc": "rockfish/coco/rockfish_train.json"
    },
    "rockfish_val": {
        "imgs_loc": "rockfish/OCNMS_port/",
        "labels_loc": "rockfish/coco/rockfish_val.json"
    }
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
