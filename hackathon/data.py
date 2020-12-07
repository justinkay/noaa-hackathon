import os
import glob

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

_DATASETS = {
    
    # original labels:
    # Abi's fish point annotations + buffered bounding boxes 220 x 220
#     "rockfish_train": {
#         "imgs_loc": "rockfish/OCNMS_port/",
#         "labels_loc": "rockfish/coco/rockfish_train.json"
#     },
#     "rockfish_val": {
#         "imgs_loc": "rockfish/OCNMS_port/",
#         "labels_loc": "rockfish/coco/rockfish_val.json"
#     },
    
    # fish point labels + Lynker manual bounding boxes + Jimmy PhD annotations
    # generated Sunday by Jake
#     "jimmy_fct_lynker_train": {
#         "imgs_loc": "",
#         "labels_loc": "coco/jimmy_fct_lynker_train.json"
#     },
#     "jimmy_fct_lynker_val": {
#         "imgs_loc": "",
#         "labels_loc": "coco/jimmy_fct_lynker_val.json"
#     },
    
    # Jake's Sunday labels minus poorly annotated sets
    "jimmy_fct_lynker_train_fixed_6k_22k": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_fct_lynker_train_fixed_6k_22k.json"
    },
    # use this as the final validation set for all experiments for consistency !
    "all_val": {
        "imgs_loc": "",
        "labels_loc": "rockfish_labels/coco/jimmy_fct_lynker_val_fixed_1484_4653.json"
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
