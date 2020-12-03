import argparse
import cv2
import glob
import os
import torch
import logging
import itertools
from fvcore.common.file_io import PathManager

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import detectron2.utils.comm as comm

from hackathon.data import register_image_dataset

_MODELS = {
    # model trained COCO -> all bottomfish data
    "frcnn-bfish": { "weights": "frcnn-bfish/model_0005159.pth", # just use the first one for now
                     "config": "frcnn-bfish/config.yaml" },
    
    # model trained COCO -> fishnet v0.1.2
    "frcnn-fishnet": { "weights": "detectron2-baselines-jul2020/frcnn-r101/model_0022945.pth",
                     "config": "detectron2-baselines-jul2020/frcnn-r101/config.yaml" },
    
    # model trained COCO -> all mouss data
    "frcnn-coco-mouss": {},
    
    # model trained COCO -> fishnet v0.1.2 -> all mouss data
    "frcnn-fishnet-mouss": {},
}

def get_model_for_eval(model_name, models_dir, score_threshold=0.05, nms_threshold=0.5):
    weights = os.path.join(models_dir, _MODELS[model_name]["weights"])
    config = os.path.join(models_dir, _MODELS[model_name]["config"])
    
    cfg = get_cfg()
    
    if config is not None:
        cfg.merge_from_file(config)
        
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
        
    # the configurable part
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    default_setup(cfg, {})
    
    return cfg

def evaluate_coco_dataset(cfg, dataset_name, distributed=False):
    """
    Perform evaluation on a Detectron2 dataset using COCO metrics.
    This is how we benchmark our models currently.
    """
    model = DefaultPredictor(cfg).model
    evaluator = COCOEvaluator(dataset_name, cfg, distributed)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(model, val_loader, evaluator)
    
def evaluate_img_dir(cfg, input_dir, height=None, width=None, distributed=False, extension=".png"):
    """
    Get predictions for an image directory using Detectron2, and save pickled
    results (predictions of Instances) to the same directory.
    
    Assumes unique directory names, as the directory name is also used as the 
    dataset name for detectron. This could be changed later to use a timestamp
    or something if desired.
    
    Note: "evaluation" (calculation of metrics) is not performed.
    """
    dataset_name = os.path.basename(os.path.dirname(input_dir))
    register_image_dataset(dataset_name, input_dir, height=height, width=width, img_extension=extension)
    model = DefaultPredictor(cfg).model
    evaluator = SimpleEvaluator(dataset_name, cfg, distributed=distributed, output_dir=input_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    print(val_loader)
    inference_on_dataset(model, val_loader, evaluator)
    
class SimpleEvaluator(DatasetEvaluator):
    """
    An evaluator which does not really evaluate. Used for distributed prediction
    with inference_on_dataset.
    
    Adapted from detectron2 COCOEvaluator.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input_, output in zip(inputs, outputs):
            prediction = { "image_id": input_["image_id"], "file_name": input_["file_name"] }

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances #instances_to_coco_json(instances, input_["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[SimpleEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        return self._predictions
    
    def load_preds(self):
        predictions = []
        pth = os.path.join(self._output_dir, "instances_predictions.pth")
        return SimpleEvaluator.load_preds(pth)
    
    @staticmethod
    def load_preds(path_to_file):
        predictions = []
        with PathManager.open(path_to_file, "rb") as f:
            predictions = torch.load(f)
        return predictions
    
def eval_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="frcnn-bfish", 
                        help="name of model to use, from .evaluate._MODELS")
    parser.add_argument("--models_dir", default="../models/", help="location of directory containing models, " +
                        "resulting in correct paths in evaluate.py")
    parser.add_argument("--score_thresh", default=0.05, help="confidence threshold for object detector.")
    parser.add_argument("--nms_thresh", default=0.5, help="nms threshold for object detector, if applicable.")
    parser.add_argument("--src", default="", help="input location of img frames")
    
    return parser
    
def main(args):
    cfg = get_model_for_eval(args.model, args.models_dir, float(args.score_thresh), float(args.nms_thresh))
    
    # assume all imgs the same resolution
    jpgs = glob.glob(args.src + "/*.jpg") 
    pngs = glob.glob(args.src + "/*.png")
    if len(jpgs):
        extension = ".jpg"
    else:
        extension = ".png"
    files = jpgs + pngs
    sample_im_p = files[0]
    sample_im = cv2.imread(sample_im_p)
    h, w, c = sample_im.shape
    
    evaluate_img_dir(cfg, args.src, height=h, width=w, extension=extension)
    
if __name__ == "__main__":
    args = eval_argument_parser().parse_args()
    main(args)