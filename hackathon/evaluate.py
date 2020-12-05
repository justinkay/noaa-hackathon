import argparse
import cv2
import glob
import os
import torch
import logging
import itertools
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.file_io import PathManager

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_setup, launch, default_argument_parser
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import detectron2.utils.comm as comm

from hackathon.data import register_image_dataset

_MODELS = {
    # retinanet w/ efficientnet-b0-bifpn backbone, trained on pts + buffered boxes, 2x schedule (36 epochs)
    "en-b0-ptannos-2x": "pt_annos/en_b0_2x",
}

def get_model_for_eval(model_name, models_dir, score_threshold=0.05, nms_threshold=0.5, device="cuda"):
    weights = os.path.join(os.path.join(models_dir, _MODELS[model_name]), "model_final.pth")
    config = os.path.join(os.path.join(models_dir, _MODELS[model_name]), "config.yaml")
    
    cfg = get_cfg()
    
    # efficientnet models
    if "en-b" in model_name:
        from detectron2_backbone import backbone
        from detectron2_backbone.config import add_backbone_config
        add_backbone_config(cfg)
    
    if config is not None:
        cfg.merge_from_file(config)
        
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
        
    # the configurable part
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    
    cfg.MODEL.DEVICE = device
    default_setup(cfg, {})
    
    return cfg

def evaluate_coco_dataset(cfg, dataset_name):
    """
    Perform evaluation on a Detectron2 dataset using COCO metrics.
    Distributed is False, like in Detectron2, because it affects evaluation
    metrics.
    """
    model = DefaultPredictor(cfg).model
    evaluator = COCOEvaluator(dataset_name, cfg, distributed=False)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(model, val_loader, evaluator)
    
def evaluate_img_dir(cfg, input_dir, height=None, width=None, distributed=False, 
                     output_fname="instances_predictions.pth"):
    """
    Get predictions for an image directory using Detectron2, and save pickled
    results (predictions of Instances) to the same directory.
    
    Assumes unique directory names, as the directory name is also used as the 
    dataset name for detectron. This could be changed later to use a timestamp
    or something if desired.
    
    Note: "evaluation" (calculation of metrics) is not performed.
    """
    dataset_name = os.path.basename(os.path.dirname(input_dir))
    register_image_dataset(dataset_name, input_dir, height=height, width=width)
    model = DefaultPredictor(cfg).model
    
    if distributed and comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    
    evaluator = SimpleEvaluator(dataset_name, cfg, distributed=distributed, output_dir=input_dir, output_fname=output_fname)
    if distributed:
        val_loader = build_detection_test_loader_multi(cfg, dataset_name)
    else:
        val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(model, val_loader, evaluator)
    
def build_detection_test_loader_multi(cfg, dataset_name, mapper=None):
    """
    Modified version of detectron2.data.build_detection_test_loader which allows for multi-gpu
    inference.
    """
    from detectron2.data.build import get_detection_dataset_dicts, DatasetFromList, DatasetMapper, MapDataset, InferenceSampler, trivial_batch_collator, build_batch_data_loader
    
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    
    # from train loader
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    
class SimpleEvaluator(DatasetEvaluator):
    """
    An evaluator which does not really evaluate. Used for distributed prediction
    with inference_on_dataset.
    
    Adapted from detectron2 COCOEvaluator.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, output_fname="instances_predictions.pth"):
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
        self._output_fname = output_fname

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
            file_path = os.path.join(self._output_dir, self._output_fname)
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
    
def find_subdirs_with_images(base_dir):
    """Recursively find subdirectories that contain ims with extension {.tif, .jpg, .png}"""
    inc = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if not f.startswith(".")]
    ims = [f for f in inc if os.path.isfile(f) if f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png")]
    subdirs = [f for f in inc if os.path.isdir(f)]
    
    dirs_with_ims = []
    if len(ims):
        dirs_with_ims.append(base_dir)
    
    for subdir in subdirs:
        dirs_with_ims += find_subdirs_with_images(subdir)
    
    return dirs_with_ims
    
def eval_argument_parser():
    # get parse with Detectron2 default commands
    parser = default_argument_parser()
    parser.add_argument("--model", default="en-b0-ptannos-2x", help="name of model to use, from .evaluate._MODELS")
    parser.add_argument("--models_dir", default="../models/", help="location of directory containing models, " +
                        "resulting in correct paths in evaluate.py")
    parser.add_argument("--score_thresh", default=0.05, help="confidence threshold for object detector.")
    parser.add_argument("--nms_thresh", default=0.5, help="nms threshold for object detector, if applicable.")
    parser.add_argument("--src", default="", help="input location of img frames")
    parser.add_argument("--subdirs", action="store_true", help="Evaluate on all subdirectories of src with images")
    
    return parser
    
def main(args):
    cfg = get_model_for_eval(args.model, args.models_dir, float(args.score_thresh), float(args.nms_thresh))
    cfg.SOLVER.IMS_PER_BATCH = 2 * args.num_gpus
    
    if args.subdirs:
        dirs = find_subdirs_with_images(args.src)
        print("Performing inference on", len(dirs), "directories")
    else:
        dirs = [args.src]
    
    # assumes all ims in a dir are the same resolution
    for d in dirs:
        inc = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith(".")]
        ims = [f for f in inc if os.path.isfile(f) if f.endswith(".tif") or f.endswith(".jpg") or f.endswith(".png")]
        print("Inference on", len(ims), "images in", d)
        h, w = cv2.imread(ims[0]).shape[:2]

        output_fname = args.model + "_preds.pth"
        evaluate_img_dir(cfg, d, height=h, width=w, distributed=True, output_fname=output_fname)
    
if __name__ == "__main__":
    args = eval_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )