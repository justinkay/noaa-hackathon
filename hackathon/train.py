import logging
import os
from collections import OrderedDict
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.engine import default_setup, DefaultTrainer, hooks, default_argument_parser, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from hackathon.data import register_all, register_mouss


_MODELS = {
    "frcnn-r101": { "weights": "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
                     "config": "/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" },
    "frcnn-r50": { "weights": "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
                     "config": "/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" },
    "retinanet-r50": { "weights": "detectron2://COCO-Detection/retinanet_R_50_FPN_3x/137849486/model_final_4cafe0.pkl",
                     "config": "/COCO-Detection/retinanet_R_50_FPN_3x.yaml" },
    "retinanet-r101": { "weights": "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl",
                     "config": "/COCO-Detection/retinanet_R_101_FPN_3x.yaml" },
}


def get_training_config(data_dir, configs_dir, model="frcnn-r101", device="cuda", num_gpus=8, loss="smooth_l1", weights_path=None, do_default_setup=True, lr=None, mouss_only=False):
    """
    Basic configuration setup for training/validating using Fishnet.ai data.
    
    Training schedule follows COCO defaults by matching the number of epochs.
    
    Arguments:
        data_dir: path to data. should contain subdirectory fishnetai/
        configs_dir: path to detectron2/configs
        model: one of { 'frcnn-r101', 'frcnn-r50', 'retinanet-r50', 'retinanet-r101' }
        device: one of { 'cuda', 'cpu' }
        bs: batch size; corresponds to SOLVER.IMS_PER_BATCH
        lr: base learning rate, corresponds to SOLVER.BASE_LR
        weights_path: used for custom weights, use weights in _MODELS if None
    """
    weights_path = weights_path or _MODELS[model]["weights"]
    config_path = configs_dir + _MODELS[model]["config"]

    cfg = get_training_config_basic(config_path, weights_path, data_dir, device, num_gpus, loss, lr, mouss_only)
    
    # name output directory after model name
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + model
    
    # run some default setup from Detectron2
    # note: this eliminates:
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    if do_default_setup:
        default_setup(cfg, {})

    return cfg

def get_training_config_basic(config_path, weights_path, data_dir, device='cuda', num_gpus=8, loss="smooth_l1", lr=None, mouss_only=False):
    """
    Note: does not run default_setup()
    """
    cfg = get_cfg()
    
    # bs scaling based on num gpus
    bs = num_gpus * 2
    
    # lr scaling based on num gpus unless a custom lr is passed in
    if not lr:
        lr = 0.02 * bs / 16
        
        # reduce Retinanet LR; TODO kind of hacky
        if 'retinanet' in weights_path:
            lr = lr / 2.0
    
    # do this first, because we will overwrite
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    
    cfg.MODEL.DEVICE = device
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.BASE_LR = lr
    
    if loss == "giou":
        cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
        cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 2.0 # Detectron2 default value (TODO)
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 10.0 # Detectron2 default value (TODO)
    
    # now the default stuff:

    # use a seed for reproducibility
    cfg.SEED = 42
    
    # set up data
    if mouss_only:
        datasets = register_mouss(data_dir)
    else:
        datasets = register_all(data_dir)
    val_datasets = ("lehi1", "gindai2", "onaga1")
    train_datasets = tuple([d for d in datasets if d not in val_datasets])
    
    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = val_datasets
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    
    # fix memory errors: num workers * dataset size = memory required
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # based on Detectron defaults, which trained COCO for 18 epochs
    # and which decrease by gamma at 12 and 16 epochs when training COCO
    num_imgs = sum([len(DatasetCatalog.get(train_key)) for train_key in train_datasets])
    print("num_imgs",num_imgs)
    epoch_size = int(num_imgs / bs)
    cfg.SOLVER.MAX_ITER = 18*epoch_size
    cfg.SOLVER.STEPS = (12*epoch_size, 16*epoch_size)
    
    # evaluate and save after every epoch
    # makes this a multiple of PeriodicWriter default period so that eval metrics always get written to 
    # Tensorboard / WandB
    cfg.TEST.EVAL_PERIOD = epoch_size // 20 * 20
    cfg.SOLVER.CHECKPOINT_PERIOD = epoch_size // 20 * 20
    
    # don't get rid of empty images; I'm not sure why 'True' is the default
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    
    return cfg

def remove_solver_states(model_file):
    """If it does not exist, write a copy of this model which does not have a scheduler, 
    optimizer, or iteration saved.
    Return: the path to the new file."""
    suffix = "_wo_solver_states"
    
    if suffix in model_file:
        return model_file
    
    model = torch.load(model_file)
    del model["optimizer"]
    del model["scheduler"]
    del model["iteration"]
    print("saving model:", model)

    filename_wo_ext, ext = os.path.splitext(model_file)
    output_file = filename_wo_ext + suffix + ext
    torch.save(model, output_file)
    print("The model without solver states is saved to {}".format(output_file))
    return output_file

def convert_cfg_to_stage(cfg, stage):
    """Modify this config to start at the second stage. Do this by decreasing the base
    learning rate and adjusting solver steps accordingly."""
    weights_path = remove_solver_states(cfg.MODEL.WEIGHTS)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * (cfg.SOLVER.GAMMA**stage)
    cfg.SOLVER.WARMUP_ITERS = 0
    original_stage_start = cfg.SOLVER.STEPS[stage - 1]
    cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER - original_stage_start
    
    new_steps = []
    for step in cfg.SOLVER.STEPS[stage:]:
        new_steps.append(step - original_stage_start)
    cfg.SOLVER.STEPS = tuple(new_steps)

def get_coco_trainer(cfg, resume=False):
    trainer = COCOTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer

class COCOTrainer(DefaultTrainer):
    """
    A basic Trainer with a COCOEvaluator, set up for distributed training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, distributed=True)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
def training_argument_parser():
    # get parse with Detectron2 default commands
    parser = default_argument_parser()

    parser.add_argument("--model", default="frcnn-r101", 
                        help="name of model to train, { 'frcnn-r101', 'frcnn-r50', 'retinanet-r50', 'retinanet-r101', 'context-rcnn-r101' }")
    parser.add_argument("--data-dir", default="../data", metavar="FILE", help="path to data/")
    parser.add_argument("--configs-dir", default="../lib/detectron2/configs", metavar="FILE", help="path to detectron2/configs")
    parser.add_argument("--device", default="cuda", help="{ 'cuda', 'cpu' }")
    parser.add_argument("--loss", default="smooth_l1", help="loss function to use, { 'smooth_l1', 'giou' }")
    parser.add_argument("--wandb", help="name of wandb project to log tensorboard results")
    parser.add_argument("--stage", default=0, type=int, help="which stage of training to start at. corresponds to cfg.SOLVER.STEPS")
    parser.add_argument("--weights", default=None, help="path to pth file for alternate weights initialization. used for models with alternate (non-COCO) pretraining, etc.")
    parser.add_argument("--lr", default=None, type=float, help="input custom learning rate")
    parser.add_argument("--mouss", action="store_true", help="train on MOUSS data only")
    
    return parser

def main(args):
    """
    Valid args are:
        model: one of _MODELS.keys
        data-dir: location of data/
        configs-dir: location of detectron2/configs/
        resume: resumes training if present
        num-gpus=x
        + any any additional Detectron2 args you want
    """
    if args.wandb:
        import wandb; wandb.init(project=args.wandb, sync_tensorboard=True)
        
    # TODO add args.skip_first_stage here, since default_setup() prints invalid values in this case
    cfg = get_training_config(model=args.model, data_dir=args.data_dir, configs_dir=args.configs_dir, 
                                device=args.device, num_gpus=args.num_gpus, loss=args.loss, weights_path=args.weights, 
                                do_default_setup=False, lr=args.lr, mouss_only=args.mouss)
    
    if args.stage > 0:
        convert_cfg_to_stage(cfg, args.stage)
        
    default_setup(cfg, {})
    
    trainer = get_coco_trainer(cfg, args.resume)
        
    return trainer.train()

if __name__ == "__main__":
    args = training_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
