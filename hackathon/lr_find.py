from torch_lr_finder import LRFinder, TrainDataLoaderIter

from detectron2.solver.build import build_optimizer
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.events import EventStorage

from hackathon.train import get_coco_trainer

class DetectronTrainDataLoaderIter(TrainDataLoaderIter):
    """
    Dataloader iterator required for this LRFinder implementation. 
    
    Just moves all the data from the train_loader into inputs. Labels are populated with 
    the category_ids but these never actually get used. 
    """
    
    def inputs_labels_from_batch(self, batch_data):
        if not isinstance(batch_data, list) and not isinstance(batch_data, tuple):
            raise ValueError(
                "Your batch type is not supported: {}. Please inherit from "
                "`TrainDataLoaderIter` or `ValDataLoaderIter` and override the "
                "`inputs_labels_from_batch` method.".format(type(batch_data))
            )

        inputs = batch_data
        
        labels = []
        for item in batch_data:
            labels.append(item["instances"].get("gt_classes"))

        return inputs, labels

class COCOLRFinder:
    """
    Implementation of PyTorch learning rate finder (https://github.com/davidtvs/pytorch-lr-finder)
    for our detectron2 configurations.
    
    Requires both a config and trainer object. For some reason trying to do both in one script 
    results in the registered rcnn components throwing errors. Maybe this would fix itself if the 
    rcnn components in train.py get moved to a new file?
    """
    def __init__(self, cfg, base_lr_scale=1000, device="cuda"):
        self.cloned_cfg = cfg.clone()
        self.trainer = get_coco_trainer(self.cloned_cfg)
        self.cloned_cfg.SOLVER.BASE_LR /= base_lr_scale
        self.trainloader = build_detection_train_loader(self.cloned_cfg)
        self.trainiter = DetectronTrainDataLoaderIter(self.trainloader)
        self.optimizer = build_optimizer(self.cloned_cfg, self.trainer.model)
        self.lr_finder = LRFinder(self.trainer.model, self.optimizer, self._criterion, device=device)
        
    def _criterion(self, outputs, labels):
        """Replaces loss function in LRFinder's format"""
        losses = sum(outputs.values())
        return losses
    
    def range_test(
        self,
        end_lr=1,
        num_iter=100,
    ):
        with EventStorage() as storage:
            self.lr_finder.range_test(
                                    self.trainiter,
                                    val_loader=None,
                                    start_lr=self.cloned_cfg.SOLVER.BASE_LR,
                                    end_lr=end_lr,
                                    num_iter=num_iter,
                                    step_mode="exp",
                                    smooth_f=0.05,
                                    diverge_th=5,
                                    accumulation_steps=1,
                                    non_blocking_transfer=True
                                    )
        self.lr_finder.plot()