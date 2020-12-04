import torch

from hackathon.data import register_dataset
from hackathon.evaluate import get_model_for_eval, evaluate_coco_dataset


if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Using CPU")
    
cfg = get_model_for_eval("frcnn-bfish", "models/",  device=device)
register_dataset("lehi1", "data/")
evaluate_coco_dataset(cfg, "lehi1")