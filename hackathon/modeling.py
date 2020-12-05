from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class FrozenRCNN(GeneralizedRCNN):
    """Idea inspired by Frustratingly Simple Few-Shot Object Detection, Xin Wang et al.
    Results are poor attempting to transfer COCO/Fishnet features; but maybe for rare bottomfish
    we can try the methods in the paper after training on the full set.
    """

    def __init__(self, *args, **kwargs):
        super(FrozenRCNN, self).__init__(*args, **kwargs)

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.proposal_generator.parameters():
            p.requires_grad = False
