import os
import cv2
from tqdm import tqdm
import torch
import argparse

from detectron2.data.catalog import Metadata
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer, _DetectedInstance
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)

from hackathon.structures import Frames

class AiFishVisualizer(object):
    
    def __init__(self, frames, output_fname, height, width, fps):
        self.frames = frames
        self.output_fname = output_fname + ".avi"
        self.height = height
        self.width = width
        self.fps = fps
        
    def run(self, tracking=False):
        print("Running on video with width", self.width, ", height", self.height, ", fps", self.fps)
        
        metadata = Metadata()
        if tracking:
            metadata.thing_classes = [ i for i in range(self.frames.num_tracks) ]
        else:
            metadata.thing_classes = ['FISH']
        
        if tracking:
            self.video_visualizer = TrackingVisualizer(metadata, ColorMode.IMAGE)
        else:
            self.video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)

        preds = self._get_detectron_preds(tracking)
        output_file = cv2.VideoWriter(
            filename=self.output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
#             fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
            fps=float(self.fps),
            frameSize=(self.width, self.height),
            isColor=True,
        )

        for vis_frame in tqdm(self._frame_gen(preds), total=len(preds)):
            output_file.write(vis_frame)

        output_file.release()
        
    def _get_detectron_preds(self, tracking=False):
        """
        Convert self.frames to the format required by Detectron2's visualizer.
        
        Args:
            tracking: use tracking results instead of raw detection results.
            
        Return:
            { frame_ind: { 'instances': Instances(...) }, ... }
        """
        preds =  { i : {} for i in range(len(self.frames)) }
        
        for i, frame in enumerate(self.frames):
            if tracking:
                detections = frame.tracked
                classes = torch.tensor([ bbox.track_id for bbox in detections ])
            else:
                detections = frame.detections
                classes = torch.tensor([ bbox.detect_class for bbox in detections ])
                
            boxes = Boxes(torch.tensor([ bbox.box for bbox in detections ]))
            scores = torch.tensor([ bbox.detect_confidence for bbox in detections ])
            
            instances = Instances((self.height, self.width), pred_boxes=boxes, scores=scores, pred_classes=classes)
            preds[i]["instances"] = instances
            
        return preds
        
    def _frame_gen(self, preds):
        """
        Load the frame image from disk and draw predictions onto it.
        
        Args:
            preds: all Detectron2-formatted predictions for this video.
            
        Yield:
            cv2 annotated frame.
        """
        for i, frame in enumerate(self.frames):
            yield self._process_predictions(cv2.imread(frame.filepath), preds[i])
        
    def _process_predictions(self, frame, predictions):
        """
        Draw predictions onto a frame.
        
        Args:
            frame: cv2 image
            predictions: Detectron2-formatted predictions for this frame.
        
        Return:
            cv2 annotated frame.
        """
        pred_key = "instances"
        if pred_key in predictions:
            predictions = predictions[pred_key].to("cpu")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis_frame = self.video_visualizer.draw_instance_predictions(frame, predictions)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame
        else:
            return frame
    
class TrackingVisualizer(VideoVisualizer):
    """
    Modified VideoVisualizer for tracking. Predictions are expected to have
    class=track_id, and metadata.thing_classes should contain all these "classes".
    
    Basically this class just makes sure each track always maintains its color.
    """
    
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        super().__init__(metadata, instance_mode)
        self._colors = [ random_color(rgb=True, maximum=1) for _ in metadata.thing_classes ]
    
    def draw_instance_predictions(self, frame, predictions):
        """
        Except where noted, this is copied straight from VideoVisualizer.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
        else:
            masks = None

        ######################
        ## CHANGES          ##
        ######################
        
        detected = [
            _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=self._colors[int(classes[i])], ttl=8)
            for i in range(num_instances)
        ]
        colors = [ d.color for d in detected ]
        
        ######################
        ## END OF  CHANGES  ##
        ######################
        
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        
        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            alpha = 0.3
        else:
            alpha = 0.5

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=masks,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output
    
    def _assign_colors(self, instances):
        """
        Simple color assignment based on "class" (track ID).
        """
        
        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.5
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]
    
def eval_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="", 
                        help="location of frames dir. should contain instances_predictions.pth")
    parser.add_argument("--out", default="", help="path to output file (without extension)")
    parser.add_argument("--fps", default=10, help="frames per second")
    return parser
    
def main(args):
    frames = Frames(args.frames, extension=".png")
    sample_frame = frames[0].filepath
    im = cv2.imread(sample_frame)
    h, w, _ = im.shape
    vis = AiFishVisualizer(frames, args.out, height=h, width=w, fps=args.fps)
    vis.run(tracking=False)
    
if __name__ == "__main__":
    args = eval_argument_parser().parse_args()
    main(args)