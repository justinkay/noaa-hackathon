import os
import glob
import torch

from hackathon.predict import SimpleEvaluator


class BoundingBox:
    """
    Representation of a bounding box and information associated with it.
    
    self.box is represented as [ x1, y1, x2, y2 ]
    """
    
    def __init__(self, box, detect_class, detect_confidence, classification_class=None, classification_confidence=None):
        self.box = box
        self.detect_class = detect_class
        self.detect_confidence = detect_confidence
        self.classification_class = classification_class
        self.classification_confidence = classification_confidence
    
    @classmethod
    def from_instances(cls, instances):
        """
        A factory constructor for any number of BoundingBoxes from an 
        Instances object of any length.
        
        Returns:
            List of BoundingBoxes
            
        Note: this purposely does not index into instances, since this
            occassionally seems to cause an error in Detectron2.
        """
        assert(instances is not None), "Instances is None."
        bb_tensors = [b for b in instances.pred_boxes] # convenient way to grab a boxes as tensors
        boxes = []
        for i in range(len(instances)):
            box = bb_tensors[i].numpy()
            detect_class = instances.pred_classes[i].item()
            detect_confidence = instances.scores[i].item()
            boxes.append(BoundingBox(box, detect_class, detect_confidence))
        return boxes
    
    def get_in_viou_format(self):
        """
        Return a representation of this bounding box in the format required by viou tracking library.
        """
        return {'bbox': tuple(self.box), 'score': torch.tensor(self.detect_confidence), 'class': torch.tensor(self.detect_class)}
    
    @staticmethod
    def get_detections_of_confidence(boxes, confidence):
        return [ bb for bb in boxes if bb.detect_confidence >= confidence ]
    
    def center(self):
        b = self.box
        return ( (b[2] - b[0])/2.0, (b[3] - b[1])/2.0 )
    
class TrackedBox(BoundingBox):
    
    def __init__(self, box, detect_class, detect_confidence, track_id, classification_class=None, classification_confidence=None):
        super().__init__(box, detect_class, detect_confidence, classification_class, classification_confidence)
        self.track_id = track_id
        self.features = None # used for ReID feature caching

class Frame:
    """
    A frame of video for performing object detection, tracking,
    and classification; and for converting between data types for detection
    results.
    
    Stores detections as a list of BoundingBoxes.
    """
    
    def __init__(self, index, filepath, instances=None):
        """
        Args
            instances: Instances (optional)
        """
        self.index = index
        self.filepath = filepath
        self.detections = []
        self.tracked = []
        
        if instances:
            self.load_instances(instances)
        
    def load_instances(self, instances):
        self.detections = BoundingBox.from_instances(instances)
    
    def get_detections_viou_format(self):
        """
        Returns this Frame's detections in the format needed for input to V-IOU
        tracking.
        """
        dets = []
        for bb in self.detections:
            s = bb.detect_confidence
            c = bb.detect_class
            dets.append(bb.get_in_viou_format())
            
        return dets
    
    def get_detections_of_confidence(self, confidence):
        """
        Get all detections in this frame with a confidence score of at least
        confidence.
        """
        return BoundingBox.get_detections_of_confidence(self.detections, confidence)
    
    def clear_tracked(self):
        self.tracked = []
        
    def add_tracked(self, box):
        self.tracked.append(box)
        
    def add_detection(self, box):
        self.detections.append(box)
    
    def clear_detections(self):
        self.detections = []

class Frames:
    """
    A collection of frames in a directory.
    
    Allows batch loading of existing detections or tracking results.
    """
    def __init__(self, frames_dir, load_preds_if_available=True, extension=".jpg"):
        """
        Args:
            frames_dir
            preds: predictions in the format written to disk by SimpleEvaluator.evaluate
        """
        self.dir = frames_dir
        self.num_tracks = 0
        self.extension = extension
        self.load_frames()
        
        default_preds_file = os.path.join(self.dir, "instances_predictions.pth")
        if load_preds_if_available and os.path.exists(default_preds_file):
            self.load_preds_file(default_preds_file)
        
    def __getitem__(self, i):
        return self.frames.__getitem__(i)
    
    def __len__(self):
        return self.frames.__len__()
    
    def load_frames(self):
        self.frames = []
        files = sorted([f for f in glob.glob(self.dir + "/*" + self.extension)])
        for i, f in enumerate(files):
            frame = Frame(i, f)
            self.frames.append(frame)
        
    def load_preds(self, preds):
        """
        Setup self.frames with existing predictions.
        """
        assert(type(preds) is list), "load_preds takes a list of predictions. Did you mean to call load_preds_file?"
        
        for img_pred in preds:
            img_id = img_pred['image_id']
            self.frames[img_id].load_instances(img_pred['instances'])
        
    def load_preds_file(self, filepath):
        print("Loading", filepath)
        self.load_preds(SimpleEvaluator.load_preds(filepath))
        
    def load_preds_df(self, df, class_column="label_l2",
                      exclude_classes=["HUMAN", "NoF"], by_idx=False):
        """Load predictions from a Pandas dataframe.
        Assumes this dataframe has column names like Fishnet.
        Assumes images are already appended with a tracking wildcard.
        
        Args: 
            include_classes: list of classes to track
            by_idx: if images in self.dir are named according to their index in dfl
                    by default, assumes that images are named df.img_id + "." + extension
        """
        for frame in self.frames:
            frame.clear_detections()
            # get image ID by subtracking tracking wildcard at end of filename
            im_id = os.path.split(frame.filepath)[1].split(".")[0]
            if by_idx:
                try:
                    im_id = df.iloc[int(im_id)].img_id
                except:
                    print("Could not find", im_id, frame.filepath)
                    continue
            im_boxes = df[(df.img_id == im_id) & (~df[class_column].isin(exclude_classes))]
            for i, row in im_boxes.iterrows():
                bbox = BoundingBox([float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)], 0, 1.0)
                frame.add_detection(bbox)
        
    def get_detections_viou_format(self):
        """
        Get existing detections in the format required by V-IOU tracking.
        """
        assert(len(self.frames) > 0), "Detection has not yet been run on video:"+self.dir
        return [frame.get_detections_viou_format() for frame in self.frames]
    
    def clear_tracks(self):
        for frame in self.frames:
            frame.clear_tracked()
            