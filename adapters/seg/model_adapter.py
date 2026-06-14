import os
import logging
import cv2
import torch
import numpy as np
import dtlpy as dl
from PIL import Image

logger = logging.getLogger(__name__)

SEGMENTATION_LABELS = {
    0: "background",
    1: "apparel",
    2: "face_neck",
    3: "hair",
    4: "left_foot",
    5: "left_hand",
    6: "left_lower_arm",
    7: "left_lower_leg",
    8: "left_shoe",
    9: "left_sock",
    10: "left_upper_arm",
    11: "left_upper_leg",
    12: "lower_clothing",
    13: "right_foot",
    14: "right_hand",
    15: "right_lower_arm",
    16: "right_lower_leg",
    17: "right_shoe",
    18: "right_sock",
    19: "right_upper_arm",
    20: "right_upper_leg",
    21: "torso",
    22: "upper_clothing",
    23: "lower_lip",
    24: "upper_lip",
    25: "lower_teeth",
    26: "upper_teeth",
    27: "tongue"
}

class SapiensSegmentationAdapter(dl.BaseModelAdapter):
    INPUT_HEIGHT = 1024
    INPUT_WIDTH = 768
    
    # Restored: BGR Mean and Std exactly as the model expects
    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    def load(self, local_path, **kwargs):
        weights_filename = self.model_entity.configuration.get("weights_filename")
        if not weights_filename:
            raise ValueError("weights_filename not found in model configuration")
        
        weights_path = os.path.join(local_path, weights_filename)
        self.input_height = self.model_entity.configuration.get("input_height", self.INPUT_HEIGHT)
        self.input_width = self.model_entity.configuration.get("input_width", self.INPUT_WIDTH)
        
        # Validate input dimensions
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(f"Invalid input dimensions: {self.input_height}x{self.input_width}")
        
        if not os.path.isfile(weights_path):
            if os.path.isfile(local_path):
                weights_path = local_path
            else:
                raise FileNotFoundError(f"No weights found at: {weights_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model.eval()
        logger.info(f"Loaded {weights_path} on {self.device}")

    def prepare_item_func(self, item: dl.Item):
        # Validate item dimensions
        if not item.width or not item.height:
            raise ValueError(f"Item has invalid dimensions: {item.width}x{item.height}")
        
        # Download and decode
        buffer = item.download(save_locally=False)
        img_pil = Image.open(buffer).convert('RGB')
        
        # Restored: Convert to BGR as required by Sapiens' internal preprocessor
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Return stateless dictionary to safely handle batching
        return {
            "image": img_bgr,
            "orig_w": item.width,
            "orig_h": item.height
        }

    def predict(self, batch, **kwargs):
        if not batch:
            return []
        
        batch_annotations = []

        for entry in batch:
            # Validate batch entry
            if not isinstance(entry, dict) or "image" not in entry:
                raise ValueError("Invalid batch entry: missing 'image' key")
            
            image = entry["image"]
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]

            # Validate image
            if image is None or image.size == 0:
                raise ValueError("Invalid image: empty or None")

            # Resize to model input
            resized = cv2.resize(
                image, 
                (self.input_width, self.input_height), 
                interpolation=cv2.INTER_LINEAR
            )

            # Restored: Normalize with BGR mean/std
            img_float = (resized.astype(np.float32) - self.MEAN) / self.STD
            tensor = torch.from_numpy(
                np.ascontiguousarray(img_float.transpose(2, 0, 1))
            ).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                
            if isinstance(output, (list, tuple)):
                output = output[0]

            segmentation = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # Resize segmentation mask back to original item dimensions
            segmentation = cv2.resize(
                segmentation.astype(np.uint8), 
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Build Dataloop annotations
            collection = dl.AnnotationCollection()
            for class_idx, label_name in SEGMENTATION_LABELS.items():
                if class_idx == 0:
                    continue  # Skip background
                
                mask = (segmentation == class_idx).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                
                collection.add(
                    annotation_definition=dl.Segmentation(geo=mask, label=label_name),
                    model_info={"name": self.model_entity.name, "confidence": 1.0}
                )
                
            batch_annotations.append(collection)

        return batch_annotations