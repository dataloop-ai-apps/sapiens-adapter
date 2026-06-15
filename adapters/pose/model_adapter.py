import os
import logging
import cv2
import torch
import numpy as np
import dtlpy as dl
from PIL import Image

logger = logging.getLogger(__name__)

class SapiensPoseAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        weights_filename = self.model_entity.configuration.get("weights_filename")
        if not weights_filename:
            raise ValueError("weights_filename not found in model configuration")
        
        weights_path = os.path.join(local_path, weights_filename)
        
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

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid image: empty or None")
        
        # Return stateless dictionary to safely handle batching
        return {
            "image": img_bgr,
            "orig_w": item.width,
            "orig_h": item.height
        }

    
    def predict(self, batch, **kwargs):
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        input_height = self.model_entity.configuration.get("input_height", 1024)
        input_width = self.model_entity.configuration.get("input_width", 768)

        # From vis_pose.py (COCO format)
        KEYPOINT_NAMES = [
            "nose",
            "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

        batch_annotations = []

        for entry in batch:
            image = entry["image"]
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]

            # --- preprocessing ---
            resized = cv2.resize(
                image,
                (input_width, input_height),
                interpolation=cv2.INTER_LINEAR
            )

            img_float = (resized.astype(np.float32) - mean) / std

            tensor = torch.from_numpy(
                np.ascontiguousarray(img_float.transpose(2, 0, 1))
            ).unsqueeze(0).float().to(self.device)

            # --- inference ---
            with torch.no_grad():
                output = self.model(tensor)

            if isinstance(output, (list, tuple)):
                output = output[0]

            # --- pose extraction ---
            heatmaps = output.squeeze().cpu().numpy()  # [K, H, W]

            keypoints_dict = {}

            for i, name in enumerate(KEYPOINT_NAMES):
                heatmap = heatmaps[i]

                # find max location (same as vis_pose.py idea)
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

                confidence = float(heatmap[y, x])

                # optional threshold (important in real usage)
                if confidence < 0.1:
                    continue

                # scale back to original image
                x = int(x * orig_w / input_width)
                y = int(y * orig_h / input_height)

                keypoints_dict[name] = {
                    "x": x,
                    "y": y,
                    "confidence": confidence
                }

            # --- build Dataloop Pose ---
            collection = dl.AnnotationCollection()

            if len(keypoints_dict) > 0:
                collection.add(
                    annotation_definition=dl.Pose(
                        points=keypoints_dict,
                        label="person",
                        template="Pose_1"
                    ),
                    model_info={
                        "name": self.model_entity.name,
                        "confidence": 1.0
                    }
                )

            batch_annotations.append(collection)

        return batch_annotations