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

        batch_annotations = []

        for entry in batch:
            image = entry["image"]
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]

            # Resize
            resized = cv2.resize(
                image,
                (input_width, input_height),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalize
            img_float = (resized.astype(np.float32) - mean) / std
            tensor = torch.from_numpy(
                np.ascontiguousarray(img_float.transpose(2, 0, 1))
            ).unsqueeze(0).float().to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(tensor)

            if isinstance(output, (list, tuple)):
                output = output[0]

            # ---- POSE LOGIC STARTS HERE ----

            heatmaps = output.squeeze().cpu().numpy()  # [K, H, W]
            num_keypoints = heatmaps.shape[0]

            keypoints = []

            for i in range(num_keypoints):
                heatmap = heatmaps[i]

                # Find peak (argmax over H,W)
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

                # Scale back to original image size
                x = int(x * orig_w / input_width)
                y = int(y * orig_h / input_height)

                keypoints.append({"x": x, "y": y, "id": i})

            # ---- BUILD DATALOOP ANNOTATION ----

            collection = dl.AnnotationCollection()

            if len(keypoints) > 0:
                collection.add(
                    annotation_definition=dl.Points(
                        points=[(kp["x"], kp["y"]) for kp in keypoints],
                        label="person"  # or your configured label
                    ),
                    model_info={"name": self.model_entity.name, "confidence": 1.0}
                )

            batch_annotations.append(collection)

        return batch_annotations