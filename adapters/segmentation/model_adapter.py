import os
import logging
import cv2
import torch
import numpy as np
import dtlpy as dl
from PIL import Image

logger = logging.getLogger(__name__)

class SapiensSegmentationAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        weights_filename = self.model_entity.configuration.get("weights_filename")
        if not weights_filename:
            raise ValueError("weights_filename not found in model configuration")
        
        weights_path = os.path.join(local_path, weights_filename)
        
        if not os.path.isfile(weights_path):
            image_weights = os.path.join('/tmp/app/weights', weights_filename)
            if os.path.isfile(image_weights):
                weights_path = image_weights
            else:
                raise FileNotFoundError(f"No weights found at: {weights_path} or {image_weights}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model.eval()
        logger.info(f"Loaded {weights_path} on {self.device}")

    def prepare_item_func(self, item: dl.Item):
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        input_height = self.model_entity.configuration.get("input_height", 1024)
        input_width = self.model_entity.configuration.get("input_width", 768)
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

        resized = cv2.resize(
                img_bgr, 
                (input_width, input_height), 
                interpolation=cv2.INTER_LINEAR
            )

        img_float = (resized.astype(np.float32) - mean) / std
        tensor = torch.from_numpy(
            np.ascontiguousarray(img_float.transpose(2, 0, 1))
        ).unsqueeze(0).float().to(self.device)
        
        # Return stateless dictionary to safely handle batching
        return {
            "orig_w": item.width,
            "orig_h": item.height,
            "tensor": tensor
        }

    def predict(self, batch, **kwargs):
        input_height = self.model_entity.configuration.get("input_height", 1024)
        input_width = self.model_entity.configuration.get("input_width", 768)
        labels = self.model_entity.labels
        
        batch_annotations = []

        if not labels:
            raise ValueError("Model has no labels configured")

        # Validate input dimensions
        if input_height <= 0 or input_width <= 0:
            raise ValueError(f"Invalid input dimensions: {input_height}x{input_width}")

        for entry in batch:
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]
            tensor = entry["tensor"]

            with torch.no_grad():
                output = self.model(tensor)
                
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Convert logits → probabilities
            probs = torch.softmax(output, dim=1)

            # Get prediction and confidence per pixel
            conf_map, segmentation = torch.max(probs, dim=1)

            segmentation = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            conf_map = conf_map.squeeze().cpu().numpy()

            # Resize segmentation mask back to original item dimensions
            segmentation = cv2.resize(
                segmentation.astype(np.uint8), 
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            conf_map = cv2.resize(
                conf_map.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )


            # Build DDOE annotations
            collection = dl.AnnotationCollection()
            for class_idx, label_name in enumerate(labels):
                if class_idx == 0:
                    continue  # Skip background
                
                mask = (segmentation == class_idx)
                
                if mask.sum() < 100:   # remove tiny noise blobs
                    continue

                threshold = 0.5
                valid_mask = mask & (conf_map > threshold)
                
                if valid_mask.sum() == 0:
                    continue

                confidence = float(conf_map[valid_mask].mean())

                collection.add(
                    annotation_definition=dl.Segmentation(geo=valid_mask, label=label_name),
                    model_info={"name": self.model_entity.name, "confidence": confidence}
                )
                
            batch_annotations.append(collection)

        return batch_annotations

dl.setenv('rc')
project_name = "menachem-onboarding"
model_name = "sapiens-segmentation-adapter-model-0.6b"
dataset_name = "new-dataset"
filepath = "/standing-m-2.jpg"

project = dl.projects.get(project_name=project_name)
model = project.models.get(model_name=model_name)
adapter = SapiensSegmentationAdapter(model_entity=model)
dataset = project.datasets.get(dataset_name=dataset_name)
item = dataset.items.get(filepath=filepath)
adapter.predict_items([item])
item.open_in_web()