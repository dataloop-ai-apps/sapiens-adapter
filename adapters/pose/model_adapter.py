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
            "orig_h": item.height,
            "item": item
        }

    
    def predict(self, batch, **kwargs):
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

        input_height = self.model_entity.configuration.get("input_height", 1024)
        input_width = self.model_entity.configuration.get("input_width", 768)

        recipe_id = self.model_entity.configuration.get("recipe_id")
        template_name = self.model_entity.configuration.get("template_name")
        keypoints = self.model_entity.labels or []

        template_id = None

        # ✅ Resolve template if exists
        if recipe_id and template_name:
            try:
                dataset = batch[0]["item"].dataset
                recipe = dataset.recipes.get(recipe_id)
                template_id = recipe.get_annotation_template_id(template_name=template_name)
                logger.info(f"Using template_id: {template_id}")
            except Exception as e:
                logger.warning(f"Template not found. Falling back to points-only. Error: {e}")
                template_id = None

        batch_annotations = []

        for entry in batch:
            image = entry["image"]
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]
            item = entry["item"]

            # --- preprocessing ---
            resized = cv2.resize(image, (input_width, input_height))
            img_float = (resized.astype(np.float32) - mean) / std

            tensor = torch.from_numpy(
                np.ascontiguousarray(img_float.transpose(2, 0, 1))
            ).unsqueeze(0).float().to(self.device)

            # --- inference ---
            with torch.no_grad():
                output = self.model(tensor)

            if isinstance(output, (list, tuple)):
                output = output[0]

            heatmaps = output.squeeze().cpu().numpy()

            keypoints_dict = {}
            hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

            for i, name in enumerate(keypoints):
                heatmap = heatmaps[i]

                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                confidence_raw = float(heatmap[y, x])
                confidence = 1.0 / (1.0 + np.exp(-confidence_raw))

                if confidence < 0.001:
                    continue

                # ✅ correct scaling
                x = int(x * orig_w / hm_w)
                y = int(y * orig_h / hm_h)

                x = max(0, min(orig_w - 1, x))
                y = max(0, min(orig_h - 1, y))

                keypoints_dict[name] = (x, y, confidence)

            # --- build annotations ---
            collection = dl.AnnotationCollection()

            if len(keypoints_dict) > 0:

                # ✅ CASE 1: Template exists → Pose + child Points
                if template_id is not None:

                    parent_annotation = item.annotations.upload(
                        dl.Annotation.new(annotation_definition=dl.Pose(
                            label='my_parent_label',
                            template_id=template_id,
                            instance_id=None  # Optional for tracking specific instances
                        ))
                    )[0]
                    builder = item.annotations.builder()

                    for name, (x, y, confidence) in keypoints_dict.items():
                        builder.add(annotation_definition=dl.Point(x=x,
                                                                    y=y,
                                                                    label=name),
                                    parent_id=parent_annotation.id)
                    batch_annotations.append(builder.annotations)

                # ✅ CASE 2: No template → simple Pose(points=...)
                else:
                    for name, (x, y, confidence) in keypoints_dict.items():
                        collection.add(
                            annotation_definition=dl.Point(
                                x=x,
                                y=y,
                                label=name
                            ),
                            model_info={
                                "name": self.model_entity.name,
                                "confidence": confidence
                            }
                        )

            logger.info(f"Generated {len(collection.annotations)} annotations")

            batch_annotations.append(collection)

        return batch_annotations