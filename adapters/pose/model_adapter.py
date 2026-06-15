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
        import time

        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        input_height = self.model_entity.configuration.get("input_height", 1024)
        input_width = self.model_entity.configuration.get("input_width", 768)

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

        TEMPLATE_ID = "b57dc98e-3d94-4c0d-b1a0-6057f06beb0a"

        batch_annotations = []

        for entry in batch:
            image = entry["image"]
            orig_w = entry["orig_w"]
            orig_h = entry["orig_h"]

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

            for i, name in enumerate(KEYPOINT_NAMES):
                heatmap = heatmaps[i]

                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                confidence = float(heatmap[y, x])

                if confidence < 0.001:
                    continue

                # scale to original image
                x = int(x * orig_w / hm_w)
                y = int(y * orig_h / hm_h)


                # ✅ clamp to image bounds
                x = max(0, min(orig_w - 1, x))
                y = max(0, min(orig_h - 1, y))


                keypoints_dict[name] = (x, y)

            # --- build annotations ---
            collection = dl.AnnotationCollection()

            if len(keypoints_dict) > 0:
                # ✅ create stable parent id
                parent_id = str(int(time.time() * 1000000))

                # ✅ parent pose (NO points inside!)
                collection.add(
                    annotation_definition=dl.Pose(
                        label="person",
                        template_id=TEMPLATE_ID
                    ),
                    object_id=parent_id,
                    model_info={
                        "name": self.model_entity.name,
                        "confidence": 1.0
                    }
                )

                # ✅ children keypoints
                for name, (x, y) in keypoints_dict.items():
                    collection.add(
                        annotation_definition=dl.Point(
                            x=x,
                            y=y,
                            label=name
                        ),
                        parent_id=parent_id
                    )

            # ✅ debug (optional but useful)
            logger.info(f"Generated {len(collection.annotations)} annotations")

            batch_annotations.append(collection)

        return batch_annotations


if __name__ == "__main__":
    dl.setenv('rc')
    # dl.login()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    project = dl.projects.get(project_name="menachem-onboarding")
    model = project.models.get(model_name="sapiens-pose-adapter-model-0.3b")

    adapter = SapiensPoseAdapter(model_entity=model)

    dataset = project.datasets.get(dataset_name="new-dataset")

    for filepath in ["/sample-person_2.jpg"]:
        try:
            item = dataset.items.get(filepath=filepath)
            logger.info(f"\n{'=' * 60}\nTesting: {filepath}\n{'=' * 60}")
            adapter.predict_items([item], upload=True)
        except Exception as e:
            logger.error(f"Failed on {filepath}: {e}")

    print("Done.")