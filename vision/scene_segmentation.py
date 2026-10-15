"""
scene_segmentation.py

Segmentación de escena usando SegFormer (ADE20K)
"""

import torch
import numpy as np
import cv2

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class SceneSegmenter:

    def __init__(self):
        print("[INFO] Cargando modelo de segmentación (ADE20K)...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ).to(self.device)

        self.model.eval()

        # Clases relevantes (ADE20K)
        self.WALL_IDS = [0]
        self.FLOOR_IDS = [3]
        self.DOOR_IDS = [14]

        # superficies transitables (un hoyo = agujero dentro de esto):
        # floor, road, sidewalk, earth, rug, field, sand, path, runway,
        # dirt track, land
        self.WALKABLE_IDS = [3, 6, 11, 13, 28, 29, 46, 52, 54, 91, 94]
        # agua: water, sea, river, pool, waterfall, lake
        self.WATER_IDS = [21, 26, 60, 109, 113, 128]
        # escaleras/escalones: stairs, stairway, escalator, step
        self.STAIRS_IDS = [53, 59, 96, 121]

    def segment(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # interpolación
        seg = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        seg = torch.argmax(seg, dim=1)[0].cpu().numpy()

        return seg

    def extract_navigation_mask(self, seg_map):

        barrier_mask = np.isin(seg_map, self.WALL_IDS + self.DOOR_IDS)
        floor_mask = np.isin(seg_map, self.FLOOR_IDS)

        return barrier_mask.astype(np.uint8), floor_mask.astype(np.uint8)

    def extract_ground_masks(self, seg_map):
        """mascaras pal detector de peligros de piso: transitable,
        agua y escaleras."""
        walkable = np.isin(seg_map, self.WALKABLE_IDS)
        water = np.isin(seg_map, self.WATER_IDS)
        stairs = np.isin(seg_map, self.STAIRS_IDS)
        return (walkable.astype(np.uint8), water.astype(np.uint8),
                stairs.astype(np.uint8))