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

        # Superficies TRANSITABLES en interiores y exteriores (para el
        # detector de peligros de piso: hoyos/coladeras = agujeros dentro
        # de la superficie transitable):
        # 3 floor, 6 road, 11 sidewalk, 13 earth, 28 rug, 29 field,
        # 46 sand, 52 path, 54 runway, 91 dirt track, 94 land
        self.WALKABLE_IDS = [3, 6, 11, 13, 28, 29, 46, 52, 54, 91, 94]
        # Agua (peligro directo): 21 water, 26 sea, 60 river,
        # 109 swimming pool, 113 waterfall, 128 lake
        self.WATER_IDS = [21, 26, 60, 109, 113, 128]
        # Escaleras / escalones (riesgo alto según la tabla de riesgo del
        # proyecto): 53 stairs, 59 stairway, 96 escalator, 121 step
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
        """Máscaras para el detector de peligros de piso (interior y
        exterior): superficie transitable, agua y escaleras/escalones."""
        walkable = np.isin(seg_map, self.WALKABLE_IDS)
        water = np.isin(seg_map, self.WATER_IDS)
        stairs = np.isin(seg_map, self.STAIRS_IDS)
        return (walkable.astype(np.uint8), water.astype(np.uint8),
                stairs.astype(np.uint8))