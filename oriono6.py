# Code aided by Gemini 3 Pro

import logging
import numpy as np
import sys
import os
from typing import Literal
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

# --- PATH SETUP ---
sys.path.append('/opt/radxa_venv')
sys.path.append('/opt/radxa_ai_hub')

try:
    from utils.image_process import preprocess_object_detect_method2
    from utils.object_detect_postprocess import postprocess_yolox
    from utils.NOE_Engine import EngineInfer
except ImportError:
    raise RuntimeError("Radxa SDK missing.")

logger = logging.getLogger(__name__)

DETECTOR_KEY = "oriono6"

class OrionO6DetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]

class OrionO6(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, config: OrionO6DetectorConfig):
        super().__init__(config)
        self.w = 640
        self.h = 640

        logger.info(f"--- ORION YOLOX (OPTIMIZED) --- Ready.")

        model_path = config.model.path
        if not os.path.exists(model_path):
             model_path = "/config/model_cache/oriono6/yolox_m.cix"

        self.engine = EngineInfer(model_path)

    def detect_raw(self, tensor_input):
        if tensor_input.ndim == 4:
            image = tensor_input[0]
        else:
            image = tensor_input

        # 1. Official Preprocessing
        # Handles BGR conversion and resizing automatically
        src_shape, new_shape, show_image, data = preprocess_object_detect_method2(
            image, target_size=(self.h, self.w), mode="BGR"
        )

        try:
            # 2. Hardware Inference
            output = self.engine.forward(data)[0]
            output = np.reshape(output, (1, 8400, 85))

            # 3. Post-Process (Returns XYWH: CenterX, CenterY, Width, Height)
            # Thresholds: 0.5 Conf, 0.45 IoU
            results = postprocess_yolox(output, (self.h, self.w), 0.5, 0.45)[0]

        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return np.zeros((20, 6), np.float32)

        # 4. Vectorized Output Formatting
        detections = np.zeros((20, 6), np.float32)
        count = min(len(results), 20)

        if count > 0:
            # Slice only the valid results
            r = results[:count]

            # Extract columns (Vectorized)
            cx = r[:, 0]
            cy = r[:, 1]
            w  = r[:, 2]
            h  = r[:, 3]
            score = r[:, 4]
            cls = r[:, 5]

            # Math: Convert XYWH (Center) -> XYXY (TopLeft/BottomRight)
            # Normalized by self.w (640) and self.h (640)
            half_w = w / 2.0
            half_h = h / 2.0

            detections[:count, 0] = cls
            detections[:count, 1] = score
            detections[:count, 2] = (cy - half_h) / self.h # ymin
            detections[:count, 3] = (cx - half_w) / self.w # xmin
            detections[:count, 4] = (cy + half_h) / self.h # ymax
            detections[:count, 5] = (cx + half_w) / self.w # xmax

        return detections