"""Real-world FoodSeg103 (103-class) Mask2Former PyTorch inference engine."""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

# Prevent broken local torchvision binary mismatch before importing transformers
if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessorPil,
)

from food_taxonomy import (
    FOOD101_THAI_NAMES,
    FOODSEG103_ID2LABEL,
    FOODSEG103_THAI_NAMES,
    FOODSEG103_TO_LOCAL_10CLASS,
    THFOOD50_THAI_NAMES,
    _THFOOD_SPICY_CHILI_DISHES,
    _UNSHIFT_LUT,
    finalize_local_mask,
    fuse_multimodel_context,
)


MODEL_REPO = "mericberktas/foodseg-models"
MODEL_SUBFOLDER = "foodseg103/mask2former"
THFOOD50_REPO = "thean/swinv2-tiny-patch4-window8-256-finetuned-THFOOD-50"
FOOD101_REPO = "skylord/swin-finetuned-food101"


def _refine_red_chili_slices(
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    thai_dish_info: Dict[str, object],
) -> None:
    """Refine small red chili slices only within spicy Thai dishes."""
    dish_code = str(thai_dish_info.get("top1_code", ""))
    conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    if dish_code not in _THFOOD_SPICY_CHILI_DISHES or conf < 60.0:
        return

    candidate_region = np.isin(raw_mask, [46, 47, 48, 50, 52, 54, 89])
    if not np.any(candidate_region):
        return

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chili_candidates = (
        candidate_region & ((hue <= 7) | (hue >= 170)) & (sat >= 145) & (val >= 65)
    )
    if int(np.count_nonzero(chili_candidates)) < 15:
        return

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        chili_candidates.astype(np.uint8)
    )
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if 15 <= area <= 800:
            raw_mask[labels == comp_id] = 94


class RealFoodSeg103Engine:
    """Tri-AI engine: SwinV2-Tiny (THFOOD-50) + Swin-Base (Food-101) + Mask2Former (FoodSeg103)."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Mask2FormerImageProcessorPil(
            do_resize=True,
            size={"shortest_edge": 512, "longest_edge": 1024},
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_REPO,
            subfolder=MODEL_SUBFOLDER,
        ).to(self.device)
        self.model.eval()

        self.thai_processor = AutoImageProcessor.from_pretrained(THFOOD50_REPO)
        self.thai_model = AutoModelForImageClassification.from_pretrained(THFOOD50_REPO).to(
            self.device
        )
        self.thai_model.eval()

        self.food101_processor = AutoImageProcessor.from_pretrained(FOOD101_REPO)
        self.food101_model = AutoModelForImageClassification.from_pretrained(FOOD101_REPO).to(
            self.device
        )
        self.food101_model.eval()

    @staticmethod
    @torch.no_grad()
    def _run_classifier_topk(
        pil_img: Image.Image,
        processor: AutoImageProcessor,
        model: AutoModelForImageClassification,
        name_map: Dict[str, str],
        device: torch.device,
        k: int = 3,
    ) -> Dict[str, object]:
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_vals, top_idxs = torch.topk(probs, k=min(k, probs.shape[0]))

        top3: List[Dict[str, object]] = []
        id2label = model.config.id2label
        for val, idx in zip(top_vals, top_idxs):
            code = str(id2label[int(idx)])
            top3.append({
                "code": code,
                "thai_name": name_map.get(code, code.replace("_", " ")),
                "conf_pct": round(float(val) * 100.0, 1),
            })

        best = top3[0]
        return {
            "top1_code": best["code"],
            "top1_thai": best["thai_name"],
            "top1_conf_pct": best["conf_pct"],
            "top3": top3,
        }

    @torch.no_grad()
    def classify_thai_dish(self, pil_img: Image.Image) -> Dict[str, object]:
        """Identify the Thai dish class (top-1 and top-3) using SwinV2-Tiny (THFOOD-50)."""
        return self._run_classifier_topk(
            pil_img, self.thai_processor, self.thai_model, THFOOD50_THAI_NAMES, self.device
        )

    @torch.no_grad()
    def classify_food101_dish(self, pil_img: Image.Image) -> Dict[str, object]:
        """Identify global/Asian dish class (top-1 and top-3) using Swin-Base (Food-101)."""
        return self._run_classifier_topk(
            pil_img, self.food101_processor, self.food101_model, FOOD101_THAI_NAMES, self.device
        )

    @torch.no_grad()
    def predict(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, List[Dict[str, object]], Dict[str, object]]:
        """Run Tri-AI inference and return 5-group nutrition masks and dish context."""
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = self.model(pixel_values=pixel_values)

        class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]
        mask_probs = outputs.masks_queries_logits.sigmoid()
        sem_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        sem_probs = F.interpolate(sem_probs, size=(h, w), mode="bilinear", align_corners=False)

        top2_vals, top2_preds = torch.topk(sem_probs, k=2, dim=1)
        thai_dish_info = self.classify_thai_dish(pil_img)
        food101_info = self.classify_food101_dish(pil_img)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_mask = _UNSHIFT_LUT[top2_preds[0, 0].cpu().numpy().astype(np.uint8)]
        raw_top2_mask = _UNSHIFT_LUT[top2_preds[0, 1].cpu().numpy().astype(np.uint8)]
        conf_map = top2_vals[0, 0].cpu().numpy()

        thai_dish_info = fuse_multimodel_context(
            raw_mask, raw_top2_mask, thai_dish_info, food101_info
        )
        _refine_red_chili_slices(image_rgb, raw_mask, thai_dish_info)
        local_mask, mean_conf, detailed_ingredients = finalize_local_mask(
            raw_mask, conf_map, thai_dish_info
        )
        return local_mask, raw_mask, mean_conf, latency_ms, detailed_ingredients, thai_dish_info
