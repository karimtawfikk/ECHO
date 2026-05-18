from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import DBSCAN
from torchvision.ops import nms as torch_nms
from ultralytics import YOLO

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HieroglyphPipelineConfig:
    """Centralised configuration for the Hieroglyph Detection Pipeline."""

    # Preprocessing (Stable Stage 2 Logic)
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    max_image_dim: int = 4096

    # Detection (YOLOv8/11 Tiled)
    detection_model_path = "src/ml_models/hieroglyph_models/detection/model.pt"
    
    # Confidence & Refinement
    classification_confidence_threshold: float = 0.15  # Lowered to reduce 'Unknown' count
    row_clustering_threshold: float = 40.0
    duplicate_iou_threshold: float = 0.45
    min_symbol_size_px: int = 15
    detection_confidence: float = 0.30
    detection_iou_threshold: float = 0.45
    detection_img_size: int = 640
    detection_tile_size: int = 640
    inference_batch_size: int = 16

    # Classification (EfficientNetV2B0 — matches ClassificationFinalNotebook)
    classification_model_path = "src/ml_models/hieroglyph_models/classification/model.weights.h5"
    
    classification_label_map_path = "src/ml_models/hieroglyph_models/classification/label_map.json"
    classification_img_size: int = 224

    # Translation (M2M-100 Fine-tuned)
    translation_model_path = "ECHO-EG/echo-m2m100-hieroglyph"

    translation_max_input_length: int = 512
    translation_max_target_length: int = 128
    translation_num_beams: int = 4

    # Clustering (DBSCAN Row Logic)
    # FIX 1: dbscan_eps_factor removed — now using two-pass DBSCAN with fixed factors
    dbscan_tight_eps_factor: float = 0.3   # Pass 1: isolate stacked groups
    dbscan_loose_eps_factor: float = 1.5   # Pass 2: group into rows/columns
    dbscan_min_samples: int = 1
    quadrant_x_overlap_threshold: float = 0.50


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class HieroglyphDetectionRuntime:
    """
    Full Hieroglyph pipeline runtime: Detect → Classify → Translate.
    Follows the clean architecture pattern of Chatbot and Video APIs.
    """

    def __init__(self, config: HieroglyphPipelineConfig | None = None) -> None:
        self.config = config or HieroglyphPipelineConfig()
        self.repo_root = Path(__file__).resolve().parents[2]
        self._detection_model: Any | None = None
        self._classification_model: Any | None = None
        self._classification_label_map: dict[int, str] | None = None
        self._translation_model: Any | None = None
        self._translation_tokenizer: Any | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self.repo_root / path

    # -------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------

    def load_detection_model(self) -> Any:
        """Load and cache the YOLO detection model."""
        if self._detection_model is not None:
            return self._detection_model

        model_path = self._resolve_path(self.config.detection_model_path)
        if not model_path.exists():
            print(f"[hieroglyph] ERROR: Detection model not found at {model_path}", flush=True)
            return None

        print(f"[hieroglyph] Loading detection model: {model_path.name}", flush=True)
        load_start = time.time()
        self._detection_model = YOLO(str(model_path))
        self._detection_model.to(self._device)
        print(f"[hieroglyph] Detection model ready in {time.time() - load_start:.2f}s", flush=True)
        return self._detection_model

    def load_classification_model(self) -> Any:
        """Load and cache the Keras classification model + label map."""
        if self._classification_model is not None:
            return self._classification_model

        model_path = self._resolve_path(self.config.classification_model_path)
        label_map_path = self._resolve_path(self.config.classification_label_map_path)

        if not model_path.exists():
            print(f"[hieroglyph] WARNING: Classification model not found at {model_path}", flush=True)
            return None

        print(f"[hieroglyph] Loading classification model: {model_path.name}", flush=True)
        load_start = time.time()

        # Suppress TF logs but keep error info
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        # CRITICAL: Allow memory growth so TF doesn't steal all VRAM from Torch
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[hieroglyph] TF Memory Growth enabled for: {gpus}", flush=True)
            except RuntimeError as e:
                print(f"[hieroglyph] TF Memory Growth Error: {e}", flush=True)
        tf.get_logger().setLevel("ERROR")
        
        device_used = "GPU" if gpus else "CPU"
        print(f"[hieroglyph] Classification model will use: {device_used}", flush=True)

        # Keras 3 native save format
        config_path = model_path.parent / "config.json"
        expected_weights = model_path.parent / "model.weights.h5"

        if config_path.exists() and not expected_weights.exists():
            import shutil
            try:
                shutil.copy2(str(model_path), str(expected_weights))
                print(f"[hieroglyph] Linked {model_path.name} → model.weights.h5", flush=True)
            except Exception as e:
                print(f"[hieroglyph] Linking warning: {e}", flush=True)

        print(f"[hieroglyph] Attempting to load model from: {model_path}", flush=True)
        if config_path.exists():
            self._classification_model = tf.keras.models.load_model(str(model_path.parent))
            print(f"[hieroglyph] Loaded model from directory: {model_path.parent}", flush=True)
        else:
            self._classification_model = tf.keras.models.load_model(str(model_path))
            print(f"[hieroglyph] Loaded model from file: {model_path}", flush=True)

        # Warm-up prediction to avoid first-hit latency
        try:
            print("[hieroglyph] Warming up classification model...", flush=True)
            img_size = self.config.classification_img_size
            dummy_input = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
            self._classification_model.predict(dummy_input, verbose=0)
            print("[hieroglyph] Classification model warmed up", flush=True)
        except Exception as e:
            print(f"[hieroglyph] Warm-up warning: {e}", flush=True)

        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            self._classification_label_map = {int(k): v for k, v in raw_map.items()}
            print(f"[hieroglyph] Classification label map loaded: {len(self._classification_label_map)} classes", flush=True)
        else:
            print(f"[hieroglyph] WARNING: Label map not found at {label_map_path}", flush=True)
            self._classification_label_map = {}

        print(f"[hieroglyph] Classification model ready in {time.time() - load_start:.2f}s", flush=True)
        return self._classification_model

    def load_translation_model(self) -> tuple[Any, Any] | None:
        """Load and cache the M2M-100 translation model + tokenizer."""
        if self._translation_model is not None:
            return self._translation_model, self._translation_tokenizer

        model_id = self.config.translation_model_path
        model_path = self._resolve_path(model_id)
        
        # Use local path if it exists, otherwise assume it's a Hugging Face repo ID
        if model_path.exists():
            load_path = str(model_path)
        else:
            load_path = model_id

        # Returning to GPU with FP16 optimization to prevent timeouts
        # However, this model is 3.5GB. If it fails, we MUST fallback to CPU.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[hieroglyph] Translation model will use: {self._device} (Optimized) from {load_path}", flush=True)

        load_start = time.time()
        try:
            from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
            
            self._translation_tokenizer = M2M100Tokenizer.from_pretrained(load_path)
            
            # Clear cache before loading big model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load in FP16 if using CUDA to be faster and use less memory
            if self._device == "cuda":
                try:
                    self._translation_model = M2M100ForConditionalGeneration.from_pretrained(
                        load_path, 
                        torch_dtype=torch.float16
                    ).to(self._device)
                except Exception as cuda_err:
                    print(f"[hieroglyph] CUDA OOM or Error: {cuda_err}. Falling back to CPU for translation...", flush=True)
                    self._device = "cpu"
                    self._translation_model = M2M100ForConditionalGeneration.from_pretrained(load_path).to("cpu")
            else:
                self._translation_model = M2M100ForConditionalGeneration.from_pretrained(load_path).to("cpu")
            
            self._translation_model.eval()

            # Match notebook: src_lang and tgt_lang both set to "en"
            self._translation_tokenizer.src_lang = "en"
            self._translation_tokenizer.tgt_lang = "en"

            print(f"[hieroglyph] Translation model ready in {time.time() - load_start:.2f}s on {self._device}", flush=True)
        except Exception as e:
            print(f"[hieroglyph] CRITICAL ERROR loading translation model: {e}", flush=True)
            self._translation_model = None
            self._translation_tokenizer = None
            
        if self._translation_model is None or self._translation_tokenizer is None:
            return None
            
        return self._translation_model, self._translation_tokenizer

    def ensure_models_loaded(self) -> None:
        """Startup warm-up — load all models."""
        self.load_detection_model()
        self.load_classification_model()
        self.load_translation_model()

    # -------------------------------------------------------------------
    # Stage 0: Decode
    # -------------------------------------------------------------------

    def decode_image(self, image_b64: str) -> np.ndarray:
        """Convert base64 string to BGR image."""
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------------------
    # Stage 1: Preprocessing
    # -------------------------------------------------------------------

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Exact match for apply_pro_enhancement from FinalDetectionIOU80.py.
        CLAHE in LAB space (clipLimit=3.0, tileGrid=8×8) + Laplacian sharpening.
        """
        if image_bgr is None: return None
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(enhanced, -1, kernel)

    # -------------------------------------------------------------------
    # Stage 2: Detection (YOLO Tiled)
    # -------------------------------------------------------------------

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Tiled YOLO inference with Stage 2 NMS."""
        h, w = image_bgr.shape[:2]
        tile = self.config.detection_tile_size
        model = self.load_detection_model()
        if model is None: return []

        # Tiling logic
        step = int(tile * 0.5)
        y_offsets = list(range(0, max(1, h - tile + 1), step))
        if h > tile: y_offsets.append(h - tile)
        x_offsets = list(range(0, max(1, w - tile + 1), step))
        if w > tile: x_offsets.append(w - tile)

        all_raw = []
        tiles, offsets = [], []
        for oy in y_offsets:
            for ox in x_offsets:
                tiles.append(image_bgr[oy:oy+tile, ox:ox+tile])
                offsets.append((ox, oy))

        # Batch inference
        bs = self.config.inference_batch_size
        for i in range(0, len(tiles), bs):
            batch_tiles = tiles[i:i+bs]
            batch_offs = offsets[i:i+bs]
            results = model.predict(source=batch_tiles, conf=self.config.detection_confidence, imgsz=self.config.detection_img_size, verbose=False)

            for r, (ox, oy) in zip(results, batch_offs):
                for j in range(len(r.boxes)):
                    box = r.boxes.xyxy[j].cpu().numpy()
                    conf = float(r.boxes.conf[j].cpu().numpy())
                    all_raw.append([box[0]+ox, box[1]+oy, box[2]+ox, box[3]+oy, conf])

        if not all_raw: return []

        # NMS
        boxes_t = torch.tensor([d[:4] for d in all_raw], dtype=torch.float32)
        scores_t = torch.tensor([d[4] for d in all_raw], dtype=torch.float32)
        keep = torch_nms(boxes_t, scores_t, iou_threshold=self.config.detection_iou_threshold)

        return [{
            "bbox": all_raw[int(i)][:4],
            "confidence": all_raw[int(i)][4],
            "centre_x": (all_raw[int(i)][0] + all_raw[int(i)][2]) / 2,
            "centre_y": (all_raw[int(i)][1] + all_raw[int(i)][3]) / 2
        } for i in keep if all_raw[int(i)][4] >= self.config.detection_confidence]

    # -------------------------------------------------------------------
    # Stage 3: Post-Detection Refinement
    # -------------------------------------------------------------------

    def remove_duplicates(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Global suppression for overlapping tiles."""
        if not detections: return []
        sorted_dets = sorted(detections, key=lambda x: -x["confidence"])
        keep = []
        for d in sorted_dets:
            should_keep = True
            for k in keep:
                b1, b2 = d["bbox"], k["bbox"]
                ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                inter = ix * iy
                union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
                if (inter / union if union > 0 else 0) > self.config.duplicate_iou_threshold:
                    should_keep = False
                    break
            if should_keep: keep.append(d)
        return keep

    def deduplicate_by_center(self, detections: list[dict[str, Any]], dist_thr: float = 25.0) -> list[dict[str, Any]]:
        """Merge boxes with nearly identical centers."""
        if not detections: return []
        sorted_dets = sorted(detections, key=lambda x: -x["confidence"])
        keep = []
        for d in sorted_dets:
            cx, cy = d["centre_x"], d["centre_y"]
            is_duplicate = False
            for k in keep:
                dx, dy = cx - k["centre_x"], cy - k["centre_y"]
                if (dx*dx + dy*dy)**0.5 < dist_thr:
                    is_duplicate = True
                    break
            if not is_duplicate: keep.append(d)
        return keep

    # -------------------------------------------------------------------
    # Stage 4: Clustering & Ordering
    # -------------------------------------------------------------------
    def cluster(self, detections: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        """
        Two-Pass DBSCAN Clustering.

        FIX 3: Two-pass DBSCAN replaces the single-pass approach.
        Pass 1 (tight eps): Isolates stacked symbol groups.
        Pass 2 (loose eps): Groups stacked-resolved symbols into rows/columns.

        FIX 1: Uses median symbol height instead of mean to be outlier-resistant.
        """
        if not detections:
            return {}

        # FIX 1: median instead of mean — outlier-resistant to large symbols like birds
        median_h = np.median([d["bbox"][3] - d["bbox"][1] for d in detections])

        # Store median_h and median_w for use in _are_stacked
        self._median_h = median_h
        self._median_w = np.median([d["bbox"][2] - d["bbox"][0] for d in detections])

        coords = np.array([[d["centre_x"], d["centre_y"]] for d in detections])

        # Pass 1: Tight eps to isolate stacked groups
        tight_eps = median_h * self.config.dbscan_tight_eps_factor
        pass1 = DBSCAN(eps=tight_eps, min_samples=self.config.dbscan_min_samples).fit(coords)

        # Compute centroid of each Pass 1 group to use as input for Pass 2
        pass1_groups: dict[int, list[dict]] = {}
        for i, label in enumerate(pass1.labels_):
            pass1_groups.setdefault(int(label), []).append(detections[i])

        group_centroids = []
        group_labels = []
        for label, syms in pass1_groups.items():
            cx = np.mean([s["centre_x"] for s in syms])
            cy = np.mean([s["centre_y"] for s in syms])
            group_centroids.append([cx, cy])
            group_labels.append(label)

        group_centroids = np.array(group_centroids)

        # Pass 2: Loose eps to group stacked clusters into rows/columns
        loose_eps = median_h * self.config.dbscan_loose_eps_factor
        pass2 = DBSCAN(eps=loose_eps, min_samples=self.config.dbscan_min_samples).fit(group_centroids)
        for i, row_label in enumerate(pass2.labels_):
            print(f"[cluster] Pass2 label={row_label} | "
                  f"centroid=({group_centroids[i][0]:.1f}, {group_centroids[i][1]:.1f})")

        # Map Pass 2 row-label → list of original detections
        final_groups: dict[int, list[dict]] = {}
        for i, row_label in enumerate(pass2.labels_):
            pass1_label = group_labels[i]
            syms = pass1_groups[pass1_label]
            final_groups.setdefault(int(row_label), []).extend(syms)

        return final_groups

    def sort_order(self, groups: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """
        Steps 4-7: Layout Classification, Inter/Intra Cluster Sorting, and Stacked Symbol Handling.
        FIX 4: Aspect ratio threshold replaces hard width > height comparison.
        """
        cluster_data = []

        for label, syms in groups.items():
            if not syms:
                continue

            x_centers = [s["centre_x"] for s in syms]
            y_centers = [s["centre_y"] for s in syms]

            wk = max(x_centers) - min(x_centers)
            hk = max(y_centers) - min(y_centers)

            # FIX 4: Aspect ratio with threshold gap instead of hard width > height
            aspect_ratio = wk / (hk + 1e-6)
            if aspect_ratio > 1.3:
                layout = "Row"
            elif aspect_ratio < 0.77:
                layout = "Column"
            else:
                layout = "Row"  # default fallback

            avg_x = np.mean(x_centers)
            avg_y = np.mean(y_centers)

            cluster_data.append({
                "label": label,
                "symbols": syms,
                "layout": layout,
                "avg_x": avg_x,
                "avg_y": avg_y
            })

        # Step 5: Inter-Cluster Sorting (RTL Assumed)
        def cluster_priority(c):
            if c["layout"] == "Row":
                return (c["avg_y"], -c["avg_x"])
            else:
                return (-c["avg_x"], c["avg_y"])

        cluster_data.sort(key=cluster_priority)

        final_sequence = []

        for cluster in cluster_data:
            syms = cluster["symbols"]

            # Step 6: Intra-Cluster Sorting (RTL Assumed)
            if cluster["layout"] == "Row":
                for s in syms:
                    print(f"[sort] symbol centre_x={s['centre_x']:.1f} bbox={s['bbox']}")
                syms.sort(key=lambda s: -s["centre_x"])
            else:
                syms.sort(key=lambda s: s["centre_y"])

            # Step 7: Stacked Symbol Detection & Handling
            resolved_syms = self._resolve_stacked_groups(syms, self._median_h, self._median_w)
            final_sequence.extend(resolved_syms)

        return final_sequence

    def _resolve_stacked_groups(self, syms: list[dict[str, Any]], median_h: float, median_w: float) -> list[dict[str, Any]]:
        """
        Detects and resolves stacked symbols/quadrants within a sequence.
        FIX 2: Checks ALL pairs instead of only adjacent pairs.
        FIX 5: Y-distance and X-center proximity gates with dynamic multipliers.
        """
        if len(syms) < 2:
            return syms

        n = len(syms)
        visited = [False] * n
        groups = []

        for i in range(n):
            if visited[i]:
                continue
            group = [syms[i]]
            visited[i] = True
            for j in range(n):
                if visited[j]:
                    continue
                for g in group:
                    if self._are_stacked(g, syms[j], median_h, median_w):
                        group.append(syms[j])
                        visited[j] = True
                        break
            groups.append(group)

        resolved = []
        for group in groups:
            print(f"[stack] group size={len(group)} "
                  f"centres={[(round(g['centre_x'],1), round(g['centre_y'],1)) for g in group]}")
            if len(group) == 1:
                resolved.append(group[0])
            elif len(group) == 2:
                # Simple stack: Top -> Bottom (by y_min)
                resolved.extend(sorted(group, key=lambda s: s["bbox"][1]))
            else:
                # Complex Quadrant Block (>= 3 symbols)
                resolved.extend(self._resolve_quadrant(group))

        return resolved

    def _resolve_quadrant(self, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Handles 2x2 or larger blocks by finding vertical sub-stacks."""
        sorted_by_x = sorted(group, key=lambda s: -s["centre_x"])

        sub_stacks = []
        if not sorted_by_x:
            return []

        curr_stack = [sorted_by_x[0]]
        for i in range(1, len(sorted_by_x)):
            if self._get_x_iou(curr_stack[0], sorted_by_x[i]) > 0.6:
                curr_stack.append(sorted_by_x[i])
            else:
                sub_stacks.append(curr_stack)
                curr_stack = [sorted_by_x[i]]
        sub_stacks.append(curr_stack)

        # Sort each sub-stack Top -> Bottom
        final = []
        for stack in sub_stacks:
            final.extend(sorted(stack, key=lambda s: s["bbox"][1]))
        return final

    def _are_stacked(self, a: dict, b: dict, median_h: float, median_w: float) -> bool:
        # Gate 0: Outlier exclusion — symbols much larger than median
        # cannot be part of a stacked group
        h_a = a["bbox"][3] - a["bbox"][1]
        h_b = b["bbox"][3] - b["bbox"][1]
        w_a = a["bbox"][2] - a["bbox"][0]
        w_b = b["bbox"][2] - b["bbox"][0]

        if h_a > 2.0 * median_h or h_b > 2.0 * median_h:
            return False
        if w_a > 2.0 * median_w or w_b > 2.0 * median_w:
            return False

        # Gate 1: X-IoU
        x_iou = self._get_x_iou(a, b)
        if x_iou <= self.config.quadrant_x_overlap_threshold:
            return False

        # Gate 2: Y-distance using smaller symbol's own height
        smaller_h = min(h_a, h_b)
        y_dist = abs(a["centre_y"] - b["centre_y"])
        y_multiplier = np.clip(median_h / (smaller_h + 1e-6), 0.8, 2.0)
        if y_dist >= (y_multiplier * smaller_h):
            return False

        # Gate 3: X-center proximity using smaller symbol's own width
        smaller_w = min(w_a, w_b)
        x_dist = abs(a["centre_x"] - b["centre_x"])
        x_multiplier = np.clip(median_w / (smaller_w + 1e-6), 0.5, 1.5)
        return x_dist < (x_multiplier * smaller_w)

    def _get_x_iou(self, a, b):
        """Calculates X-axis Intersection over Union."""
        ax1, _, ax2, _ = a["bbox"]
        bx1, _, bx2, _ = b["bbox"]
        inter = max(0, min(ax2, bx2) - max(ax1, bx1))
        min_w = min(ax2 - ax1, bx2 - bx1)
        return inter / min_w if min_w > 0 else 0
    # -------------------------------------------------------------------
    # Stage 5: Classification (EfficientNetV2B0)
    # -------------------------------------------------------------------

    def classify_symbols(
        self,
        image_bgr: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Crop each detected bounding box, resize to classification_img_size,
        and run through the EfficientNetV2B0 model to get Gardiner codes.

        NOTE: The model already includes preprocess_input in its graph,
        so crops are fed as raw float32 (range 0-255), NOT normalised.
        """
        model = self.load_classification_model()
        if model is None or not self._classification_label_map:
            for d in detections:
                d["gardiner_code"] = "Unknown"
                d["classification_confidence"] = 0.0
            return detections

        img_size = self.config.classification_img_size
        img_h, img_w = image_bgr.shape[:2]
        crops = []
        valid_indices = []

        padding_ratio = 0.05

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

            bw = x2 - x1
            bh = y2 - y1
            pad_w = int(bw * padding_ratio)
            pad_h = int(bh * padding_ratio)

            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(img_w, x2 + pad_w)
            py2 = min(img_h, y2 + pad_h)

            crop = image_bgr[py1:py2, px1:px2].copy()

            if crop.size == 0:
                crop = image_bgr[y1:y2, x1:x2].copy()

            if crop.size == 0:
                det["gardiner_code"] = "Unknown"
                det["classification_confidence"] = 0.0
                continue

            for other_det in detections:
                if other_det is det: continue
                ox1, oy1, ox2, oy2 = [int(c) for c in other_det["bbox"]]

                rx1 = max(0, ox1 - px1)
                ry1 = max(0, oy1 - py1)
                rx2 = min(crop.shape[1], ox2 - px1)
                ry2 = min(crop.shape[0], oy2 - py1)

                if rx2 > rx1 and ry2 > ry1:
                    bg_color = np.median(crop, axis=(0, 1)).tolist()
                    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), bg_color, -1)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (img_size, img_size))
            crop_float = crop_resized.astype(np.float32)
            crops.append(crop_float)
            valid_indices.append(idx)

        if not crops:
            return detections

        batch = np.array(crops)
        preds = model.predict(batch, verbose=0)

        for i, p in enumerate(preds):
            idx_in_detections = valid_indices[i]
            label_idx = np.argmax(p)
            confidence = float(np.max(p))

            code = self._classification_label_map.get(label_idx, f"Class_{label_idx}")
            if confidence < self.config.classification_confidence_threshold:
                code = "Unknown"

            detections[idx_in_detections]["gardiner_code"] = code
            detections[idx_in_detections]["classification_confidence"] = confidence

            print(f"[hieroglyph] Symbol {idx_in_detections}: Class {label_idx} -> {code} ({confidence:.2%})", flush=True)

        return detections

    # -------------------------------------------------------------------
    # Stage 6: Translation (M2M-100 Fine-tuned)
    # -------------------------------------------------------------------

    def translate_gardiner_sequence(self, gardiner_codes: list[str]) -> str:
        """
        Translate a sequence of Gardiner codes into English text
        using the fine-tuned M2M-100 model.

        Input format matches notebook: "translate MdC to English: D21 Q3 D36 F4 D36"
        """
        result = self.load_translation_model()
        if result is None:
            return ""

        model, tokenizer = result

        valid_codes = [c for c in gardiner_codes if c != "Unknown"]
        if not valid_codes:
            return ""

        mdc_sequence = " ".join(valid_codes)
        input_text = f"translate MdC to English: {mdc_sequence}"

        print(f"[hieroglyph] Final Gardiner Sequence: {mdc_sequence}", flush=True)
        print(f"[hieroglyph] Translation input: {input_text}", flush=True)

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.translation_max_input_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=self.config.translation_max_target_length,
                num_beams=3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True,
                do_sample=False
            )

        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"[hieroglyph] Translation output: {translation}", flush=True)

        return translation

    # -------------------------------------------------------------------
    # Full Pipeline Orchestration
    # -------------------------------------------------------------------

    def run_pipeline(self, image_bgr: np.ndarray, on_step: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Full pipeline: Preprocess → Detect → Refine → Cluster → Sort → Classify → Translate.
        """
        t0 = time.time()

        # Start (Scanning Inscription)
        if on_step: on_step(0.1) # Show the first icon immediately

        # 1. Preprocess
        enhanced = self.preprocess(image_bgr)

        # 2. Detect
        raw = self.detect(enhanced)
        
        # Phase 1 DONE -> Move to Determining Sequence (Icon 2)
        if on_step: on_step(1.0) 

        t_detect = time.time()
        print(f"[hieroglyph] Detection: {len(raw)} raw detections in {t_detect - t0:.2f}s", flush=True)

        # 3. Refine
        t_refine_start = time.time()
        clean = self.remove_duplicates(raw)
        filtered = self.deduplicate_by_center(clean, dist_thr=15.0)
        print(f"[hieroglyph] Refinement: {len(filtered)} final detections in {time.time() - t_refine_start:.2f}s", flush=True)

        empty_metadata = {
            "num_symbols_detected": 0,
            "num_clusters": 0,
            "pipeline_time_ms": int((time.time() - t0) * 1000),
        }

        if not filtered:
            return {
                "symbols": [],
                "num_symbols_detected": 0,
                "num_clusters": 0,
                "translation_text": "",
                "annotated_image_base64": None,
            }, empty_metadata

        # 4. Cluster & Sort
        t_order_start = time.time()
        groups = self.cluster(filtered)
        ordered = self.sort_order(groups)
        
        # Phase 2 DONE -> Move to Recognizing Symbols (Icon 3)
        if on_step: on_step(2.0) 
        
        print(f"[hieroglyph] Clustering & Sorting done in {time.time() - t_order_start:.2f}s", flush=True)

        # 5. Classify (Gardiner codes)
        t_classify_start = time.time()
        classified_symbols = self.classify_symbols(image_bgr, ordered)
        
        # Phase 3 DONE -> Move to Generating Translation (Icon 4)
        if on_step: on_step(3.0) 
        
        print(f"[hieroglyph] Classification: {len(classified_symbols)} symbols in {time.time() - t_classify_start:.2f}s", flush=True)

        # 6. Translate (Gardiner → English)
        t_translate_start = time.time()
        gardiner_codes = [s["gardiner_code"] for s in classified_symbols]
        translation_text = self.translate_gardiner_sequence(gardiner_codes)
        print(f"[hieroglyph] Translation done in {time.time() - t_translate_start:.2f}s", flush=True)
        
        # Phase 4 DONE
        if on_step: on_step(4.0)
        
        print(f"[hieroglyph] Translation done in {time.time() - t_translate_start:.2f}s", flush=True)

        # 7. Format output
        symbols = [{
            "gardiner_code": s["gardiner_code"],
            "classification_confidence": float(s.get("classification_confidence", 0.0)),
            "bbox": [float(c) for c in s["bbox"]],
            "detection_confidence": float(s["confidence"]),
        } for s in classified_symbols]

        # 8. Debug Annotation (Server-side)
        try:
            os.makedirs("debug", exist_ok=True)
            debug_img = image_bgr.copy()
            for s in classified_symbols:
                x1, y1, x2, y2 = [int(c) for c in s["bbox"]]
                label = f"{s.get('gardiner_code', '???')} {s.get('classification_confidence', 0.0):.2f}"
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (60, 178, 230), 2)
                cv2.putText(debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 178, 230), 1)
            cv2.imwrite("debug/annotated_debug.jpg", debug_img)
            print(f"[hieroglyph] Saved debug visualization to debug/annotated_debug.jpg", flush=True)
        except Exception as e:
            print(f"[hieroglyph] WARNING: Could not save debug image: {e}", flush=True)

        # 9. Encode original image (no bboxes for frontend)
        _, buf = cv2.imencode(".jpg", image_bgr)

        pipeline_time_ms = int((time.time() - t0) * 1000)
        print(f"[hieroglyph] Full pipeline completed in {pipeline_time_ms}ms", flush=True)

        result = {
            "symbols": symbols,
            "num_symbols_detected": len(symbols),
            "num_clusters": len(groups),
            "translation_text": translation_text,
            "annotated_image_base64": f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}",
        }
        metadata = {
            "num_symbols_detected": len(symbols),
            "num_clusters": len(groups),
            "pipeline_time_ms": pipeline_time_ms,
        }
        return result, metadata


hieroglyph_runtime = HieroglyphDetectionRuntime()