import os
import io
import time
import ctypes
import numpy as np
from PIL import Image

# Force load the correct cuDNN library from the venv to fix the 9.1 vs 9.3 mismatch
try:
    ctypes.CDLL("/workspace/venv/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    # Try the global dist-packages path if venv fails
    try:
        ctypes.CDLL("/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib/libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)
    except:
        print(f"[ML] Warning: Could not force-load cuDNN 9: {e}")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras') # Silence Keras structure warnings
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"[ML] Error setting memory growth: {e}")
    tf = None

import joblib
from src.app.core.config import settings


IMG_SIZE = 256

PREPROCESS_MODES = {
    "binary":   "raw_255",
    "pharaoh":  "convnext",
    "landmark": "scale_01",
}

MODEL_FILES = {
    "binary":   "Binary.keras",
    "pharaoh":  "Pharohs.keras",
    "landmark": "Landmarks.keras",
}
ENCODER_FILES = {
    "binary":   "Binary.pkl",
    "pharaoh":  "Pharohs.pkl",
    "landmark": "Landmarks.pkl",
}


class RecognitionInference:

    def __init__(self):
        self.model_path = settings.MODEL_PATH

        if not os.path.isdir(self.model_path):
            fallback = os.path.join(settings.BASE_DIR, "ml_models", "recognition_models")
            if os.path.isdir(fallback):
                self.model_path = fallback

        # Load encoders
        self.binary_encoder = self._load_encoder("binary")
        self.pharaoh_encoder = self._load_encoder("pharaoh")
        self.landmark_encoder = self._load_encoder("landmark")

        # Load models
        if tf is None:
            print("[ML] WARNING: TensorFlow not installed.")
            self.binary_model = self.pharaoh_model = self.landmark_model = None
            return

        try:
            self.binary_model = self._load_model("binary")
            self.pharaoh_model = self._load_model("pharaoh")
            self.landmark_model = self._load_model("landmark")
            print(f"[ML] Models loaded from {self.model_path}")
            device = "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU"
            print(f"[ML] Recognition running on: {device}")
        except Exception as e:
            print(f"[ML] CRITICAL: {e}")
            self.binary_model = self.pharaoh_model = self.landmark_model = None

    def _load_encoder(self, key: str):
        path = os.path.join(self.model_path, ENCODER_FILES[key])
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    def _load_model(self, key: str):
        path = os.path.join(self.model_path, MODEL_FILES[key])
        if not os.path.exists(path):
            return None
        return tf.keras.models.load_model(path)

    def preprocess(self, image: Image.Image, mode: str) -> np.ndarray:
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

        if mode == "convnext":
            arr = tf.keras.applications.convnext.preprocess_input(arr)
        elif mode == "scale_01":
            arr = arr / 255.0

        return np.expand_dims(arr, axis=0)

    async def run_hierarchical_inference(self, image_data: bytes, debug: bool = False) -> dict:
        start_time = time.perf_counter()
        if not self.binary_model:
            raise RuntimeError("Binary model not loaded.")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Stage 1: Binary — pharaoh or landmark?
        bin_tensor = tf.convert_to_tensor(self.preprocess(image, PREPROCESS_MODES["binary"]))
        bin_pred = self.binary_model(bin_tensor, training=False).numpy()[0]

        # Sigmoid gives 1 value; expand to [class_0_prob, class_1_prob] so we can argmax
        p = float(bin_pred[0])
        probs = np.array([1.0 - p, p])
        bin_idx = int(np.argmax(probs))
        predicted_type = self.binary_encoder.inverse_transform([bin_idx])[0].lower()
        bin_conf = float(probs[bin_idx])

        if "pharaoh" in predicted_type:
            predicted_type = "pharaoh"
        else:
            predicted_type = "landmark"

        # Stage 2: Specialized — which pharaoh / which landmark?
        model = self.pharaoh_model if predicted_type == "pharaoh" else self.landmark_model
        encoder = self.pharaoh_encoder if predicted_type == "pharaoh" else self.landmark_encoder

        if not model:
            raise RuntimeError(f"{predicted_type.title()} model not loaded.")

        spec_tensor = tf.convert_to_tensor(self.preprocess(image, PREPROCESS_MODES[predicted_type]))
        spec_pred = model(spec_tensor, training=False).numpy()[0]

        idx = int(np.argmax(spec_pred))
        predicted_name = str(encoder.inverse_transform([idx])[0])
        final_conf = float(spec_pred[idx])

        elapsed = time.perf_counter() - start_time
        print(f"[ML] Recognition: '{predicted_name}' ({predicted_type}) identified in {elapsed:.2f}s")

        return {
            "type": predicted_type,
            "name": predicted_name,
            "confidence": final_conf,
            "binary_confidence": bin_conf,
        }

    def warmup(self):
        """Perform a dummy inference to warm up GPU kernels."""
        if not self.binary_model:
            return
        try:
            print("[ML] Warming up GPU kernels...")
            dummy_img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            # Binary warmup
            bin_tensor = tf.convert_to_tensor(self.preprocess(dummy_img, PREPROCESS_MODES["binary"]))
            _ = self.binary_model(bin_tensor, training=False)
            
            # Specialized warmup (using pharaoh model as proxy)
            if self.pharaoh_model:
                spec_tensor = tf.convert_to_tensor(self.preprocess(dummy_img, PREPROCESS_MODES["pharaoh"]))
                _ = self.pharaoh_model(spec_tensor, training=False)
            print("[ML] GPU Warmup complete.")
        except Exception as e:
            print(f"[ML] Warmup failed: {e}")


recognition_inference = RecognitionInference()
recognition_inference.warmup()
