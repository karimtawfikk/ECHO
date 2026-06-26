import os, shutil, yaml, random, cv2, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm.auto import tqdm
import albumentations as A

KAGGLE_INPUT   = "/kaggle/input/finaldatadetection1/Final_Dataset" 
KAGGLE_WORKING = "/kaggle/working/hiero_pro"
SLICED_DIR     = f"{KAGGLE_WORKING}/sliced_dataset"
YAML_PATH      = f"{KAGGLE_WORKING}/hiero.yaml"
RUN_DIR        = f"{KAGGLE_WORKING}/runs"

for d in [KAGGLE_WORKING, SLICED_DIR, RUN_DIR]:
    os.makedirs(d, exist_ok=True)

SLICE_SIZE  = 640       
OVERLAP     = 0.50      
STEP        = int(SLICE_SIZE * (1 - OVERLAP))   # 320
MIN_VIS     = 0.30     
SPLITS      = ["train", "val", "test"]

print(f"✅ Workspace: {KAGGLE_WORKING}")
print(f"   tile={SLICE_SIZE}  step={STEP}  min_visibility={MIN_VIS}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

def apply_pro_enhancement(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, kernel)

_test_files = list(Path(KAGGLE_INPUT).rglob("*.jpg"))[:1]
if _test_files:
    _img = cv2.imread(str(_test_files[0]))
    _enh = apply_pro_enhancement(_img)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original")
    ax[1].imshow(cv2.cvtColor(_enh, cv2.COLOR_BGR2RGB)); ax[1].set_title("CLAHE + Sharpened")
    plt.tight_layout(); plt.show()

# 
def count_dataset(root: str):
    stats = {}
    for split in SPLITS:
        img_dir = Path(root) / split / "images"
        lbl_dir = Path(root) / split / "labels"
        imgs = list(img_dir.glob("*.*")) if img_dir.exists() else []
        lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        box_count = sum(len(open(l).readlines()) for l in lbls)
        stats[split] = {"images": len(imgs), "labels": len(lbls), "boxes": box_count}
    df = pd.DataFrame(stats).T
    print("\n📊 Dataset stats:\n", df.to_string())
    return df

count_dataset(KAGGLE_INPUT)

def yolo_to_pixel(label_line: str, img_w: int, img_h: int):
    parts = label_line.strip().split()
    cls = int(parts[0])
    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return cls, x1, y1, x2, y2


def pixel_to_yolo(cls: int, x1, y1, x2, y2, tile_w: int, tile_h: int) -> str:
    """Convert pixel bbox in tile space → YOLO label string."""
    xc = ((x1 + x2) / 2) / tile_w
    yc = ((y1 + y2) / 2) / tile_h
    bw = (x2 - x1) / tile_w
    bh = (y2 - y1) / tile_h
    # Clamp to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def slice_image_and_labels(
    img_path: Path,
    lbl_path: Path | None,
    out_img_dir: Path,
    out_lbl_dir: Path,
    slice_size: int = SLICE_SIZE,
    step: int = STEP,
    min_visibility: float = MIN_VIS,
    enhance: bool = True,
) -> int:

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return 0

    if enhance:
        img_bgr = apply_pro_enhancement(img_bgr)

    H, W = img_bgr.shape[:2]

    raw_labels: list[str] = []
    if lbl_path and lbl_path.exists():
        raw_labels = open(lbl_path).readlines()

    parsed_boxes: list[tuple] = []
    for line in raw_labels:
        line = line.strip()
        if not line:
            continue
        try:
            parsed_boxes.append(yolo_to_pixel(line, W, H))
        except Exception:
            pass

    y_offs = list(range(0, max(1, H - slice_size + 1), step))
    if H > slice_size and y_offs[-1] != H - slice_size:
        y_offs.append(H - slice_size)
    x_offs = list(range(0, max(1, W - slice_size + 1), step))
    if W > slice_size and x_offs[-1] != W - slice_size:
        x_offs.append(W - slice_size)

    stem = img_path.stem
    tile_count = 0

    for yi, oy in enumerate(y_offs):
        for xi, ox in enumerate(x_offs):
            tile = img_bgr[oy: oy + slice_size, ox: ox + slice_size]
            th, tw = tile.shape[:2]

            tile_labels: list[str] = []
            for cls, bx1, by1, bx2, by2 in parsed_boxes:
                tx1 = max(bx1, ox) - ox
                ty1 = max(by1, oy) - oy
                tx2 = min(bx2, ox + tw) - ox
                ty2 = min(by2, oy + th) - oy

                if tx2 <= tx1 or ty2 <= ty1:
                    continue  # no intersection

                # Visibility check
                orig_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
                inter_area = (tx2 - tx1) * (ty2 - ty1)
                if inter_area / orig_area < min_visibility:
                    continue

                tile_labels.append(pixel_to_yolo(cls, tx1, ty1, tx2, ty2, tw, th))

            # Only save tiles that have at least one label
            if not tile_labels:
                continue

            tile_name = f"{stem}_t{yi:02d}_{xi:02d}"
            cv2.imwrite(str(out_img_dir / f"{tile_name}.jpg"), tile)
            with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                f.write("\n".join(tile_labels))
            tile_count += 1

    return tile_count


def process_split(split: str, enhance: bool = True):
    src_img = Path(KAGGLE_INPUT) / split / "images"
    src_lbl = Path(KAGGLE_INPUT) / split / "labels"

    dst_img = Path(SLICED_DIR) / split / "images"
    dst_lbl = Path(SLICED_DIR) / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(src_img.glob("*.*"))
    total_tiles = 0
    for img_path in tqdm(images, desc=f"Slicing {split}"):
        lbl_path = src_lbl / (img_path.stem + ".txt")
        total_tiles += slice_image_and_labels(
            img_path, lbl_path, dst_img, dst_lbl, enhance=(split != "test")
        )

    print(f"  [{split}] {len(images)} images → {total_tiles} tiles saved")
    return total_tiles


print("  Slicing dataset (CLAHE + sharpening applied to train/val)...")
for split in SPLITS:
    process_split(split)

print("\n✅ Slicing complete!")
count_dataset(SLICED_DIR)

aug_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.ISONoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.15),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Perspective(scale=(0.03, 0.08), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=8, p=0.5),
        A.RandomShadow(p=0.15),
        A.ToGray(p=0.1),   # some stone images are greyscale
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,
    ),
)


def augment_split(n_aug_per_image: int = 2):
    img_dir = Path(SLICED_DIR) / "train" / "images"
    lbl_dir = Path(SLICED_DIR) / "train" / "labels"

    original_tiles = sorted(img_dir.glob("*.jpg"))
    added = 0

    for img_path in tqdm(original_tiles, desc="Augmenting train tiles"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, classes = [], []
        for line in open(lbl_path).readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                classes.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])

        if not bboxes:
            continue

        for aug_idx in range(n_aug_per_image):
            try:
                result = aug_pipeline(
                    image=img_rgb, bboxes=bboxes, class_labels=classes
                )
                aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                aug_bboxes = result["bboxes"]
                aug_classes = result["class_labels"]

                if not aug_bboxes:
                    continue

                aug_stem = f"{img_path.stem}_aug{aug_idx}"
                cv2.imwrite(str(img_dir / f"{aug_stem}.jpg"), aug_img)
                with open(lbl_dir / f"{aug_stem}.txt", "w") as f:
                    for cls, bbox in zip(aug_classes, aug_bboxes):
                        f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                added += 1
            except Exception as e:
                pass

    print(f" Augmentation done: +{added} tiles  (total train tiles: {len(list(img_dir.glob('*.jpg')))})")


augment_split(n_aug_per_image=2)

lbl_files = list((Path(SLICED_DIR) / "train" / "labels").glob("*.txt"))
all_classes = set()
for lf in lbl_files:
    for line in open(lf):
        parts = line.strip().split()
        if parts:
            all_classes.add(int(parts[0]))

NC = len(all_classes) if all_classes else 1
print(f" Detected {NC} class(es): {sorted(all_classes)}")

dataset_yaml = {
    "path": SLICED_DIR,
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    NC,
    "names": {i: f"class_{i}" for i in sorted(all_classes)},
}

with open(YAML_PATH, "w") as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)


model = YOLO("yolo11x.pt")  

results = model.train(
    data=YAML_PATH,
    project=RUN_DIR,
    name="hiero_v1",

    epochs=200,
    patience=40,          
    imgsz=640,
    batch=16,           
    workers=4,
    seed=42,
    deterministic=True,

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    box=7.5,
    cls=0.5,
    dfl=1.5,


    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    degrees=6.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    erasing=0.4,
    auto_augment="randaugment",

    val=True,
    save=True,
    save_period=20,        
    plots=True,
    verbose=True,

    iou=0.5,
    conf=0.001,            
    max_det=500,
)

print("\n✅ Training complete!")
print(f"   Best weights: {RUN_DIR}/hiero_v1/weights/best.pt")

best_weights = f"{RUN_DIR}/hiero_v1/weights/best.pt"
model_eval = YOLO(best_weights)

metrics = model_eval.val(
    data=YAML_PATH,
    split="test",
    imgsz=640,
    conf=0.05,
    iou=0.20,
    plots=True,
    save_json=True,
    project=RUN_DIR,
    name="hiero_v1_test",
    verbose=True,
)

print("\n📈 Test Metrics:")
print(f"   mAP50:    {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall:    {metrics.box.mr:.4f}")

def run_inference_visual(model_path: str, img_dir: str, n_samples: int = 6,
                          conf: float = 0.05, iou: float = 0.20):
    """
    Run prediction on test images with CLAHE preprocessing (matches backend).
    Draws golden boxes on the original image.
    """
    inf_model = YOLO(model_path)
    images = sorted(Path(img_dir).glob("*.*"))
    samples = random.sample(images, min(n_samples, len(images)))

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for ax, img_path in zip(axes, samples):
        raw = cv2.imread(str(img_path))
        enhanced = apply_pro_enhancement(raw)

        results = inf_model.predict(
            source=[enhanced],
            conf=conf,
            iou=iou,
            imgsz=640,
            augment=False,
            verbose=False,
        )

        annotated = raw.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 178, 230), 2)
                cv2.putText(annotated, f"{score:.2f}", (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 178, 230), 1)

        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.name}\nboxes={len(results[0].boxes)}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.suptitle("🏺 Test Inference (CLAHE + YOLO11x)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{KAGGLE_WORKING}/test_inference.png", dpi=150, bbox_inches="tight")
    plt.show()


run_inference_visual(
    model_path=best_weights,
    img_dir=f"{SLICED_DIR}/test/images",
    n_samples=6,
)

results_csv = Path(f"{RUN_DIR}/hiero_v1/results.csv")
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Class Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
        ("metrics/precision(B)", "metrics/recall(B)", "Precision vs Recall"),
        ("metrics/mAP50(B)", None, "mAP@50"),
        ("metrics/mAP50-95(B)", None, "mAP@50-95"),
    ]
    for ax, (col1, col2, title) in zip(axes.flatten(), pairs):
        if col1 in df.columns:
            ax.plot(df[col1], label=col1.split("/")[-1], linewidth=2)
        if col2 and col2 in df.columns:
            ax.plot(df[col2], label=col2.split("/")[-1], linewidth=2, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("📈 Training Curves — YOLO11x Hieroglyph Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{KAGGLE_WORKING}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


import shutil

output_dir = Path("/kaggle/working")
shutil.copy(best_weights, output_dir / "hieroglyph_detection_v2.pt")
