from __future__ import annotations

import os
import time

from open_clip import create_model_and_transforms


def main() -> None:
    model_name = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    hf_home = os.getenv("HF_HOME", "<not set>")
    start_time = time.time()

    print(
        f"[video_preload] Starting CLIP preload model={model_name} pretrained={pretrained}",
        flush=True,
    )
    print(f"[video_preload] HF_HOME={hf_home}", flush=True)

    # Preload the CLIP assets used by the video runtime so the image is warm.
    create_model_and_transforms(model_name, pretrained=pretrained)

    elapsed = time.time() - start_time
    print(f"[video_preload] CLIP preload finished in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
