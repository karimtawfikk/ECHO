from __future__ import annotations

from open_clip import create_model_and_transforms


def main() -> None:
    # Preload the CLIP assets used by the video runtime so the image is warm.
    create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    print("Video models preloaded.")


if __name__ == "__main__":
    main()
