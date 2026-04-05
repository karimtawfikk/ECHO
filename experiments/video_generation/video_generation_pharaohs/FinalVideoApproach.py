from __future__ import annotations

import asyncio
import hashlib 
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
import time
from pathlib import Path
from typing import List, Sequence

import boto3
import edge_tts
import librosa
import numpy as np
import open_clip
import pillow_avif  
import soundfile as sf
import torch
from dotenv import load_dotenv
from PIL import Image
from scipy.io import wavfile
from sqlalchemy import text
from sqlalchemy.orm import Session

# -----------------------------------------------------------------------------
# Project import setup
# -----------------------------------------------------------------------------
root = Path.cwd()
while root != root.parent and not (root / "src").exists():
    root = root.parent

sys.path.append(str(root))
from src.db.session import engine 


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class VideoPipelineConfig:
    output_dir: str = "tts_Outputs"
    temp_frames_dir: str = "temp_frames"
    temp_clips_dir: str = "temp_clips"

    voice: str = "en-US-ChristopherNeural"
    rate: str = "+0%"
    trim_top_db: int = 20

    seconds_per_image: float = 6.0
    fade: float = 0.45

    fps: int = 30
    target_w: int = 1920
    target_h: int = 1080

    max_chars_per_line: int = 35
    max_lines: int = 2
    min_subtitle_duration: float = 1.0

    image_sim_weight: float = 0.3
    desc_sim_weight: float = 0.7

    use_nvenc: bool = True
    cleanup_intermediate: bool = True


# -----------------------------------------------------------------------------
# CLIP model cache
# -----------------------------------------------------------------------------
_CLIP_MODEL = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_clip_model():
    global _CLIP_MODEL, _CLIP_TOKENIZER
    if _CLIP_MODEL is None or _CLIP_TOKENIZER is None:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained="laion2b_s32b_b79k",
        )
        model = model.to(_CLIP_DEVICE)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-H-14")

        _CLIP_MODEL = model
        _CLIP_TOKENIZER = tokenizer
    return _CLIP_MODEL, _CLIP_TOKENIZER, _CLIP_DEVICE


# -----------------------------------------------------------------------------
# Script loading and text splitting
# -----------------------------------------------------------------------------
def get_script_by_name(name: str, is_landmark: bool = False) -> str | None:
    with Session(engine) as session:
        if is_landmark:
            result = session.execute(
                text(
                    """
                    SELECT landmark_script
                    FROM landmarks_scripts AS ls, landmarks AS l
                    WHERE ls.landmark_id = l.id AND l.name = :name
                    """
                ),
                {"name": name},
            )
        else:
            result = session.execute(
                text(
                    """
                    SELECT pharaoh_script
                    FROM pharaohs_scripts AS ps, pharaohs AS p
                    WHERE ps.pharaoh_id = p.id AND p.name = :name
                    """
                ),
                {"name": name},
            )
        rows = result.fetchall()

    if not rows:
        return None
    return rows[0][0]


def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def split_script_into_paragraph_sentences(script: str) -> tuple[List[str], List[List[str]]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script.strip()) if p.strip()]
    sentence_groups = [split_into_sentences(p) for p in paragraphs]
    return paragraphs, sentence_groups


# -----------------------------------------------------------------------------
# Audio generation
# -----------------------------------------------------------------------------
async def edge_tts_save(text: str, out_path: str | Path, voice: str, rate: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(str(out_path))

def trim_trailing_silence(audio_path: str | Path, top_db: int = 20) -> None:
    y, sr = librosa.load(str(audio_path), sr=None)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    sf.write(str(audio_path), yt, sr)

async def generate_tts_audio(
    sentence_groups: Sequence[Sequence[str]],
    output_dir: str = "tts_Outputs",
    voice: str = "en-US-ChristopherNeural",
    rate: str = "+0%",
    trim_top_db: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for paragraph in sentence_groups:
        for sentence in paragraph:
            out_path = Path(output_dir) / f"output_{i}.wav"

            await edge_tts_save(sentence, out_path, voice=voice, rate=rate)
            trim_trailing_silence(out_path, top_db=trim_top_db)

            i += 1


def combine_audio_files(output_dir: str = "tts_Outputs", final_name: str = "Final audio.wav") -> str:
    wav_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".wav") and f.startswith("output_")],
        key=lambda x: int(re.search(r"output_(\d+)\.wav", x).group(1)),
    )

    audio_data = []
    samplerate = None

    for file_name in wav_files:
        file_path = Path(output_dir) / file_name
        data, sr = sf.read(file_path)
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise ValueError("Sample rates do not match.")
        audio_data.append(data)

    if not audio_data:
        raise ValueError("No sentence wav files were found to combine.")

    combined = np.concatenate(audio_data, axis=0)
    final_path = Path(output_dir) / final_name
    sf.write(final_path, combined, samplerate)
    return str(final_path)


def compute_audio_durations(
    sentence_groups: Sequence[Sequence[str]],
    output_dir: str = "tts_Outputs",
    seconds_per_image: float = 6.0,
    delete_sentence_files: bool = True,
) -> tuple[List[float], List[float], List[int]]:
    paragraph_durations: List[float] = []
    sentence_durations: List[float] = []
    images_needed: List[int] = []

    i = 0
    for paragraph in sentence_groups:
        duration_seconds = 0.0
        for _sentence in paragraph:
            file_path = Path(output_dir) / f"output_{i}.wav"
            fs, data = wavfile.read(file_path)
            sentence_duration = len(data) / float(fs)
            duration_seconds += sentence_duration
            sentence_durations.append(sentence_duration)
            if delete_sentence_files and file_path.exists():
                file_path.unlink()
            i += 1

        paragraph_durations.append(duration_seconds)
        images_needed.append(max(1, int(duration_seconds / seconds_per_image)))

    return paragraph_durations, sentence_durations, images_needed


def create_image_chunks(
    sentence_groups: Sequence[Sequence[str]],
    images_needed: Sequence[int],
    sentence_durations: Sequence[float],
) -> tuple[List[List[str]], List[float]]:
    sentence_start = 0
    image_text_chunks: List[List[str]] = []
    seconds_for_chunk: List[float] = []

    for para_idx, paragraph_sentences in enumerate(sentence_groups):
        images_for_paragraph = min(images_needed[para_idx], len(paragraph_sentences))

        total_sentences = len(paragraph_sentences)
        base = total_sentences // images_for_paragraph
        remainder = total_sentences % images_for_paragraph

        groups = []
        start = 0
        for img_idx in range(images_for_paragraph):
            extra = 1 if img_idx < remainder else 0
            end = start + base + extra

            chunk_duration = sum(sentence_durations[sentence_start + start : sentence_start + end])
            seconds_for_chunk.append(chunk_duration)

            chunk = " ".join(paragraph_sentences[start:end])
            groups.append(chunk)
            start = end

        sentence_start += total_sentences
        image_text_chunks.append(groups)

    return image_text_chunks, seconds_for_chunk

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _fetch_candidate_images(name: str, is_landmark: bool = False):
    with Session(engine) as session:
        if is_landmark:
            result = session.execute(
                text(
                    """
                    SELECT
                        li.id,
                        li.image_path,
                        li.image_embedding
                    FROM landmark_images li
                    JOIN landmarks l ON li.landmark_id = l.id
                    WHERE l.name = :name
                    """
                ),
                {"name": name},
            )
        else:
            result = session.execute(
                text(
                    """
                    SELECT
                        pi.id,
                        pi.image_path,
                        pi.image_embedding,
                        pi.image_description
                    FROM pharaohs_images pi
                    JOIN pharaohs p ON pi.pharaoh_id = p.id
                    WHERE p.name = :name
                    """
                ),
                {"name": name},
            )
        return result.fetchall()


def retrieve_images_semantic(
    name: str,
    image_text_chunks: Sequence[Sequence[str]],
    is_landmark: bool = False,
    image_sim_weight: float = 0.3,
    desc_sim_weight: float = 0.7,
) -> tuple[List[int], List[str], float, float]:
    
    model, tokenizer, device = get_clip_model()
    images_data = _fetch_candidate_images(name, is_landmark=is_landmark)

    fetched_image_ids: List[int] = []
    fetched_image_paths: List[str] = []
    repeated_image_ids: List[int] = []
    best_scores: List[float] = []
    trials: List[int] = []
    processed_images = []

    for row in images_data:
        if is_landmark:
            image_id, image_path, image_embedding = row
            image_description = None
        else:
            image_id, image_path, image_embedding, image_description = row
            
        if isinstance(image_embedding, str):
            image_embedding = json.loads(image_embedding)

        image_embedding = np.array(image_embedding, dtype=np.float32)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)

        desc_emb = None
        if (not is_landmark) and (not image_description):
            tokens = tokenizer([image_description]).to(device)
            with torch.no_grad():
                desc_emb = model.encode_text(tokens)
                desc_emb /= desc_emb.norm(dim=-1, keepdim=True)
            desc_emb = desc_emb.cpu().numpy()[0]

        processed_images.append(
            {
                "id": image_id,
                "path": image_path,
                "img_emb": image_embedding,
                "desc_emb": desc_emb,
            }
        )

    for paragraph_chunks in image_text_chunks:
        for chunk in paragraph_chunks:
            text_tokens = tokenizer([chunk]).to(device)
            with torch.no_grad():
                emb = model.encode_text(text_tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
            scene_emb = emb.cpu().numpy()[0]

            ranked = []
            for img in processed_images:
                image_sim = cosine(scene_emb, img["img_emb"])
                if is_landmark or img["desc_emb"] is None:
                    score = image_sim
                else:
                    desc_sim = cosine(scene_emb, img["desc_emb"])
                    score = image_sim_weight * image_sim + desc_sim_weight * desc_sim
                ranked.append((score, img["id"], img["path"]))

            ranked.sort(reverse=True, key=lambda x: x[0])
            ranked_backup = ranked.copy()
            best_score, best_id, best_path = ranked[0]
            j = 0

            while best_id in fetched_image_ids:
                ranked.pop(0)
                j += 1
                if not ranked:
                    best_score, best_id, best_path = ranked_backup[0]
                    while best_id in repeated_image_ids:
                        ranked_backup.pop(0)
                        j += 1
                        if not ranked_backup:
                            raise RuntimeError("No more images available for retrieval.")
                        best_score, best_id, best_path = ranked_backup[0]
                    repeated_image_ids.append(best_id)
                    break
                best_score, best_id, best_path = ranked[0]

            fetched_image_ids.append(best_id)
            fetched_image_paths.append(best_path)
            best_scores.append(best_score)
            trials.append(j)

    average_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
    average_trials = sum(trials) / len(trials) if trials else 0.0
    return fetched_image_ids, fetched_image_paths, average_score, average_trials


# -----------------------------------------------------------------------------
# R2 download and image normalization
# -----------------------------------------------------------------------------
def _build_r2_client():
    load_dotenv()
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY")
    secret_key = os.getenv("R2_SECRET_KEY")

    if not all([account_id, access_key, secret_key]):
        raise ValueError("Missing one or more Cloudflare R2 environment variables.")

    session = boto3.session.Session()
    return session.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def download_images_from_r2(remote_paths: Sequence[str], local_frames_dir: str = "temp_frames") -> List[str]:
    load_dotenv()
    bucket_name = os.getenv("R2_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("R2_BUCKET_NAME is not set.")

    client = _build_r2_client()
    local_dir = Path(local_frames_dir)
    local_dir.mkdir(exist_ok=True)

    def download_image(idx_key):
        idx, image_key = idx_key
        suffix = Path(image_key).suffix.lower() or ".jpg"
        local_file = local_dir / f"{idx:04d}{suffix}"
        client.download_file(bucket_name, image_key, str(local_file))
        return str(local_file)

    with ThreadPoolExecutor(max_workers=8) as executor:
        ordered_local_paths = list(executor.map(download_image, enumerate(remote_paths)))

    return ordered_local_paths


def normalize_images_to_jpeg(image_files: Sequence[str | Path]) -> List[str]:
    normalized_paths: List[str] = []
    for image_path in image_files:
        p = Path(image_path)
        jpg_path = p.with_suffix(".jpg")
        with Image.open(p) as img:
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
        normalized_paths.append(str(jpg_path))
    return sorted(normalized_paths)


# -----------------------------------------------------------------------------
# Subtitles
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "’": "'",
        "‘": "'",
        "‚": ",",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "—": "-",
        "–": "-",
        "―": "-",
        "…": "...",
        "\u00A0": " ",
        "\u200B": "",
        "\u200C": "",
        "\u200D": "",
        "\uFEFF": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return text


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def format_stage_time(seconds: float) -> str:
    return f"{seconds:.2f}s ({seconds / 60:.2f} min)"

def split_long_text(text: str, max_chars: int = 35, max_lines: int = 2) -> List[str]:
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
            proposed = f"{current_line} {word}".strip()
            if len(proposed) <= max_chars:
                current_line = proposed
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

    if current_line:
        lines.append(current_line)

    blocks = []
    for i in range(0, len(lines), max_lines):
        blocks.append("\n".join(lines[i : i + max_lines]))
    return blocks


def generate_srt(
    paragraphs: Sequence[str],
    sentence_durations: Sequence[float],
    output_path: str = "subtitles.srt",
    max_chars_per_line: int = 35,
    max_lines: int = 2,
    min_duration: float = 1.0,
) -> str:
    current_time = 0.0
    subtitle_index = 1
    srt_blocks = []
    duration_index = 0

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        sentences = split_into_sentences(paragraph)

        for sentence in sentences:
            sentence_duration = sentence_durations[duration_index]
            duration_index += 1

            chunks = split_long_text(sentence, max_chars=max_chars_per_line, max_lines=max_lines)
            total_chars = sum(len(c.replace("\n", "")) for c in chunks)

            for chunk in chunks:
                chunk_char_count = len(chunk.replace("\n", ""))
                chunk_duration = max(min_duration, (chunk_char_count / total_chars) * sentence_duration)

                start_time = current_time
                end_time = current_time + chunk_duration

                srt_blocks.append(
                    f"{subtitle_index}\n"
                    f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
                    f"{chunk}\n\n"
                )
                current_time = end_time
                subtitle_index += 1

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.writelines(srt_blocks)
    return output_path

# -----------------------------------------------------------------------------
# FFmpeg helpers
# -----------------------------------------------------------------------------
def run_ffmpeg(cmd: Sequence[str]) -> None:
    cmd = [str(c) for c in cmd]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg failed.")

def _stable_int(seed: str) -> int:
    return int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)


def _stable_unit(seed: str) -> float:
    return (_stable_int(seed) % 10_000) / 10_000.0


# -----------------------------------------------------------------------------
# Ken Burns planning and rendering
# -----------------------------------------------------------------------------
def plan_kenburns_sequence(
    image_files: Sequence[str | Path],
    durations: Sequence[float],
    target_w: int = 1920,
    target_h: int = 1080,
    threshold: int = 40,
    min_pan_travel: int = 140,
    max_pan_speed: int = 120,
    vertical_travel_ratio: float = 0.80,
    horizontal_travel_ratio: float = 0.55,
):
    palette = [("zoom", "in"), ("pan", "forward"), ("zoom", "out"), ("pan", "reverse")]
    plans = []
    palette_idx = 0
    last_sig = None
    target_ar = target_w / target_h

    for img_path, duration in zip(image_files, durations):
        img_path = Path(img_path)

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        scale_factor = max(target_w / img_w, target_h / img_h)
        scaled_w = int(round(img_w * scale_factor))
        scaled_h = int(round(img_h * scale_factor))
        scaled_w = scaled_w if scaled_w % 2 == 0 else scaled_w + 1
        scaled_h = scaled_h if scaled_h % 2 == 0 else scaled_h + 1

        max_x = max(0, scaled_w - target_w)
        max_y = max(0, scaled_h - target_h)

        img_ar = img_w / img_h
        is_vertical = img_ar < target_ar
        pan_axis = "vertical" if is_vertical else "horizontal"

        travel_x = min(max_x, min(int(max_pan_speed * duration), int(max_x * horizontal_travel_ratio)))
        travel_y = min(max_y, min(int(max_pan_speed * duration), int(max_y * vertical_travel_ratio)))
        can_pan = (travel_y > max(threshold, min_pan_travel)) if is_vertical else (
            travel_x > max(threshold, min_pan_travel)
        )

        mode = direction = None
        for attempt in range(len(palette)):
            cand_mode, cand_dir = palette[(palette_idx + attempt) % len(palette)]
            if cand_mode == "pan" and not can_pan:
                continue
            if (cand_mode, cand_dir) == last_sig:
                continue
            mode, direction = cand_mode, cand_dir
            palette_idx = (palette_idx + attempt + 1) % len(palette)
            break

        if mode is None:
            mode = "zoom"
            direction = "out" if (last_sig and last_sig[1] == "in") else "in"
            palette_idx = (palette_idx + 1) % len(palette)

        last_sig = (mode, direction)
        plans.append(
            {
                "mode": mode,
                "direction": direction,
                "is_vertical": is_vertical,
                "pan_axis": pan_axis,
            }
        )

    return plans


def create_kenburns_clip(
    image_path: str | Path,
    duration: float,
    output_path: str | Path,
    fps: int = 30,
    target_w: int = 1920,
    target_h: int = 1080,
    threshold: int = 40,
    min_pan_travel: int = 140,
    prefer_pan_when_possible: bool = True,
    zoom_min: float = 1.03,
    zoom_max: float = 1.08,
    zoom_rate_per_sec: float = 0.055,
    zoom_anchor_vertical_y: float = 0.18,
    zoom_anchor_horizontal_y: float = 0.50,
    max_pan_speed: int = 200,
    vertical_travel_ratio: float = 0.80,
    horizontal_travel_ratio: float = 0.55,
    top_bias: float = 0.00,
    motion_scale: float = 1.4,
    use_nvenc: bool = True,
    motion_mode: str | None = None,
    motion_direction: str | None = None,
):
    total_frames = max(2, int(round(duration * fps)))
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    scale_factor = max(target_w / img_w, target_h / img_h)
    scaled_w = int(round(img_w * scale_factor))
    scaled_h = int(round(img_h * scale_factor))
    scaled_w = scaled_w if scaled_w % 2 == 0 else scaled_w + 1
    scaled_h = scaled_h if scaled_h % 2 == 0 else scaled_h + 1

    max_x = max(0, scaled_w - target_w)
    max_y = max(0, scaled_h - target_h)

    denom = total_frames - 1
    ease_pan = f"(0.5-0.5*cos(PI*n/{denom}))"
    ease_zoom = f"(0.5-0.5*cos(PI*on/{denom}))"

    target_ar = target_w / target_h
    img_ar = img_w / img_h
    is_vertical = img_ar < target_ar

    u = _stable_unit(str(image_path))
    zoom_direction = "out" if u < 0.5 else "in"
    pan_reverse = u > 0.66

    if motion_direction is not None:
        if motion_direction in ("in", "out"):
            zoom_direction = motion_direction
        elif motion_direction in ("forward", "reverse"):
            pan_reverse = motion_direction == "reverse"

    dyn_cap = 1.0 + zoom_rate_per_sec * duration
    zmax = min(zoom_max, dyn_cap)
    z_target = max(zoom_min, min(zoom_min + (zmax - zoom_min) * (0.25 + 0.50 * u), zmax))

    travel_x = min(max_x, min(int(max_pan_speed * duration), int(max_x * horizontal_travel_ratio)))
    travel_y = min(max_y, min(int(max_pan_speed * duration), int(max_y * vertical_travel_ratio)))
    can_pan = (travel_y > max(threshold, min_pan_travel)) if is_vertical else (
        travel_x > max(threshold, min_pan_travel)
    )

    ms = max(1.0, float(motion_scale))
    mw, mh = int(round(target_w * ms)), int(round(target_h * ms))
    scaled_w2, scaled_h2 = int(round(scaled_w * ms)), int(round(scaled_h * ms))
    max_x2, max_y2 = max(0, scaled_w2 - mw), max(0, scaled_h2 - mh)
    travel_x2 = int(round(min(max_x2, travel_x * ms)))
    travel_y2 = int(round(min(max_y2, travel_y * ms)))

    def build_pan_vf():
        if is_vertical and travel_y2 < 220:
            return None
        if not is_vertical and travel_x2 < 220:
            return None

        if is_vertical:
            start_y = int(round(max_y2 * top_bias))
            end_y = min(start_y + travel_y2, max_y2)
            if pan_reverse:
                start_y, end_y = end_y, start_y
            x_expr = f"{max_x2 // 2}"
            y_expr = f"trunc({start_y}+({end_y}-{start_y})*{ease_pan})"
        else:
            start_x, end_x = 0, travel_x2
            if pan_reverse:
                start_x, end_x = end_x, start_x
            x_expr = f"trunc({start_x}+({end_x}-{start_x})*{ease_pan})"
            y_expr = f"{max_y2 // 2}"

        return (
            f"scale={scaled_w2}:{scaled_h2}:flags=lanczos,"
            f"crop={mw}:{mh}:x='{x_expr}':y='{y_expr}',"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            f"setsar=1,setdar=16/9,format=yuv420p"
        )

    def build_zoom_vf():
        if zoom_direction == "out":
            z_expr = f"{z_target}-({z_target}-1.0)*{ease_zoom}"
        else:
            z_expr = f"1.0+({z_target}-1.0)*{ease_zoom}"

        ax = 0.5
        ay = zoom_anchor_vertical_y if is_vertical else zoom_anchor_horizontal_y
        x0 = f"max(0,min(({ax})*iw-ow/2,iw-ow))"
        y0 = f"max(0,min(({ay})*ih-oh/2,ih-oh))"

        return (
            f"scale={scaled_w2}:{scaled_h2}:flags=lanczos,"
            f"zoompan=z='{z_expr}':x='if(eq(on,1),{x0},px)':y='if(eq(on,1),{y0},py)'"
            f":d=1:s={mw}x{mh}:fps={fps},"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            f"setsar=1,setdar=16/9,format=yuv420p"
        )

    if motion_mode == "pan":
        vf = build_pan_vf() or build_zoom_vf()
    elif motion_mode == "zoom":
        vf = build_zoom_vf()
    else:
        vf = (build_pan_vf() or build_zoom_vf()) if (prefer_pan_when_possible and can_pan) else build_zoom_vf()

    vcodec = ["-c:v", "h264_nvenc", "-preset", "p1"] if use_nvenc \
             else ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]

    run_ffmpeg([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(fps), "-t", str(duration), "-i", str(image_path),
        "-vf", vf, "-frames:v", str(total_frames),
        *vcodec, "-pix_fmt", "yuv420p","-fps_mode", "cfr",
        output_path,
    ])


def generate_all_clips(
    image_files: Sequence[str | Path],
    durations: Sequence[float],
    temp_dir: str = "temp_clips",
    fps: int = 30,
    target_w: int = 1920,
    target_h: int = 1080,
    use_nvenc: bool = True,
) -> List[str]:
    Path(temp_dir).mkdir(exist_ok=True)
    plans = plan_kenburns_sequence(image_files, durations, target_w=target_w, target_h=target_h)

    outputs = []
    for i, (img, dur, plan) in enumerate(zip(image_files, durations, plans)):
        out = Path(temp_dir) / f"clip_{i}.mp4"
        create_kenburns_clip(
            image_path=img,
            duration=dur,
            output_path=out,
            fps=fps,
            target_w=target_w,
            target_h=target_h,
            use_nvenc=use_nvenc,
            motion_mode=plan["mode"],
            motion_direction=plan["direction"],
        )
        outputs.append(str(out))
    return outputs

def distribute_durations_exact(seconds_for_chunk: Sequence[float], n_clips: int, fade: float) -> List[float]:
    target_extra = fade * (n_clips - 1)
    extra_for_each = target_extra / n_clips
    return [s + extra_for_each for s in seconds_for_chunk]


def get_duration(path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def concatenate_clips(clips: Sequence[str | Path], output_path: str | Path, fade: float = 0.45):
    durations_local = [get_duration(c) for c in clips]

    cmd = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", str(c)]

    filter_parts = [
        "[0:v]fps=30,scale=1920:1080:flags=lanczos,setsar=1,setdar=16/9,format=yuv420p[v0]"
    ]
    for i in range(1, len(clips)):
        filter_parts.append(
            f"[{i}:v]fps=30,scale=1920:1080:flags=lanczos,setsar=1,setdar=16/9,format=yuv420p[v{i}src]"
        )

    cumulative = durations_local[0]
    last = "v0"
    for i in range(1, len(clips)):
        offset = cumulative - fade
        out = f"vx{i}"
        filter_parts.append(
            f"[{last}][v{i}src]xfade=transition=fade:duration={fade}:offset={offset}[{out}]"
        )
        cumulative += durations_local[i] - fade
        last = out

    filter_complex = ";".join(filter_parts)

    cmd += [
            "-safe", "0",
            "-filter_complex", filter_complex,
            "-map", f"[{last}]",
            "-c:v", "h264_nvenc",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        
            str(output_path)
        ]

    run_ffmpeg(cmd)


def add_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path):
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",str(video_path),
        "-i",str(audio_path),
        "-c","copy",
        "-c:a","aac",
        str(output_path),
    ])


def add_subtitles(video_path: str | Path, srt_path: str | Path, output_path: str | Path):
    video_path = str(video_path)
    output_path = str(output_path)

    srt_path = Path(srt_path).as_posix()  
    srt_path = srt_path.replace("\\", "/")
    srt_path = srt_path.replace(":", r"\:")

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+genpts",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-c:a", "copy",
        output_path
    ]

    run_ffmpeg(cmd)


def cleanup_files(output_dir: str = "tts_Outputs", temp_clips_dir: str = "temp_clips", temp_frames_dir: str = "temp_frames"):
    for temp_dir in [temp_clips_dir, temp_frames_dir]:
        p = Path(temp_dir)
        if p.exists():
            shutil.rmtree(p)

    out = Path(output_dir)
    if out.exists():
        for f in out.iterdir():
            if f.name.startswith(("combined", "with_audio", "output_subtitles", "output_", "temp_output_")) or f.suffix == ".wav":
                try:
                    f.unlink()
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
def build_final_video(
    entity_name: str,
    is_landmark: bool = False,
    config: VideoPipelineConfig | None = None,
) -> str:
    config = config or VideoPipelineConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    stage_times = {}
    total_start = time.time()

    # 1) Load script
    script = get_script_by_name(entity_name, is_landmark=is_landmark)
    if not script:
        raise ValueError(f"No script found for entity: {entity_name}")

    # 2) Split script
    paragraphs, sentence_groups = split_script_into_paragraph_sentences(script)

    # TTS Stage
    tts_start = time.time()
    # 3) Generate sentence-level TTS
    asyncio.run(
        generate_tts_audio(
            sentence_groups=sentence_groups,
            output_dir=config.output_dir,
            voice=config.voice,
            rate=config.rate,
            trim_top_db=config.trim_top_db,
        )
    )
    # 4) Combine all sentence audios
    final_audio_path = combine_audio_files(config.output_dir, "Final audio.wav")

    # 5) Compute durations
    _, sentence_durations, images_needed = compute_audio_durations(
        sentence_groups = sentence_groups,
        output_dir=config.output_dir,
        seconds_per_image=config.seconds_per_image,
        delete_sentence_files=True,
    )
    stage_times["TTS"] = time.time() - tts_start

    # Retrieval Stage
    retrieval_start = time.time()
    # 6) Build retrieval chunks
    image_text_chunks, seconds_for_chunk = create_image_chunks(
        sentence_groups=sentence_groups,
        images_needed=images_needed,
        sentence_durations=sentence_durations,
    )

    # 7) Retrieve images semantically
    _, fetched_image_paths, _, _ = retrieve_images_semantic(
        name=entity_name,
        image_text_chunks=image_text_chunks,
        is_landmark=is_landmark,
        image_sim_weight=config.image_sim_weight,
        desc_sim_weight=config.desc_sim_weight,
    )

    # 8) Download images
    image_files = download_images_from_r2(remote_paths = fetched_image_paths, local_frames_dir=config.temp_frames_dir)
    
    # 9) Normalize images
    image_files = normalize_images_to_jpeg(image_files)

    stage_times["Retrieval"] = time.time() - retrieval_start
    
    # Subtitles Stage
    subtitles_start = time.time()
    
    # 10) Subtitles 
    srt_path = generate_srt(
        paragraphs,
        sentence_durations,
        output_path=str(output_dir / "output_subtitles.srt"),
        max_chars_per_line=config.max_chars_per_line,
        max_lines=config.max_lines,
        min_duration=config.min_subtitle_duration,
    )
    stage_times["Subtitles"] = time.time() - subtitles_start

    # Motion Stage
    motion_start = time.time()

    # 11) Per-image durations with fade compensation
    seconds = distribute_durations_exact(seconds_for_chunk, n_clips=len(image_files), fade=config.fade)
    
    # 12) Generate Ken Burns clips
    clips = generate_all_clips(
        image_files,
        seconds,
        temp_dir=config.temp_clips_dir,
        fps=config.fps,
        target_w=config.target_w,
        target_h=config.target_h,
        use_nvenc=config.use_nvenc,
    )

    stage_times["Motion"] = time.time() - motion_start

    # Rendering Stage
    rendering_start = time.time()

    # 13) Concatenate clips
    combined_video = output_dir / "combined.mp4"
    concatenate_clips(clips, combined_video, fade=config.fade)

    # 14) Add audio
    with_audio = output_dir / "with_audio.mp4"
    add_audio(combined_video, final_audio_path, with_audio)

    # 15) Add subtitles
    final_output = output_dir / f"{entity_name.replace(' ', '_')}_final_video.mp4"
    add_subtitles(with_audio, srt_path, final_output)

    stage_times["Rendering"] = time.time() - rendering_start

    # Total time
    stage_times["Total"] = time.time() - total_start

    # 16) Cleanup intermediate files
    if config.cleanup_intermediate:
        cleanup_files(
            output_dir=config.output_dir,
            temp_clips_dir=config.temp_clips_dir,
            temp_frames_dir=config.temp_frames_dir,
        )

    print(f"Done: {final_output}")
    print("\nStage Times:")
    print(f"TTS       : {format_stage_time(stage_times['TTS'])}")
    print(f"Retrieval : {format_stage_time(stage_times['Retrieval'])}")
    print(f"Motion    : {format_stage_time(stage_times['Motion'])}")
    print(f"Subtitles : {format_stage_time(stage_times['Subtitles'])}")
    print(f"Rendering : {format_stage_time(stage_times['Rendering'])}")
    print(f"Total     : {format_stage_time(stage_times['Total'])}")

    return str(final_output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate final entity video.")
    parser.add_argument("entity_name", type=str, help="Entity name exactly as stored in the DB.")
    parser.add_argument("--landmark", action="store_true", help="Use landmarks tables instead of pharaohs tables.")
    args = parser.parse_args()

    build_final_video(entity_name=args.entity_name, is_landmark=args.landmark)