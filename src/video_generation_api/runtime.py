from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pillow_avif
import soundfile as sf
from boto3.session import Session as Boto3Session
from dotenv import load_dotenv
from edge_tts import Communicate
from librosa import effects, load
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from scipy.io import wavfile
from sqlalchemy import text
from sqlalchemy.orm import Session
from torch import cuda, no_grad

from src.db.session import engine

load_dotenv()


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


class VideoGenerationRuntime:
    CLIP_MODEL_NAME = "ViT-H-14"
    CLIP_PRETRAINED = "laion2b_s32b_b79k"

    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_device = "cuda" if cuda.is_available() else "cpu"

    def get_clip_model(self):
        if self.clip_model is None or self.clip_tokenizer is None:
            model, _, _ = create_model_and_transforms(
                self.CLIP_MODEL_NAME,
                pretrained=self.CLIP_PRETRAINED,
            )
            model = model.to(self.clip_device)
            model.eval()
            tokenizer = get_tokenizer(self.CLIP_MODEL_NAME)

            self.clip_model = model
            self.clip_tokenizer = tokenizer

        return self.clip_model, self.clip_tokenizer, self.clip_device

    def get_script_by_name(self, name: str, is_landmark: bool = False) -> str | None:
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

    def split_into_sentences(self, text_value: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_value.strip()) if s.strip()]

    def split_script_into_paragraph_sentences(self, script: str) -> tuple[list[str], list[list[str]]]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script.strip()) if p.strip()]
        sentence_groups = [self.split_into_sentences(paragraph) for paragraph in paragraphs]
        return paragraphs, sentence_groups

    async def edge_tts_save(self, text_value: str, out_path: str | Path, voice: str, rate: str) -> None:
        communicate = Communicate(text=text_value, voice=voice, rate=rate)
        await communicate.save(str(out_path))

    def trim_trailing_silence(self, audio_path: str | Path, top_db: int = 20) -> None:
        waveform, sample_rate = load(str(audio_path), sr=None)
        trimmed_waveform, _ = effects.trim(waveform, top_db=top_db)
        sf.write(str(audio_path), trimmed_waveform, sample_rate)

    async def generate_tts_audio(
        self,
        sentence_groups: Sequence[Sequence[str]],
        output_dir: str,
        voice: str,
        rate: str,
        trim_top_db: int,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        sentence_index = 0

        for paragraph in sentence_groups:
            for sentence in paragraph:
                out_path = Path(output_dir) / f"output_{sentence_index}.wav"
                await self.edge_tts_save(sentence, out_path, voice=voice, rate=rate)
                self.trim_trailing_silence(out_path, top_db=trim_top_db)
                sentence_index += 1

    def combine_audio_files(self, output_dir: str, final_name: str = "Final audio.wav") -> str:
        wav_files = sorted(
            [file_name for file_name in os.listdir(output_dir) if file_name.endswith(".wav") and file_name.startswith("output_")],
            key=lambda file_name: int(re.search(r"output_(\d+)\.wav", file_name).group(1)),
        )

        audio_chunks = []
        sample_rate = None

        for file_name in wav_files:
            file_path = Path(output_dir) / file_name
            data, current_sample_rate = sf.read(file_path)
            if sample_rate is None:
                sample_rate = current_sample_rate
            elif current_sample_rate != sample_rate:
                raise ValueError("Sample rates do not match.")
            audio_chunks.append(data)

        if not audio_chunks:
            raise ValueError("No sentence wav files were found to combine.")

        combined_audio = np.concatenate(audio_chunks, axis=0)
        final_path = Path(output_dir) / final_name
        sf.write(final_path, combined_audio, sample_rate)
        return str(final_path)

    def compute_audio_durations(
        self,
        sentence_groups: Sequence[Sequence[str]],
        output_dir: str,
        seconds_per_image: float,
        delete_sentence_files: bool = True,
    ) -> tuple[list[float], list[float], list[int]]:
        paragraph_durations: list[float] = []
        sentence_durations: list[float] = []
        images_needed: list[int] = []
        sentence_index = 0

        for paragraph in sentence_groups:
            duration_seconds = 0.0
            for _sentence in paragraph:
                file_path = Path(output_dir) / f"output_{sentence_index}.wav"
                sample_rate, data = wavfile.read(file_path)
                sentence_duration = len(data) / float(sample_rate)
                duration_seconds += sentence_duration
                sentence_durations.append(sentence_duration)
                if delete_sentence_files and file_path.exists():
                    file_path.unlink()
                sentence_index += 1

            paragraph_durations.append(duration_seconds)
            images_needed.append(max(1, int(duration_seconds / seconds_per_image)))

        return paragraph_durations, sentence_durations, images_needed

    def create_image_chunks(
        self,
        sentence_groups: Sequence[Sequence[str]],
        images_needed: Sequence[int],
        sentence_durations: Sequence[float],
    ) -> tuple[list[list[str]], list[float]]:
        sentence_start = 0
        image_text_chunks: list[list[str]] = []
        seconds_for_chunk: list[float] = []

        for paragraph_index, paragraph_sentences in enumerate(sentence_groups):
            images_for_paragraph = min(images_needed[paragraph_index], len(paragraph_sentences))
            total_sentences = len(paragraph_sentences)
            base = total_sentences // images_for_paragraph
            remainder = total_sentences % images_for_paragraph

            groups = []
            start = 0
            for image_index in range(images_for_paragraph):
                extra = 1 if image_index < remainder else 0
                end = start + base + extra
                chunk_duration = sum(sentence_durations[sentence_start + start : sentence_start + end])
                seconds_for_chunk.append(chunk_duration)
                groups.append(" ".join(paragraph_sentences[start:end]))
                start = end

            sentence_start += total_sentences
            image_text_chunks.append(groups)

        return image_text_chunks, seconds_for_chunk

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def fetch_candidate_images(self, name: str, is_landmark: bool = False):
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
        self,
        name: str,
        image_text_chunks: Sequence[Sequence[str]],
        is_landmark: bool = False,
        image_sim_weight: float = 0.3,
        desc_sim_weight: float = 0.7,
    ) -> tuple[list[int], list[str], float, float]:
        model, tokenizer, device = self.get_clip_model()
        images_data = self.fetch_candidate_images(name, is_landmark=is_landmark)

        if not images_data:
            raise ValueError(f"No candidate images found in the database for '{name}'.")

        fetched_image_ids: list[int] = []
        fetched_image_paths: list[str] = []
        repeated_image_ids: list[int] = []
        best_scores: list[float] = []
        trials: list[int] = []
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
            if (not is_landmark) and  image_description:
                tokens = tokenizer([image_description]).to(device)
                with no_grad():
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
                with no_grad():
                    embedding = model.encode_text(text_tokens)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                scene_embedding = embedding.cpu().numpy()[0]

                ranked = []
                for image_data in processed_images:
                    image_similarity = self.cosine(scene_embedding, image_data["img_emb"])
                    if is_landmark or image_data["desc_emb"] is None:
                        score = image_similarity
                    else:
                        desc_similarity = self.cosine(scene_embedding, image_data["desc_emb"])
                        score = image_sim_weight * image_similarity + desc_sim_weight * desc_similarity
                    ranked.append((score, image_data["id"], image_data["path"]))

                ranked.sort(reverse=True, key=lambda item: item[0])
                ranked_backup = ranked.copy()
                best_score, best_id, best_path = ranked[0]
                attempts = 0

                while best_id in fetched_image_ids:
                    ranked.pop(0)
                    attempts += 1
                    if not ranked:
                        best_score, best_id, best_path = ranked_backup[0]
                        while best_id in repeated_image_ids:
                            ranked_backup.pop(0)
                            attempts += 1
                            if not ranked_backup:
                                raise RuntimeError("No more images available for retrieval.")
                            best_score, best_id, best_path = ranked_backup[0]
                        repeated_image_ids.append(best_id)
                        break
                    best_score, best_id, best_path = ranked[0]

                fetched_image_ids.append(best_id)
                fetched_image_paths.append(best_path)
                best_scores.append(best_score)
                trials.append(attempts)

        average_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
        average_trials = sum(trials) / len(trials) if trials else 0.0
        return fetched_image_ids, fetched_image_paths, average_score, average_trials

    def build_r2_client(self):
        load_dotenv()
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY")
        secret_key = os.getenv("R2_SECRET_KEY")

        if not all([account_id, access_key, secret_key]):
            raise ValueError("Missing one or more Cloudflare R2 environment variables.")

        session = Boto3Session()
        return session.client(
            "s3",
            region_name="auto",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def download_images_from_r2(self, remote_paths: Sequence[str], local_frames_dir: str) -> list[str]:
        load_dotenv()
        bucket_name = os.getenv("R2_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("R2_BUCKET_NAME is not set.")

        client = self.build_r2_client()
        local_dir = Path(local_frames_dir)
        local_dir.mkdir(exist_ok=True)

        def download_image(index_and_key: tuple[int, str]) -> str:
            index, image_key = index_and_key
            suffix = Path(image_key).suffix.lower() or ".jpg"
            local_file = local_dir / f"{index:04d}{suffix}"
            client.download_file(bucket_name, image_key, str(local_file))
            return str(local_file)

        with ThreadPoolExecutor(max_workers=8) as executor:
            ordered_local_paths = list(executor.map(download_image, enumerate(remote_paths)))

        return ordered_local_paths

    def normalize_images_to_jpeg(self, image_files: Sequence[str | Path]) -> list[str]:
        normalized_paths: list[str] = []
        for image_path in image_files:
            image_file = Path(image_path)
            jpg_path = image_file.with_suffix(".jpg")
            with Image.open(image_file) as image:
                image = image.convert("RGB")
                image.save(jpg_path, "JPEG", quality=95)
            normalized_paths.append(str(jpg_path))
        return sorted(normalized_paths)

    def normalize_text(self, text_value: str) -> str:
        text_value = unicodedata.normalize("NFKC", text_value)
        replacements = {
            "???": "'",
            "???": "'",
            "???": ",",
            "???": "'",
            "???": '"',
            "??": '"',
            "???": '"',
            "???": "-",
            "???": "-",
            "???": "-",
            "???": "...",
            " ": " ",
            "​": "",
            "‌": "",
            "‍": "",
            "﻿": "",
        }
        for bad, good in replacements.items():
            text_value = text_value.replace(bad, good)

        return "".join(char for char in text_value if unicodedata.category(char)[0] != "C")

    def normalize_text_for_subtitles(self, text_value: str) -> str:
        text_value = unicodedata.normalize("NFKC", text_value)
        replacements = {
            "\u00e2\u20ac\u2122": "'",
            "\u00e2\u20ac\u02dc": "'",
            "\u00e2\u20ac\u0161": ",",
            "\u00e2\u20ac\u203a": "'",
            "\u00e2\u20ac\u0153": '"',
            "\u00e2\u20ac\u009d": '"',
            "\u00e2\u20ac\u017e": '"',
            "\u00e2\u20ac\u201d": "-",
            "\u00e2\u20ac\u201c": "-",
            "\u00e2\u20ac\u2022": "-",
            "\u00e2\u20ac\u00a6": "...",
            "\u00A0": " ",
            "\u200B": "",
            "\u200C": "",
            "\u200D": "",
            "\uFEFF": "",
        }
        for bad, good in replacements.items():
            text_value = text_value.replace(bad, good)

        return "".join(char for char in text_value if unicodedata.category(char)[0] != "C")

    def format_timestamp(self, seconds: float) -> str:
        delta = timedelta(seconds=seconds)
        total_seconds = int(delta.total_seconds())
        millis = int((seconds - total_seconds) * 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def format_stage_time(self, seconds: float) -> str:
        return f"{seconds:.2f}s ({seconds / 60:.2f} min)"

    def split_long_text(self, text_value: str, max_chars: int = 35, max_lines: int = 2) -> list[str]:
        words = text_value.split()
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

        return ["\n".join(lines[index : index + max_lines]) for index in range(0, len(lines), max_lines)]

    def generate_srt(
        self,
        paragraphs: Sequence[str],
        sentence_durations: Sequence[float],
        output_path: str,
        max_chars_per_line: int,
        max_lines: int,
        min_duration: float,
    ) -> str:
        current_time = 0.0
        subtitle_index = 1
        srt_blocks = []
        duration_index = 0

        for paragraph in paragraphs:
            normalized_paragraph = self.normalize_text_for_subtitles(paragraph)
            sentences = self.split_into_sentences(normalized_paragraph)

            for sentence in sentences:
                sentence_duration = sentence_durations[duration_index]
                duration_index += 1

                chunks = self.split_long_text(
                    sentence,
                    max_chars=max_chars_per_line,
                    max_lines=max_lines,
                )
                total_chars = sum(len(chunk.replace("\n", "")) for chunk in chunks)

                for chunk in chunks:
                    chunk_char_count = len(chunk.replace("\n", ""))
                    chunk_duration = max(min_duration, (chunk_char_count / total_chars) * sentence_duration)
                    start_time = current_time
                    end_time = current_time + chunk_duration

                    srt_blocks.append(
                        f"{subtitle_index}\n"
                        f"{self.format_timestamp(start_time)} --> {self.format_timestamp(end_time)}\n"
                        f"{chunk}\n\n"
                    )
                    current_time = end_time
                    subtitle_index += 1

        with open(output_path, "w", encoding="utf-8-sig") as file:
            file.writelines(srt_blocks)
        return output_path

    def run_ffmpeg(self, cmd: Sequence[str]) -> None:
        normalized_cmd = [str(item) for item in cmd]
        result = subprocess.run(normalized_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_output = result.stderr.strip() or "FFmpeg failed."
            raise RuntimeError(error_output)

    def stable_int(self, seed: str) -> int:
        return int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)

    def stable_unit(self, seed: str) -> float:
        return (self.stable_int(seed) % 10_000) / 10_000.0

    def plan_kenburns_sequence(
        self,
        image_files: Sequence[str | Path],
        durations: Sequence[float],
        target_w: int = 1920,
        target_h: int = 1080,
        threshold: int = 40,
        min_pan_travel: int = 140,
        max_pan_speed: int = 120,
        vertical_travel_ratio: float = 0.80,
        horizontal_travel_ratio: float = 0.55,
    ) -> list[dict[str, object]]:
        palette = [("zoom", "in"), ("pan", "forward"), ("zoom", "out"), ("pan", "reverse")]
        plans = []
        palette_index = 0
        last_signature = None
        target_aspect_ratio = target_w / target_h

        for image_path, duration in zip(image_files, durations):
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            scale_factor = max(target_w / image_width, target_h / image_height)
            scaled_width = int(round(image_width * scale_factor))
            scaled_height = int(round(image_height * scale_factor))
            scaled_width = scaled_width if scaled_width % 2 == 0 else scaled_width + 1
            scaled_height = scaled_height if scaled_height % 2 == 0 else scaled_height + 1

            max_x = max(0, scaled_width - target_w)
            max_y = max(0, scaled_height - target_h)
            image_aspect_ratio = image_width / image_height
            is_vertical = image_aspect_ratio < target_aspect_ratio
            pan_axis = "vertical" if is_vertical else "horizontal"

            travel_x = min(max_x, min(int(max_pan_speed * duration), int(max_x * horizontal_travel_ratio)))
            travel_y = min(max_y, min(int(max_pan_speed * duration), int(max_y * vertical_travel_ratio)))
            can_pan = (travel_y > max(threshold, min_pan_travel)) if is_vertical else (
                travel_x > max(threshold, min_pan_travel)
            )

            mode = direction = None
            for attempt in range(len(palette)):
                candidate_mode, candidate_direction = palette[(palette_index + attempt) % len(palette)]
                if candidate_mode == "pan" and not can_pan:
                    continue
                if (candidate_mode, candidate_direction) == last_signature:
                    continue
                mode, direction = candidate_mode, candidate_direction
                palette_index = (palette_index + attempt + 1) % len(palette)
                break

            if mode is None:
                mode = "zoom"
                direction = "out" if (last_signature and last_signature[1] == "in") else "in"
                palette_index = (palette_index + 1) % len(palette)

            last_signature = (mode, direction)
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
        self,
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
    ) -> None:
        total_frames = max(2, int(round(duration * fps)))
        image_path = Path(image_path)
        output_path = Path(output_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        scale_factor = max(target_w / image_width, target_h / image_height)
        scaled_width = int(round(image_width * scale_factor))
        scaled_height = int(round(image_height * scale_factor))
        scaled_width = scaled_width if scaled_width % 2 == 0 else scaled_width + 1
        scaled_height = scaled_height if scaled_height % 2 == 0 else scaled_height + 1

        max_x = max(0, scaled_width - target_w)
        max_y = max(0, scaled_height - target_h)
        denom = total_frames - 1
        ease_pan = f"(0.5-0.5*cos(PI*n/{denom}))"
        ease_zoom = f"(0.5-0.5*cos(PI*on/{denom}))"

        target_aspect_ratio = target_w / target_h
        image_aspect_ratio = image_width / image_height
        is_vertical = image_aspect_ratio < target_aspect_ratio

        stable_value = self.stable_unit(str(image_path))
        zoom_direction = "out" if stable_value < 0.5 else "in"
        pan_reverse = stable_value > 0.66

        if motion_direction is not None:
            if motion_direction in ("in", "out"):
                zoom_direction = motion_direction
            elif motion_direction in ("forward", "reverse"):
                pan_reverse = motion_direction == "reverse"

        dynamic_cap = 1.0 + zoom_rate_per_sec * duration
        zmax = min(zoom_max, dynamic_cap)
        z_target = max(zoom_min, min(zoom_min + (zmax - zoom_min) * (0.25 + 0.50 * stable_value), zmax))

        travel_x = min(max_x, min(int(max_pan_speed * duration), int(max_x * horizontal_travel_ratio)))
        travel_y = min(max_y, min(int(max_pan_speed * duration), int(max_y * vertical_travel_ratio)))
        can_pan = (travel_y > max(threshold, min_pan_travel)) if is_vertical else (
            travel_x > max(threshold, min_pan_travel)
        )

        motion_multiplier = max(1.0, float(motion_scale))
        motion_width = int(round(target_w * motion_multiplier))
        motion_height = int(round(target_h * motion_multiplier))
        scaled_width_2 = int(round(scaled_width * motion_multiplier))
        scaled_height_2 = int(round(scaled_height * motion_multiplier))
        max_x_2 = max(0, scaled_width_2 - motion_width)
        max_y_2 = max(0, scaled_height_2 - motion_height)
        travel_x_2 = int(round(min(max_x_2, travel_x * motion_multiplier)))
        travel_y_2 = int(round(min(max_y_2, travel_y * motion_multiplier)))

        def build_pan_vf() -> str | None:
            if is_vertical and travel_y_2 < 220:
                return None
            if not is_vertical and travel_x_2 < 220:
                return None

            if is_vertical:
                start_y = int(round(max_y_2 * top_bias))
                end_y = min(start_y + travel_y_2, max_y_2)
                if pan_reverse:
                    start_y, end_y = end_y, start_y
                x_expr = f"{max_x_2 // 2}"
                y_expr = f"trunc({start_y}+({end_y}-{start_y})*{ease_pan})"
            else:
                start_x, end_x = 0, travel_x_2
                if pan_reverse:
                    start_x, end_x = end_x, start_x
                x_expr = f"trunc({start_x}+({end_x}-{start_x})*{ease_pan})"
                y_expr = f"{max_y_2 // 2}"

            return (
                f"scale={scaled_width_2}:{scaled_height_2}:flags=lanczos,"
                f"crop={motion_width}:{motion_height}:x='{x_expr}':y='{y_expr}',"
                f"scale={target_w}:{target_h}:flags=lanczos,"
                f"setsar=1,setdar=16/9,format=yuv420p"
            )

        def build_zoom_vf() -> str:
            if zoom_direction == "out":
                z_expr = f"{z_target}-({z_target}-1.0)*{ease_zoom}"
            else:
                z_expr = f"1.0+({z_target}-1.0)*{ease_zoom}"

            anchor_x = 0.5
            anchor_y = zoom_anchor_vertical_y if is_vertical else zoom_anchor_horizontal_y
            x0 = f"max(0,min(({anchor_x})*iw-ow/2,iw-ow))"
            y0 = f"max(0,min(({anchor_y})*ih-oh/2,ih-oh))"

            return (
                f"scale={scaled_width_2}:{scaled_height_2}:flags=lanczos,"
                f"zoompan=z='{z_expr}':x='if(eq(on,1),{x0},px)':y='if(eq(on,1),{y0},py)'"
                f":d=1:s={motion_width}x{motion_height}:fps={fps},"
                f"scale={target_w}:{target_h}:flags=lanczos,"
                f"setsar=1,setdar=16/9,format=yuv420p"
            )

        if motion_mode == "pan":
            video_filter = build_pan_vf() or build_zoom_vf()
        elif motion_mode == "zoom":
            video_filter = build_zoom_vf()
        else:
            video_filter = (build_pan_vf() or build_zoom_vf()) if (prefer_pan_when_possible and can_pan) else build_zoom_vf()

        video_codec = ["-c:v", "h264_nvenc", "-preset", "p1"] if use_nvenc else [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
        ]

        self.run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                str(duration),
                "-i",
                str(image_path),
                "-vf",
                video_filter,
                "-frames:v",
                str(total_frames),
                *video_codec,
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "cfr",
                str(output_path),
            ]
        )

    def generate_all_clips(
        self,
        image_files: Sequence[str | Path],
        durations: Sequence[float],
        temp_dir: str,
        fps: int,
        target_w: int,
        target_h: int,
        use_nvenc: bool,
    ) -> list[str]:
        Path(temp_dir).mkdir(exist_ok=True)
        plans = self.plan_kenburns_sequence(
            image_files,
            durations,
            target_w=target_w,
            target_h=target_h,
        )

        outputs = []
        for index, (image_path, duration, plan) in enumerate(zip(image_files, durations, plans)):
            output_path = Path(temp_dir) / f"clip_{index}.mp4"
            self.create_kenburns_clip(
                image_path=image_path,
                duration=duration,
                output_path=output_path,
                fps=fps,
                target_w=target_w,
                target_h=target_h,
                use_nvenc=use_nvenc,
                motion_mode=plan["mode"],
                motion_direction=plan["direction"],
            )
            outputs.append(str(output_path))
        return outputs

    def distribute_durations_exact(self, seconds_for_chunk: Sequence[float], n_clips: int, fade: float) -> list[float]:
        target_extra = fade * (n_clips - 1)
        extra_for_each = target_extra / n_clips
        return [seconds + extra_for_each for seconds in seconds_for_chunk]

    def get_duration(self, path: str | Path) -> float:
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

    def concatenate_clips(
        self,
        clips: Sequence[str | Path],
        output_path: str | Path,
        fade: float = 0.45,
        use_nvenc: bool = True,
    ) -> None:
        durations_local = [self.get_duration(clip) for clip in clips]

        cmd = ["ffmpeg", "-y"]
        for clip in clips:
            cmd += ["-i", str(clip)]

        filter_parts = [
            "[0:v]fps=30,scale=1920:1080:flags=lanczos,setsar=1,setdar=16/9,format=yuv420p[v0]"
        ]
        for index in range(1, len(clips)):
            filter_parts.append(
                f"[{index}:v]fps=30,scale=1920:1080:flags=lanczos,setsar=1,setdar=16/9,format=yuv420p[v{index}src]"
            )

        cumulative = durations_local[0]
        last = "v0"
        for index in range(1, len(clips)):
            offset = cumulative - fade
            out = f"vx{index}"
            filter_parts.append(
                f"[{last}][v{index}src]xfade=transition=fade:duration={fade}:offset={offset}[{out}]"
            )
            cumulative += durations_local[index] - fade
            last = out

        video_codec = ["-c:v", "h264_nvenc", "-preset", "medium"] if use_nvenc else [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
        ]

        cmd += [
            "-safe",
            "0",
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{last}]",
            *video_codec,
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        self.run_ffmpeg(cmd)

    def add_audio(self, video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> None:
        self.run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c",
                "copy",
                "-c:a",
                "aac",
                str(output_path),
            ]
        )

    def add_subtitles(
        self,
        video_path: str | Path,
        srt_path: str | Path,
        output_path: str | Path,
        use_nvenc: bool = True,
    ) -> None:
        normalized_srt_path = Path(srt_path).as_posix().replace("\\", "/").replace(":", r"\:")
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p1"] if use_nvenc else [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
        ]

        self.run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(video_path),
                "-vf",
                f"subtitles={normalized_srt_path}",
                *video_codec,
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "cfr",
                "-c:a",
                "copy",
                str(output_path),
            ]
        )

    def cleanup_files(self, output_dir: str, temp_clips_dir: str, temp_frames_dir: str) -> None:
        for temp_dir in [temp_clips_dir, temp_frames_dir]:
            path = Path(temp_dir)
            if path.exists():
                shutil.rmtree(path)

        output_path = Path(output_dir)
        if output_path.exists():
            for file_path in output_path.iterdir():
                if file_path.name.startswith(
                    ("combined", "with_audio", "output_subtitles", "output_", "temp_output_")
                ) or file_path.suffix == ".wav":
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

    def build_final_video(
        self,
        entity_name: str,
        is_landmark: bool = False,
        config: VideoPipelineConfig | None = None,
    ) -> str:
        config = config or VideoPipelineConfig()
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        stage_times: dict[str, float] = {}
        total_start = time.time()

        script = self.get_script_by_name(entity_name, is_landmark=is_landmark)
        if not script:
            raise ValueError(f"No script found for entity: {entity_name}")

        paragraphs, sentence_groups = self.split_script_into_paragraph_sentences(script)

        tts_start = time.time()
        asyncio.run(
            self.generate_tts_audio(
                sentence_groups=sentence_groups,
                output_dir=config.output_dir,
                voice=config.voice,
                rate=config.rate,
                trim_top_db=config.trim_top_db,
            )
        )

        final_audio_path = self.combine_audio_files(config.output_dir, "Final audio.wav")
        _, sentence_durations, images_needed = self.compute_audio_durations(
            sentence_groups=sentence_groups,
            output_dir=config.output_dir,
            seconds_per_image=config.seconds_per_image,
            delete_sentence_files=True,
        )
        stage_times["TTS"] = time.time() - tts_start

        retrieval_start = time.time()
        image_text_chunks, seconds_for_chunk = self.create_image_chunks(
            sentence_groups=sentence_groups,
            images_needed=images_needed,
            sentence_durations=sentence_durations,
        )

        _, fetched_image_paths, _, _ = self.retrieve_images_semantic(
            name=entity_name,
            image_text_chunks=image_text_chunks,
            is_landmark=is_landmark,
            image_sim_weight=config.image_sim_weight,
            desc_sim_weight=config.desc_sim_weight,
        )

        image_files = self.download_images_from_r2(
            remote_paths=fetched_image_paths,
            local_frames_dir=config.temp_frames_dir,
        )
        image_files = self.normalize_images_to_jpeg(image_files)
        stage_times["Retrieval"] = time.time() - retrieval_start

        subtitles_start = time.time()
        srt_path = self.generate_srt(
            paragraphs=paragraphs,
            sentence_durations=sentence_durations,
            output_path=str(output_dir / "output_subtitles.srt"),
            max_chars_per_line=config.max_chars_per_line,
            max_lines=config.max_lines,
            min_duration=config.min_subtitle_duration,
        )
        stage_times["Subtitles"] = time.time() - subtitles_start

        motion_start = time.time()
        seconds = self.distribute_durations_exact(
            seconds_for_chunk,
            n_clips=len(image_files),
            fade=config.fade,
        )
        clips = self.generate_all_clips(
            image_files=image_files,
            durations=seconds,
            temp_dir=config.temp_clips_dir,
            fps=config.fps,
            target_w=config.target_w,
            target_h=config.target_h,
            use_nvenc=config.use_nvenc,
        )
        stage_times["Motion"] = time.time() - motion_start

        rendering_start = time.time()
        combined_video = output_dir / "combined.mp4"
        self.concatenate_clips(
            clips=clips,
            output_path=combined_video,
            fade=config.fade,
        )

        with_audio = output_dir / "with_audio.mp4"
        self.add_audio(combined_video, final_audio_path, with_audio)

        final_output = output_dir / f"{entity_name.replace(' ', '_')}_final_video.mp4"
        self.add_subtitles(
            video_path=with_audio,
            srt_path=srt_path,
            output_path=final_output,
        )
        stage_times["Rendering"] = time.time() - rendering_start
        stage_times["Total"] = time.time() - total_start

        if config.cleanup_intermediate:
            self.cleanup_files(
                output_dir=config.output_dir,
                temp_clips_dir=config.temp_clips_dir,
                temp_frames_dir=config.temp_frames_dir,
            )

        print(f"Done: {final_output}")
        print("\nStage Times:")
        print(f"TTS       : {self.format_stage_time(stage_times['TTS'])}")
        print(f"Retrieval : {self.format_stage_time(stage_times['Retrieval'])}")
        print(f"Motion    : {self.format_stage_time(stage_times['Motion'])}")
        print(f"Subtitles : {self.format_stage_time(stage_times['Subtitles'])}")
        print(f"Rendering : {self.format_stage_time(stage_times['Rendering'])}")
        print(f"Total     : {self.format_stage_time(stage_times['Total'])}")

        return str(final_output)

    def generate_video(self, *, entity_name: str, is_landmark: bool) -> Path:
        start_time = time.time()
        output_path = Path(
            self.build_final_video(
                entity_name=entity_name,
                is_landmark=is_landmark,
            )
        ).resolve()

        if not output_path.exists():
            raise RuntimeError(
                f"Video generation completed but output file was not created: {output_path}"
            )

        elapsed = time.time() - start_time
        if elapsed < 0:
            raise RuntimeError("Video generation finished with an invalid runtime measurement.")

        return output_path


video_generation_runtime = VideoGenerationRuntime()
