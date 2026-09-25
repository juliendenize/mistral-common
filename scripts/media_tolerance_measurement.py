"""Temporary fork-only measurement of public image and transcription media arrays."""

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download
from PIL import Image
from PIL import __version__ as pillow_version

from mistral_common.protocol.instruct.chunk import AudioChunk, ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV13
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV13
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.test_tokenizer_v15 import get_v15_mistral_tokenizer_with_audio

ARTIFACTS = {
    "v7_image": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "68faf511d618ef198fef186659617cfd2eb8e33a",
        14801330,
        "c604f35d1035f534519622c0ec83fed6184978d4fdee92a5bd2a50bc05438094",
    ),
    "v7_audio": (
        "mistralai/Voxtral-Mini-3B-2507",
        "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        14894206,
        "4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8",
    ),
    "v15_image": (
        "mistralai/Mistral-Small-4-119B-2603",
        "a11f36bebf709121056b1dbcc943d1c6afbe494d",
        16275354,
        "b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7",
    ),
}


def download(directory: Path) -> None:
    r"""Fetch only three immutable tokenizer JSON files, rejecting changed bytes."""
    for profile, (repo, revision, size, digest) in ARTIFACTS.items():
        path = Path(hf_hub_download(repo_id=repo, filename="tekken.json", revision=revision))
        data = path.read_bytes()
        actual_digest = hashlib.sha256(data).hexdigest()
        if len(data) != size or actual_digest != digest:
            raise ValueError(f"{profile} tokenizer mismatch: size={len(data)}, sha256={actual_digest}")
        destination = directory / profile / "tekken.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print(f"verified {profile}: {repo}@{revision} size={size} sha256={digest}")


def image_input(changed: bool) -> Image.Image:
    r"""Create a fresh structured RGB field with a visible local perturbation."""
    y, x = np.indices((157, 193), dtype=np.int32)
    channels = ((x * 7 + y * 3) % 256, (x * y + 37) % 256, (x * 2 + y * 11) % 256)
    pixels = np.stack(channels, axis=-1).astype(np.uint8)
    if changed:
        pixels[50:82, 70:102, 0] = 255 - pixels[50:82, 70:102, 0]
    return Image.fromarray(pixels)


def audio_input(changed: bool, sampling_rate: int) -> Audio:
    r"""Create two seconds of non-silent PCM input from fresh samples."""
    time = np.arange(2 * sampling_rate, dtype=np.float64) / sampling_rate
    samples = 0.17 * np.sin(2 * np.pi * 220 * time) + 0.09 * np.sin(2 * np.pi * 440 * time)
    if changed:
        samples *= 0.8
    return Audio(audio_array=samples.astype(np.float32), sampling_rate=sampling_rate, format="wav")


def encode_image(
    tokenizer: MistralTokenizer, changed: bool, reasoning_effort: ReasoningEffort | None
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the complete public chat image and its observable metadata."""
    request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[TextChunk(text="Describe this pattern"), ImageChunk(image=image_input(changed))])
        ],
        reasoning_effort=reasoning_effort,
    )
    result = tokenizer.encode_chat_completion(request=request)
    if len(result.images) != 1 or result.audios:
        raise ValueError("Expected exactly one returned image and no audio")
    return result.images[0], {
        "tokens": result.tokens,
        "text": tokenizer.decode(tokens=result.tokens, special_token_policy=SpecialTokenPolicy.KEEP),
        "shape": list(result.images[0].shape),
        "dtype": str(result.images[0].dtype),
    }


def encode_audio(
    tokenizer: MistralTokenizer, changed: bool, sampling_rate: int, chat: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the complete public transcription or chat waveform and metadata."""
    input_audio = audio_input(changed=changed, sampling_rate=sampling_rate).to_base64(format="wav")
    if chat:
        chat_request = ChatCompletionRequest(messages=[UserMessage(content=[AudioChunk(input_audio=input_audio)])])
        result = tokenizer.encode_chat_completion(request=chat_request)
    else:
        transcription_request = TranscriptionRequest(
            model="measurement", audio=input_audio, language=None, target_streaming_delay_ms=None
        )
        result = tokenizer.encode_transcription(request=transcription_request)
    if result.images or len(result.audios) != 1:
        raise ValueError("Expected exactly one returned audio and no images")
    audio = result.audios[0]
    return audio.audio_array, {
        "tokens": result.tokens,
        "text": tokenizer.decode(tokens=result.tokens, special_token_policy=SpecialTokenPolicy.KEEP),
        "shape": list(audio.audio_array.shape),
        "dtype": str(audio.audio_array.dtype),
        "sampling_rate": audio.sampling_rate,
        "format": audio.format,
    }


def synthetic_v13_audio() -> MistralTokenizer:
    r"""Recreate the existing v13 audio fixture with its public validator and normalizer."""
    tokenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False, add_audio=True),
        pattern=r".+",
        vocab_size=356,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    config = AudioConfig(
        sampling_rate=24000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
    )
    ids = SpecialAudioIDs(
        audio=tokenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    return MistralTokenizer(
        instruct_tokenizer=InstructTokenizerV13(
            tokenizer=tokenizer, audio_encoder=AudioEncoder(audio_config=config, special_ids=ids)
        ),
        validator=MistralRequestValidatorV13(),
        request_normalizer=InstructRequestNormalizerV13.normalizer(),
    )


def load_profile(profile: str, directory: Path) -> tuple[MistralTokenizer, dict[str, Any]]:
    r"""Load a checked released or bundled file, or a clearly labeled synthetic fixture."""
    if profile == "v13_audio_synthetic":
        return synthetic_v13_audio(), {"source": "synthetic tests/test_tokenizer_v13.py:71-99"}
    if profile == "v15_audio_synthetic":
        return get_v15_mistral_tokenizer_with_audio(), {"source": "synthetic tests/test_tokenizer_v15.py:174-210"}
    identity: dict[str, Any]
    if profile == "v3_image":
        path = MistralTokenizer._data_path() / "tekken_240911.json"
        identity = {"source": "bundled tekken_240911.json at f3bb6e8"}
    else:
        repo, revision, size, digest = ARTIFACTS[profile]
        path = directory / profile / "tekken.json"
        identity = {"source": "pinned Hub tokenizer", "repo": repo, "revision": revision}
    data = path.read_bytes()
    actual_digest = hashlib.sha256(data).hexdigest()
    if profile != "v3_image" and (len(data) != size or actual_digest != digest):
        raise ValueError(f"Local {profile} tokenizer mismatch: size={len(data)}, sha256={actual_digest}")
    identity.update({"size": len(data), "sha256": actual_digest})
    return MistralTokenizer.from_file(tokenizer_filename=path), identity


def differences(reference: np.ndarray, other: np.ndarray) -> dict[str, float | int]:
    r"""Inspect all elements without discarding zeros or silently accepting NaNs."""
    if reference.shape != other.shape or reference.dtype != other.dtype:
        raise ValueError("Array shape or dtype changed")
    if not np.isfinite(reference).all() or not np.isfinite(other).all():
        raise ValueError("Non-finite media array")
    delta = np.abs(reference.astype(np.float64) - other.astype(np.float64))
    relative = delta / np.maximum(np.abs(reference.astype(np.float64)), 1e-8)
    return {
        "max_abs": float(delta.max()),
        "max_relative_floor_1e-8": float(relative.max()),
        "changed_elements": int(np.count_nonzero(delta)),
        "total_elements": int(delta.size),
    }


def measure(directory: Path, output: Path) -> None:
    r"""Measure repeats and perturbations with verified local files only."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Calibration requires Linux x86_64")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "mistral-common": version("mistral-common"),
            "numpy": np.__version__,
            "pillow": pillow_version,
            "opencv": cv2.__version__,
            "soundfile": sf.__version__,
            "soxr": soxr.__version__,
            "huggingface-hub": version("huggingface-hub"),
        },
        "recipe": {
            "image": {
                "input": "RGB uint8 157x193: R=(7x+3y)%256 G=(xy+37)%256 B=(2x+11y)%256",
                "changed": "invert R at y=50:82,x=70:102",
            },
            "audio": {
                "input": "2s float32 samples: 0.17sin(2pi220t)+0.09sin(2pi440t); WAV PCM_16 base64",
                "rates": "v7 transcription 16000 or 22050 Hz; synthetic chat 22050 Hz",
                "changed": "scale samples by 0.8",
            },
        },
    }
    profiles = (
        ("v7_image", TokenizerVersion.v7),
        ("v3_image", TokenizerVersion.v3),
        ("v15_image", TokenizerVersion.v15),
        ("v7_audio", TokenizerVersion.v7),
        ("v7_audio_resampled", TokenizerVersion.v7),
        ("v13_audio_synthetic", TokenizerVersion.v13),
        ("v15_audio_synthetic", TokenizerVersion.v15),
    )
    for profile, expected_version in profiles:
        artifact_profile = "v7_audio" if profile == "v7_audio_resampled" else profile
        tokenizer, identity = load_profile(profile=artifact_profile, directory=directory)
        if tokenizer.version != expected_version:
            raise ValueError(f"{profile} unexpected tokenizer version: {tokenizer.version}")
        if "image" in profile:
            if tokenizer.instruct_tokenizer.image_encoder is None:
                raise ValueError(f"{profile} lacks an image encoder")
            effort = ReasoningEffort.high if profile == "v15_image" else None
            results = [
                encode_image(tokenizer=tokenizer, changed=changed, reasoning_effort=effort)
                for changed in (False, False, True)
            ]
        else:
            if tokenizer.instruct_tokenizer.audio_encoder is None:
                raise ValueError(f"{profile} lacks an audio encoder")
            sampling_rate = 16000 if profile == "v7_audio" else 22050
            chat = profile in ("v13_audio_synthetic", "v15_audio_synthetic")
            results = [
                encode_audio(tokenizer=tokenizer, changed=changed, sampling_rate=sampling_rate, chat=chat)
                for changed in (False, False, True)
            ]
        baseline, baseline_metadata = results[0]
        repeated, repeated_metadata = results[1]
        perturbed, perturbed_metadata = results[2]
        if baseline_metadata != repeated_metadata:
            raise ValueError(f"{profile} repeat metadata differs")
        for key in ("shape", "dtype", "sampling_rate", "format"):
            if baseline_metadata.get(key) != perturbed_metadata.get(key):
                raise ValueError(f"{profile} perturbation changed {key}")
        sensitivity = differences(baseline, perturbed)
        if not sensitivity["changed_elements"]:
            raise ValueError(f"{profile} input perturbation did not change returned array")
        for label, array in (("baseline", baseline), ("repeat", repeated), ("perturbed", perturbed)):
            np.save(output / f"{profile}-{label}.npy", array, allow_pickle=False)
        report[profile] = {
            **identity,
            "version": tokenizer.version.value,
            "mode": tokenizer.mode.value,
            "operation": "chat" if profile.endswith("audio_synthetic") or "image" in profile else "transcription",
            "input_sampling_rate": None if "image" in profile else sampling_rate,
            "reasoning_effort": "high" if profile == "v15_image" else None,
            "baseline": baseline_metadata,
            "perturbed": perturbed_metadata,
            "repeat_difference": differences(baseline, repeated),
            "perturbation_difference": sensitivity,
            "baseline_min": float(baseline.min()),
            "baseline_max": float(baseline.max()),
            "baseline_rms": float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2))),
            "baseline_distinct_values": int(np.unique(baseline).size),
        }
    (output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def compare(directory: Path) -> None:
    r"""Compare every uploaded full array and public metadata to Python 3.10."""
    reports: dict[str, Any] = {}
    for py in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        reports[py] = json.loads((directory / f"py{py}" / "metadata.json").read_text())
    profiles = (
        "v7_image",
        "v3_image",
        "v15_image",
        "v7_audio",
        "v7_audio_resampled",
        "v13_audio_synthetic",
        "v15_audio_synthetic",
    )
    for profile in profiles:
        reference = reports["3.10"][profile]
        max_abs = 0.0
        max_relative = 0.0
        for py, report in reports.items():
            if report["recipe"] != reports["3.10"]["recipe"]:
                raise ValueError(f"Python {py} input recipe differs")
            actual = report[profile]
            if actual.keys() != reference.keys():
                raise ValueError(f"{profile} Python {py} metadata fields differ")
            for key in actual:
                if key in (
                    "repeat_difference",
                    "perturbation_difference",
                    "baseline_min",
                    "baseline_max",
                    "baseline_rms",
                    "baseline_distinct_values",
                ):
                    continue
                if actual.get(key) != reference.get(key):
                    raise ValueError(f"{profile} Python {py} metadata differs: {key}")
            for label in ("baseline", "repeat", "perturbed"):
                array = np.load(directory / f"py{py}" / f"{profile}-{label}.npy", allow_pickle=False)
                expected = np.load(directory / "py3.10" / f"{profile}-{label}.npy", allow_pickle=False)
                delta = differences(reference=expected, other=array)
                max_abs = max(max_abs, delta["max_abs"])
                max_relative = max(max_relative, delta["max_relative_floor_1e-8"])
                print(f"{profile} py{py} {label} vs py3.10 {label}: {delta}")
            current_baseline = np.load(directory / f"py{py}" / f"{profile}-baseline.npy", allow_pickle=False)
            repeated = np.load(directory / f"py{py}" / f"{profile}-repeat.npy", allow_pickle=False)
            repeat_delta = differences(reference=current_baseline, other=repeated)
            if repeat_delta != actual["repeat_difference"]:
                raise ValueError(f"{profile} Python {py} repeat measurement differs from full arrays")
            max_abs = max(max_abs, repeat_delta["max_abs"])
            max_relative = max(max_relative, repeat_delta["max_relative_floor_1e-8"])
            print(f"{profile} py{py} baseline vs repeat: {repeat_delta}")
            perturbed = np.load(directory / f"py{py}" / f"{profile}-perturbed.npy", allow_pickle=False)
            sensitivity = differences(reference=current_baseline, other=perturbed)
            if sensitivity != actual["perturbation_difference"] or not sensitivity["changed_elements"]:
                raise ValueError(f"{profile} Python {py} invalid input perturbation measurement")
            print(f"{profile} py{py} baseline vs changed input: {sensitivity}")
        print(f"{profile} maximum cross-environment/repeat absolute={max_abs} relative={max_relative}")
    for py, report in reports.items():
        print(f"py{py}: {report['platform']} {report['versions']}")


def main() -> None:
    r"""Select a networked provision step or an offline measurement/comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download", "measure", "compare"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "download":
        download(directory=args.directory)
    elif args.operation == "measure":
        if args.output is None:
            parser.error("measure requires --output")
        measure(directory=args.directory, output=args.output)
    else:
        compare(directory=args.directory)


if __name__ == "__main__":
    main()
