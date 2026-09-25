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

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

ARTIFACTS = {
    "image": (
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "68faf511d618ef198fef186659617cfd2eb8e33a",
        14801330,
        "c604f35d1035f534519622c0ec83fed6184978d4fdee92a5bd2a50bc05438094",
    ),
    "audio": (
        "mistralai/Voxtral-Mini-3B-2507",
        "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        14894206,
        "4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8",
    ),
}


def download(directory: Path) -> None:
    r"""Fetch only two immutable tokenizer JSON files, rejecting changed bytes."""
    for kind, (repo, revision, size, digest) in ARTIFACTS.items():
        path = Path(hf_hub_download(repo_id=repo, filename="tekken.json", revision=revision))
        data = path.read_bytes()
        actual_digest = hashlib.sha256(data).hexdigest()
        if len(data) != size or actual_digest != digest:
            raise ValueError(f"{kind} tokenizer mismatch: size={len(data)}, sha256={actual_digest}")
        destination = directory / kind / "tekken.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print(f"verified {kind}: {repo}@{revision} size={size} sha256={digest}")


def image_input(changed: bool) -> Image.Image:
    r"""Create a fresh structured RGB field with a visible local perturbation."""
    y, x = np.indices((157, 193), dtype=np.int32)
    channels = ((x * 7 + y * 3) % 256, (x * y + 37) % 256, (x * 2 + y * 11) % 256)
    pixels = np.stack(channels, axis=-1).astype(np.uint8)
    if changed:
        pixels[50:82, 70:102, 0] = 255 - pixels[50:82, 70:102, 0]
    return Image.fromarray(pixels)


def audio_input(changed: bool) -> Audio:
    r"""Create two seconds of non-silent 16-kHz PCM input from fresh samples."""
    time = np.arange(32000, dtype=np.float64) / 16000
    samples = 0.17 * np.sin(2 * np.pi * 220 * time) + 0.09 * np.sin(2 * np.pi * 440 * time)
    if changed:
        samples *= 0.8
    return Audio(audio_array=samples.astype(np.float32), sampling_rate=16000, format="wav")


def encode_image(tokenizer: MistralTokenizer, changed: bool) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the complete public chat image and its observable metadata."""
    request = ChatCompletionRequest(
        messages=[
            UserMessage(content=[TextChunk(text="Describe this pattern"), ImageChunk(image=image_input(changed))])
        ]
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


def encode_audio(tokenizer: MistralTokenizer, changed: bool) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Return the complete public transcription waveform and metadata."""
    request = TranscriptionRequest(
        model="measurement",
        audio=audio_input(changed).to_base64(format="wav"),
        language=None,
        target_streaming_delay_ms=None,
    )
    result = tokenizer.encode_transcription(request=request)
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
                "input": "16000 Hz 32000 float32 samples: 0.17sin(2pi220t)+0.09sin(2pi440t); WAV PCM_16 base64",
                "changed": "scale samples by 0.8",
            },
        },
    }
    for kind, encode in (("image", encode_image), ("audio", encode_audio)):
        repo, revision, size, digest = ARTIFACTS[kind]
        path = directory / kind / "tekken.json"
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Local {kind} tokenizer mismatch")
        tokenizer = MistralTokenizer.from_file(tokenizer_filename=path)
        if tokenizer.instruct_tokenizer.tokenizer.version != TokenizerVersion.v7:
            raise ValueError(f"{kind} tokenizer is not v7")
        results = [encode(tokenizer, changed=changed) for changed in (False, False, True)]
        baseline, baseline_metadata = results[0]
        repeated, repeated_metadata = results[1]
        perturbed, perturbed_metadata = results[2]
        if baseline_metadata != repeated_metadata:
            raise ValueError(f"{kind} repeat metadata differs")
        if (baseline_metadata["shape"], baseline_metadata["dtype"]) != (
            perturbed_metadata["shape"],
            perturbed_metadata["dtype"],
        ):
            raise ValueError(f"{kind} perturbation changed shape or dtype")
        sensitivity = differences(baseline, perturbed)
        if not sensitivity["changed_elements"]:
            raise ValueError(f"{kind} input perturbation did not change returned array")
        for label, array in (("baseline", baseline), ("repeat", repeated), ("perturbed", perturbed)):
            np.save(output / f"{kind}-{label}.npy", array, allow_pickle=False)
        report[kind] = {
            "repo": repo,
            "revision": revision,
            "size": size,
            "sha256": digest,
            "mode": "test",
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
    for kind in ("image", "audio"):
        reference = reports["3.10"][kind]
        baseline = np.load(directory / "py3.10" / f"{kind}-baseline.npy", allow_pickle=False)
        for py, report in reports.items():
            actual = report[kind]
            for key in ("repo", "revision", "size", "sha256", "mode", "baseline", "perturbed"):
                if actual[key] != reference[key]:
                    raise ValueError(f"{kind} Python {py} metadata differs: {key}")
            for label in ("baseline", "repeat", "perturbed"):
                array = np.load(directory / f"py{py}" / f"{kind}-{label}.npy", allow_pickle=False)
                expected = np.load(directory / "py3.10" / f"{kind}-{label}.npy", allow_pickle=False)
                print(f"{kind} py{py} {label} vs py3.10 {label}: {differences(expected, array)}")
            current_baseline = np.load(directory / f"py{py}" / f"{kind}-baseline.npy", allow_pickle=False)
            repeated = np.load(directory / f"py{py}" / f"{kind}-repeat.npy", allow_pickle=False)
            print(f"{kind} py{py} baseline vs repeat: {differences(current_baseline, repeated)}")
        perturbed = np.load(directory / "py3.10" / f"{kind}-perturbed.npy", allow_pickle=False)
        print(f"{kind} py3.10 baseline vs changed input: {differences(baseline, perturbed)}")
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
