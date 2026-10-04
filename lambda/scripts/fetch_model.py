"""Download the pinned Strands Decider checkpoint and its base model into one directory.

Layout written to OUT:
    OUT/checkpoint/   the Decider LoRA adapter, readout head and tokenizer
    OUT/base/         Qwen3.5-2B-Base text decoder, split into layer-sized shards
    OUT/_hub/base/    the untouched Hub download (not copied into the image)

The checkpoint does not contain the base weights, and its config names them by Hub id with
no revision. base_model is rewritten to OUT/base so the loader never touches the network.

The Decider only loads the text decoder, so the vision tower and the MTP head are dropped.
Tensors are copied byte for byte. The shards are small so that each becomes its own Docker
layer: a dropped connection while pushing to ECR then costs one shard, not the whole model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINT_REPO = "StrandsAgents/strands-decider-2B-hobson-v19"
CHECKPOINT_REV = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
BASE_REPO = "Qwen/Qwen3.5-2B-Base"
BASE_REV = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
BASE_FILE = "model.safetensors-00001-of-00001.safetensors"
BASE_SHA256 = "928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d"

DROPPED_PREFIXES = ("model.visual.", "mtp.")
SHARD_BYTES = 512 * 1024 * 1024
# The Dockerfile has one COPY per shard. This fails the build if the layout drifts from it.
EXPECTED_SHARDS = 8

CONFIG_NAMES = ("strands_decider_config.json", "hobson_config.json")
CHECKPOINT_FILES = [
    *CONFIG_NAMES,
    "head.safetensors",
    "lora/*",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE.md",
    "MANIFEST.sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_against_manifest(checkpoint: Path) -> int:
    """Check every downloaded file that MANIFEST.sha256 lists. Returns the count checked."""
    checked = 0
    for line in (checkpoint / "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        target = checkpoint / name.strip().lstrip("*")
        if not target.is_file():
            continue
        if sha256(target) != expected:
            raise SystemExit(f"checksum mismatch: {target}")
        checked += 1
    return checked


def read_header(path: Path) -> tuple[dict, int]:
    """Return the safetensors header and the offset where tensor data starts."""
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(length)), 8 + length


def plan_shards(tensors: list[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    """Pack tensors in file order; a tensor larger than SHARD_BYTES gets a shard to itself."""
    shards: list[list[tuple[str, dict]]] = []
    current: list[tuple[str, dict]] = []
    size = 0
    for name, info in tensors:
        start, end = info["data_offsets"]
        if current and size + (end - start) > SHARD_BYTES:
            shards.append(current)
            current, size = [], 0
        current.append((name, info))
        size += end - start
    if current:
        shards.append(current)
    return shards


def write_shard(src: Path, data_start: int, dst: Path, tensors: list[tuple[str, dict]]) -> int:
    header: dict = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name, info in tensors:
        length = info["data_offsets"][1] - info["data_offsets"][0]
        header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [cursor, cursor + length]}
        cursor += length
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)

    with src.open("rb") as reader, dst.open("wb") as writer:
        writer.write(struct.pack("<Q", len(raw)))
        writer.write(raw)
        for _, info in tensors:
            start, end = info["data_offsets"]
            reader.seek(data_start + start)
            remaining = end - start
            while remaining:
                block = reader.read(min(remaining, 8 << 20))
                if not block:
                    raise SystemExit(f"unexpected end of {src}")
                writer.write(block)
                remaining -= len(block)
    return cursor


def reshard_base(hub_dir: Path, out_dir: Path) -> None:
    src = hub_dir / BASE_FILE
    if sha256(src) != BASE_SHA256:
        raise SystemExit(f"{src}: sha256 differs from the Hub's published {BASE_SHA256}")

    header, data_start = read_header(src)
    header.pop("__metadata__", None)
    kept = sorted(
        ((n, i) for n, i in header.items() if not n.startswith(DROPPED_PREFIXES)),
        key=lambda item: item[1]["data_offsets"][0],
    )
    shards = plan_shards(kept)
    if len(shards) != EXPECTED_SHARDS:
        raise SystemExit(f"{len(shards)} shards planned, the Dockerfile expects {EXPECTED_SHARDS}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.safetensors"):
        stale.unlink()

    weight_map: dict[str, str] = {}
    total = 0
    for index, group in enumerate(shards):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        total += write_shard(src, data_start, out_dir / name, group)
        weight_map.update({tensor: name for tensor, _ in group})
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )
    print(f"base: kept {len(kept)} of {len(header)} tensors, {total / 1e9:.2f} GB in {len(shards)} shards")

    for extra in hub_dir.iterdir():
        if extra.suffix in {".json", ".txt"} and extra.name not in {"model.safetensors.index.json"}:
            (out_dir / extra.name).write_bytes(extra.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out: Path = args.out.resolve()

    checkpoint = out / "checkpoint"
    hub_base = out / "_hub" / "base"
    base = out / "base"

    snapshot_download(
        CHECKPOINT_REPO,
        revision=CHECKPOINT_REV,
        local_dir=checkpoint,
        allow_patterns=CHECKPOINT_FILES,
    )
    snapshot_download(
        BASE_REPO,
        revision=BASE_REV,
        local_dir=hub_base,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )

    print(f"verified {verify_against_manifest(checkpoint)} checkpoint files against MANIFEST.sha256")
    reshard_base(hub_base, base)

    config_path = next((checkpoint / n for n in CONFIG_NAMES if (checkpoint / n).is_file()), None)
    if config_path is None:
        sys.exit(f"no decider config in {checkpoint}")
    config = json.loads(config_path.read_text())
    config["base_model"] = str(base)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"{config_path.name}: base_model -> {base}")


if __name__ == "__main__":
    main()
