#!/usr/bin/env python3
"""Utility for downloading LLM models (safe for shared servers).

- Downloads from Hugging Face to a local folder (default: ./llm_models)
- Uses repo_id-based folder names to avoid duplicates (e.g., google__gemma-3-270m)
- Skips download if the model already exists (tokenizer + weights OR .download_complete)
- Prevents concurrent downloads with a simple lock file
- (Optional) Pin to a specific revision (commit hash or tag)

Examples:
    # List known shorthands
    python download_hugging.py --list-models

    # Download by shorthand
    python download_hugging.py --model gemma-3-270m

    # Download by repo_id directly (no need to edit MODEL_REPOS)
    python download_hugging.py --model google/gemma-3-270m

    # Download to a shared directory
    python download_hugging.py --model google/gemma-3-270m --base-dir /mnt/hdd1/llm_models

    # Pin to a specific revision
    python download_hugging.py --model google/gemma-3-270m --revision <commit_or_tag>

Auth:
    - Put your token in HF_TOKEN env var, or
    - Create a file `hf_token.txt` in this script directory (or two levels above).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Mapping of shorthand names to Hugging Face repositories.
MODEL_REPOS = {
    # "gemma": "google/gemma-2b",
    # "gemma-7b": "google/gemma-7b",
    "gemma-3-270m": "google/gemma-3-270m",
    # "gemma-3-4b-it": "google/gemma-3-4b-it",
    # "llama": "meta-llama/Meta-Llama-3-8B",
    # "llama-7b": "meta-llama/Llama-2-7b-hf",
    # "gpt-oss-120b": "openai/gpt-oss-120b",
    # "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    # "gpt-oss-20b": "openai/gpt-oss-20b",
    # "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    # "qwen": "Qwen/Qwen-7B",
}


def _read_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from file or environment."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    token_paths = [
        Path(__file__).resolve().parent / "hf_token.txt",
        Path(__file__).resolve().parents[2] / "hf_token.txt",
    ]
    for path in token_paths:
        if path.is_file():
            return path.read_text().strip()
    return None


def _safe_dirname(repo_id: str) -> str:
    """Make a filesystem-safe directory name from repo_id."""
    return repo_id.replace("/", "__")


def _is_model_downloaded(model_dir: Path) -> bool:
    """Conservative check to decide if a model is fully downloaded."""
    if not model_dir.exists():
        return False

    # Prefer an explicit completion marker to avoid half-downloaded states.
    if (model_dir / ".download_complete").exists():
        return True

    # Tokenizer present?
    tok_ok = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists()

    # Weight files present? (supporting both safetensors and PyTorch bin sharded/single)
    weight_ok = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))

    return tok_ok and weight_ok


def _write_completion_marker(model_dir: Path, repo_id: str, revision: Optional[str]) -> None:
    try:
        content = f"repo_id={repo_id}\nrevision={revision or 'None'}\n"
        (model_dir / ".download_complete").write_text(content)
    except Exception:
        # Marker write failure should not be fatal
        pass


def download_model(
    model: str,
    base_dir: Optional[os.PathLike[str]] = None,
    *,
    revision: Optional[str] = None,
    wait_lock_seconds: int = 120,
) -> Path:
    """Download an LLM model and return the local path.

    Args:
        model: Shorthand name or Hugging Face repo_id.
        base_dir: Base directory in which models are stored. When None,
            defaults to "<this_script_dir>/llm_models".
        revision: Optional HF commit hash or tag to pin a specific version.
        wait_lock_seconds: How long to wait if another process is downloading.

    Returns:
        Path to the downloaded model directory.
    """
    repo_id = MODEL_REPOS.get(model, model)
    safe_name = _safe_dirname(repo_id)

    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent / "llm_models"
    target_dir = base / safe_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Simple lock to avoid concurrent downloads into the same directory.
    lock = target_dir.parent / f"{target_dir.name}.lock"
    if lock.exists():
        print(f"다른 프로세스가 다운로드 중입니다: {lock}")
        # Wait (poll) for a while
        waited = 0
        while lock.exists() and waited < wait_lock_seconds:
            time.sleep(1)
            waited += 1
        if lock.exists():
            raise SystemExit("잠금이 풀리지 않았습니다. 잠시 후 다시 시도하세요.")

    if _is_model_downloaded(target_dir):
        print(f"모델 '{repo_id}'이 이미 다운로드되어 있습니다: {target_dir}")
        return target_dir

    # Acquire lock
    try:
        lock.touch(exist_ok=False)
    except Exception:
        # If we fail to create lock but it's present, bail out to be safe
        if lock.exists():
            raise SystemExit("다른 프로세스가 잠금을 보유 중입니다. 잠시 후 재시도하세요.")
        else:
            # Unexpected, but proceed without lock
            pass

    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub package is required. Install with 'pip install huggingface_hub'."
            ) from exc

        print(f"모델 '{repo_id}' 다운로드 중...")
        token = _read_hf_token()

        # Perform the download. Note: local_dir_use_symlinks is deprecated.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token,
            revision=revision,
        )

        # Mark completion (best-effort)
        _write_completion_marker(target_dir, repo_id, revision)

        print(f"모델 '{repo_id}' 다운로드 완료: {target_dir}")
        return target_dir

    except Exception as e:
        # Friendlier hints for common gated repo issues.
        msg = str(e)
        if "GatedRepoError" in msg or "401" in msg or "403" in msg:
            print(
                "⚠️ 접근 권한 문제로 보입니다. 모델 페이지에서 라이선스 동의/승인을 먼저 완료하세요 "
                f"(예: https://huggingface.co/{repo_id})."
            )
        print("가능한 해결 방법:")
        print("1. 인터넷 연결 확인")
        print("2. Hugging Face 토큰 설정 확인 (hf_token.txt 또는 HF_TOKEN)")
        print("3. 모델 이름/권한(라이선스 동의) 확인")
        raise
    finally:
        # Release lock
        try:
            if lock.exists():
                lock.unlink()
        except Exception:
            pass


def _cli() -> None:
    epilog_lines = ["Available models (shorthand: repo_id):"]
    seen_repos = set()
    for name, repo in MODEL_REPOS.items():
        if repo in seen_repos:
            continue
        seen_repos.add(repo)
        epilog_lines.append(f"  {name}: {repo}")

    parser = argparse.ArgumentParser(
        description="Download LLM models (safe for shared servers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )
    parser.add_argument(
        "--model",
        help="Model shorthand or Hugging Face repo_id (e.g., 'gemma-3-270m' or 'google/gemma-3-270m').",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for storing models (defaults to '<script_dir>/llm_models').",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF revision (commit hash or tag) to pin a specific version.",
    )
    parser.add_argument(
        "--wait-lock-seconds",
        type=int,
        default=120,
        help="Seconds to wait if another process is downloading the same model (default: 120).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported model shorthands and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        for name in sorted(MODEL_REPOS):
            print(name)
        return

    if not args.model:
        parser.error("--model is required unless --list-models is given")

    path = download_model(
        args.model,
        args.base_dir,
        revision=args.revision,
        wait_lock_seconds=args.wait_lock_seconds,
    )
    print(f"Model downloaded to: {path}")


if __name__ == "__main__":
    _cli()