#!/usr/bin/env python3
"""Utility for downloading LLM models.

This module provides a helper function ``download_model`` that fetches
models from the Hugging Face Hub and stores them locally under the
``<script_dir>/llm_models`` directory by default.
Other applications can import this module and call ``download_model`` to obtain a local path
to a model that can be loaded without re-downloading.

**경로 설정 안내:**
- 기본 저장 경로: 스크립트 위치 기준 ``llm_models`` 디렉토리
- 사용자별 환경에 맞게 ``base_dir`` 파라미터로 변경 가능
- 토큰 파일: ``hf_token.txt`` 또는 ``HF_TOKEN`` 환경변수 사용

Example:
    >>> from download_model import download_model
    >>> path = download_model("gemma")
    >>> print(path)

The script can also be used directly from the command line::

    python3 download_model.py

If authentication is required, place your Hugging Face token in
``hf_token.txt`` (this directory or the repository root) or set the
``HF_TOKEN`` environment variable.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


# Mapping of shorthand names to Hugging Face repositories.
MODEL_REPOS = {
    "gemma": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "gemma-3-270m": "google/gemma-3-270m",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "llama": "meta-llama/Meta-Llama-3-8B",
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen": "Qwen/Qwen-7B",
}

def _read_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from file or environment."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()
    # 토큰 파일 경로 설정 (사용자별 환경에 맞게 수정 필요)
    # 1. 스크립트와 같은 디렉토리: /mnt/hdd1/jihye0e/huggingface/huggingface/hf_token.txt
    # 2. 프로젝트 루트 디렉토리: /mnt/hdd1/jihye0e/huggingface/hf_token.txt
    token_paths = [
        Path(__file__).resolve().parent / "hf_token.txt",  # 스크립트 위치 기준
        Path(__file__).resolve().parents[2] / "hf_token.txt",  # 프로젝트 루트 기준
    ]
    for path in token_paths:
        if path.is_file():
            return path.read_text().strip()
    return None

def _is_model_downloaded(model_dir: Path) -> bool:
    """Check if a model is already downloaded by looking for essential files."""
    if not model_dir.exists():
        return False
    # 필수 파일 확인
    if (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists():
        if any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin")):
            return True
    # 보수적으로 추가 확인
    for name in ["config.json", "pytorch_model.bin", "model.safetensors", "pytorch_model-00001-of-00001.bin"]:
        if (model_dir / name).exists():
            return True
    return False

def download_model(model: str, base_dir: Optional[os.PathLike[str]] = None) -> Path:
    """Download an LLM model and return the local path.

    Args:
        model: Shorthand name or Hugging Face ``repo_id``.
        base_dir: Base directory in which models are stored. When ``None``
            defaults to ``<script_dir>/llm_models``.

    Returns:
        Path to the downloaded model directory.
    """
    repo_id = MODEL_REPOS.get(model, model)
    
    # 모델명 정규화: google/gemma-3-270m -> gemma-3-270m
    model_name = repo_id.split('/')[-1] if '/' in repo_id else model
    
    # 기본 저장 경로 설정
    # 사용자별 환경에 맞게 수정 필요:
    # - 현재: /mnt/hdd1/jihye0e/huggingface/huggingface/llm_models
    # - 개인 환경: ~/models 또는 원하는 경로로 변경
    # - 서버 환경: 충분한 디스크 공간이 있는 경로로 설정
    base = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent / "llm_models"  # 스크립트 위치 기준 상대 경로
    )
    target_dir = base / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 이미 다운로드된 모델이 있는지 확인
    if _is_model_downloaded(target_dir):
        # print(f"모델 '{model}'이 이미 다운로드되어 있습니다: {target_dir}")
        print(f"모델 '{model}'이 이미 다운로드되어 있습니다.")
        return target_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub package is required. Install with 'pip install huggingface_hub'."
        ) from exc

    print(f"모델 '{model}' 다운로드 중...")
    token = _read_hf_token()

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  
            token=token,
            resume_download=True,  # 중간에 끊겼을 때 이어받기
            max_workers=4,  # 병렬 워커 줄여서 메모리/FD 폭주 방지
            # allow_patterns=["*.safetensors", "pytorch_model*.bin", "config.json", "tokenizer.*"], # 필요할 때만
        )
        print(f"모델 '{model}' 다운로드 완료: {target_dir}")
    except Exception as e:
        for attempt in range(3):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                    max_workers=4,
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"실패, {2**attempt}초 후 재시도...")
                    time.sleep(2**attempt)
                else:
                    raise

    return target_dir

def _cli() -> None:
    epilog_lines = ["Available models:"]
    seen_repos = set()
    for name, repo in MODEL_REPOS.items():
        if repo in seen_repos:
            continue
        seen_repos.add(repo)
        epilog_lines.append(f"  {name}: {repo}")
    parser = argparse.ArgumentParser(
        description="Download LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(epilog_lines),
    )
    parser.add_argument(
        "--model",
        help="Model shorthand (e.g., 'gemma', 'llama-7b') or full Hugging Face repo_id (e.g., 'google/gemma-2b').",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for storing models (defaults to '<script_dir>/llm_models').",
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
    path = download_model(args.model, args.base_dir)
    print(f"Model downloaded to: {path}")

if __name__ == "__main__":
    _cli()