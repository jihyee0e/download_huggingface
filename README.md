# Hugging Face 모델 다운로드 및 관리 도구

이 프로젝트는 Hugging Face Hub에서 대규모 언어 모델(LLM)을 다운로드하고 로컬에서 관리하기 위한 유틸리티 도구입니다.

## 주요 기능

- **모델 자동 다운로드**: Hugging Face Hub에서 모델을 자동으로 다운로드
- **중복 다운로드 방지**: 이미 다운로드된 모델은 재다운로드하지 않음
- **재시도 메커니즘**: 네트워크 오류 시 자동 재시도
- **토큰 관리**: Hugging Face 인증 토큰 자동 처리
- **다양한 모델 지원**: Gemma, Llama, GPT-OSS, Qwen 등 주요 모델 지원

## 지원 모델

| 단축명 | 전체 경로 | 설명 |
|--------|-----------|------|
| `gemma` | `google/gemma-2b` | Google Gemma 2B 모델 |
| `gemma-7b` | `google/gemma-7b` | Google Gemma 7B 모델 |
| `gemma-3-270m` | `google/gemma-3-270m` | Google Gemma 3 270M 모델 |
| `gemma-3-4b-it` | `google/gemma-3-4b-it` | Google Gemma 3 4B Instruct 모델 |
| `llama` | `meta-llama/Meta-Llama-3-8B` | Meta Llama 3 8B 모델 |
| `llama-7b` | `meta-llama/Llama-2-7b-hf` | Meta Llama 2 7B 모델 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | OpenAI GPT-OSS 120B 모델 |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | OpenAI GPT-OSS 20B 모델 |
| `qwen` | `Qwen/Qwen-7B` | Qwen 7B 모델 |

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Hugging Face 토큰 설정

다음 중 하나의 방법으로 Hugging Face 토큰을 설정하세요:

**방법 1: 파일로 설정**
```bash
echo "your_hf_token_here" > hf_token.txt
```

**방법 2: 환경변수로 설정**
```bash
export HF_TOKEN="your_hf_token_here"
```

## 사용법

### 명령줄에서 사용

```bash
# 특정 모델 다운로드
python3 huggingface/download_models.py --model gemma

# 지원되는 모델 목록 확인
python3 huggingface/download_models.py --list-models

# 사용자 정의 디렉토리에 모델 다운로드
python3 huggingface/download_models.py --model llama --base-dir /path/to/custom/dir
```

## 프로젝트 구조

```
huggingface/
├── huggingface/
│   ├── download_models.py    # 메인 다운로드 스크립트
│   ├── hf_token.txt         # Hugging Face 토큰
│   └── llm_models/          # 다운로드된 모델 저장소
│       ├── gemma-3-270m/
│       ├── gpt-oss-20b/
│       └── ...
├── requirements.txt         # Python 의존성
├── venv/                   # 가상환경
└── README.md               # 이 파일
```

## 주요 특징

### 1. 스마트 다운로드 관리
- 이미 다운로드된 모델은 재다운로드하지 않음
- 필수 파일 존재 여부를 확인하여 완전한 다운로드 검증
- 중단된 다운로드 재개 지원

### 2. 안정성
- 네트워크 오류 시 지수 백오프 재시도
- 병렬 워커 수 제한으로 메모리/파일 디스크립터 폭주 방지
- 상세한 오류 메시지와 해결 방법 제시

### 3. 유연성
- 단축명과 전체 repo_id 모두 지원
- 사용자 정의 저장 경로 설정 가능
- 명령줄과 Python API 모두 제공

## 테스트

```bash
# 테스트 스크립트 실행
python3 huggingface/test.py
```

## 문제 해결

### 일반적인 문제들

1. **토큰 인증 오류**
   - `hf_token.txt` 파일이 올바른 위치에 있는지 확인
   - 토큰이 유효한지 확인

2. **다운로드 실패**
   - 인터넷 연결 상태 확인
   - 디스크 공간 충분한지 확인
   - 모델 라이선스 동의 여부 확인

3. **메모리 부족**
   - `max_workers` 파라미터를 더 낮게 설정
   - 시스템 메모리 확인