# Hugging Face 모델 다운로드 및 관리 도구

Hugging Face Hub에서 LLM 모델을 다운로드하고 로컬에서 관리하는 간단한 도구입니다.

## 주요 기능

- **모델 자동 다운로드**: Hugging Face Hub에서 모델을 자동으로 다운로드
- **중복 다운로드 방지**: 이미 다운로드된 모델은 재다운로드하지 않음
- **재시도 메커니즘**: 네트워크 오류 시 자동 재시도
- **토큰 관리**: Hugging Face 인증 토큰 자동 처리
- **다양한 모델 지원**: Gemma, Llama, GPT-OSS, Qwen 등 주요 모델 지원


### 경로 설정 

**⚠️ 중요**: 로컬 시스템에서 사용할 때는 다음 경로들을 사용자 환경에 맞게 수정 필요:

- **모델 저장 경로**: 코드 내 기본 경로를 충분한 디스크 공간이 있는 경로로 변경
- **토큰 파일 경로**: `hf_token.txt` 파일 위치를 사용자 환경에 맞게 조정
- **스크립트 경로**: `download_models.py` 실행 시 경로를 사용자 환경에 맞게 수정

### 1. 토큰 설정
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

### 3. 모델 다운로드
```bash
# 특정 모델 다운로드
python3 huggingface/download_models.py --model gemma

# 지원되는 모델 목록 확인
python3 huggingface/download_models.py --list-models
```

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
└── README.md               # 이 파일
```

## 주요 특징

- ✅ 중복 다운로드 방지
- ✅ 자동 재시도 (네트워크 오류 시)
- ✅ 이어받기 지원
- ✅ 메모리 최적화

## 문제 해결

- **토큰 오류**: `hf_token.txt` 파일 위치 및 토큰 유효성 확인
- **다운로드 실패**: 인터넷 연결, 디스크 공간, 모델 라이선스 동의 확인
- **메모리 부족**: 시스템 메모리 확인 및 충분한 여유 공간 확보