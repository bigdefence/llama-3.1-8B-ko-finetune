# 🚀 LLaMA-3.1-8B 한국어 LLM 파인튜닝 

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%2B-red)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Latest-green)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*한국어 LLM의 힘을 극대화하세요! 🇰🇷✨*
## 파인튜닝한 모델은 Hugging Face [bigdefence/Llama-3.1-8B-Ko-bigdefence](https://huggingface.co/bigdefence/Llama-3.1-8B-Ko-bigdefence)에서 사용해 보실 수 있습니다.

## 📚 목차
- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [시작하기](#-시작하기)
- [파일 구조](#-파일-구조)
- [사용 방법](#-사용-방법)
- [성능 분석](#-성능-분석)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)
- [연락처](#-연락처)

## 🌟 프로젝트 개요

이 프로젝트는 최첨단 Unsloth 라이브러리를 활용하여 한국어 LLM(Large Language Model)을 놀라운 속도로 파인튜닝합니다. meta-llama/Meta-Llama-3.1-8B 모델을 기반으로, 우리는 한국어 자연어 처리의 새로운 지평을 열어갑니다.

## 🎯 주요 기능

- 🚄 **초고속 학습**: Unsloth의 최적화 기술로 학습 시간을 대폭 단축
- 🧠 **고급 모델 사용**: meta-llama/Meta-Llama-3.1-8B 모델 적용
- 🛠 **LoRA 기법**: 효율적인 파인튜닝을 위한 Low-Rank Adaptation 활용
- 📊 **커스텀 데이터셋**: 특화된 한국어 데이터로 정확도 향상
- 🌐 **간편한 배포**: Hugging Face Hub 연동으로 손쉬운 모델 공유
## 📜 데이터셋 
- MarkrAI/KoCommercial-Dataset(HugginFace)

## 🚀 시작하기

### 필수 조건

- Python 3.7+
- CUDA 지원 GPU
- 열정과 호기심 😉

### 설치 과정

```bash
# Unsloth 및 의존성 설치
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 추가 라이브러리 설치
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
```

## 💻 사용 방법

1. **데이터 준비**
   ```bash
   from datasets import load_dataset
   dataset = load_dataset("MarkrAI/KoCommercial-Dataset", split = "train")
   ```

2. **모델 학습**
   ```bash
   trainer.train()
   ```

3. **모델 배포**
   ```python
   model.push_to_hub_merged(
       "your-repo-name",
       tokenizer,
       save_method="merged_16bit",
       token="your-huggingface-token"
   )
   ```

## 📊 성능 분석

| 메트릭 | 값 |
|--------|------|
| 학습 시간 | 7 분 |
| 최대 메모리 사용량 | 6.0 GB |


## 🤝 기여하기

여러분의 기여를 환영합니다! 다음 단계를 따라주세요:

1. 프로젝트를 Fork합니다.
2. 새 Branch를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3. 변경사항을 Commit합니다 (`git commit -m 'Add some AmazingFeature'`).
4. Branch에 Push합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 생성합니다.

## 📞 연락처

프로젝트 관리자 - bigdefence@naver.com

프로젝트 링크: [[https://github.com/bigdefence/unsloth-finetune](https://github.com/bigdefence/llama-3.1-8B-ko-finetune/tree/main)]

---

  <sub>Built with ❤️ by [정강빈] and contributors</sub>

