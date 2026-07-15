# Environment Record

## Author-code source

- Repository: <https://github.com/IINemo/lm-polygraph>
- Reproduction package: `lm-polygraph==0.5.0`
- Paper: Vashurin et al., TACL 2025

## Pilot environment

- Platform: Apple silicon, macOS
- Python: 3.12
- PyTorch: 2.10.0
- LM-Polygraph: 0.5.0
- Transformers: 4.50.0
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, run in text-only mode
- Device: MPS when available, otherwise CPU
- Generation: greedy decoding, at most 32 new tokens

The environment was created outside the repository. LM-Polygraph 0.5.0 declares
several benchmark, translation, service, and optional-model dependencies in its
base installation. For this pilot, the minimum imports required by the official
white-box estimator path were installed; `bitsandbytes` and `unbabel-comet`
were not required by the executed estimators.

## Suggested setup

```bash
python -m venv --system-site-packages /tmp/bayes-lab-part0-venv
source /tmp/bayes-lab-part0-venv/bin/activate
pip install lm-polygraph==0.5.0 --no-deps
pip install \
  'transformers==4.50.0' 'datasets>=2.19,<4' \
  openai hydra-core sentence-transformers rouge-score sacrebleu evaluate \
  diskcache wget bs4 pytreebank nlpaug hf-lfs bert-score fastchat fschat \
  'spacy>=3.4,<3.8'
python part0_reproductions/01_uncertainty_signals/experiments/run_lm_polygraph_signal_pilot.py
```

The model weights are not stored in this repository. Hugging Face access and
approximately 7 GB of local model storage are required for the default model.
