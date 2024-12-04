


<img width="1216" align="center" alt="whisperx-arch" src="figures/pipeline.png">


### 1. Create Python3.10 environment -  Install PyTorch on UBUNTU 22.04 with CUDA11.8:

'conda env update -f cuda-ubuntu22.04-python3.10.yml -n whisperx-web-ui'


Base CUDA and Python setup:
Python 3.10
CUDA 11.8
CUDNN 9.1
PyTorch ecosystem:
PyTorch 2.0.0
TorchAudio 2.0.0
Additional torch 1.10.0+cu102 for pyannote.audio compatibility
Critical dependencies:
numpy < 2.0 (to avoid compatibility issues)
pyannote.audio 0.0.1 (specific version required)
whisperx and its dependencies
UI and processing:
gradio and streamlit for UI
ffmpeg for audio processing


TRY ALTERNATIVELY with UV (Python environment manager, similar to Conda, just Many times faster):

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv whisperx-uv-env --python 3.10
source whisperx-uv-env/bin/activate
uv pip install -r requirements.txt
pip install ctranslate2==4.4.0

cd web
python app.py


```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
