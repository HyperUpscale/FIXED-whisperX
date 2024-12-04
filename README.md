# WhisperX: Time-Accurate Speech Transcription

![WhisperX Pipeline](figures/pipeline.png)

## Overview
WhisperX is an advanced speech transcription tool designed for accurate, time-aligned transcription of long-form audio.

## System Requirements
- Ubuntu 22.04
- Python 3.10
- CUDA 11.8
- CUDNN 9.1

## Installation

### Prerequisites
Install CUDNN and ffmpeg:
```bash
sudo apt-get install cudnn ffmpeg
```

### Environment Setup

#### Option 1: Conda Environment
```bash
conda env update -f cuda-ubuntu22.04-python3.10.yml -n whisperx-web-ui
```

#### Option 2: UV (Faster Python Environment Manager)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv whisperx-uv-env --python 3.10
source whisperx-uv-env/bin/activate

# Install dependencies
uv pip install -r requirements.txt
pip install ctranslate2==4.4.0

# Run the web application
cd web
python app.py
```

## Technology Stack
- **Python**: 3.10
- **CUDA**: 11.8
- **PyTorch Ecosystem**:
  - PyTorch 2.0.0
  - TorchAudio 2.0.0
  - Additional torch 1.10.0+cu102 for pyannote.audio compatibility

## Critical Dependencies
- numpy < 2.0 (compatibility)
- pyannote.audio 0.0.1 (specific version)
- WhisperX and its dependencies

## UI and Processing
- Gradio and Streamlit for UI
- FFmpeg for audio processing

## Citation
If you use WhisperX in your research, please cite:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]
