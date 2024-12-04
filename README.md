


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



If you don't have access to your own GPUs, use the links above to try out WhisperX. 

<h2 align="left" id="whisper-mod">Technical Details 👷‍♂️</h2>

For specific details on the batching and alignment, the effect of VAD, as well as the chosen alignment model, see the preprint [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory requirements, try any of the following (2. & 3. can affect quality):
1.  reduce batch size, e.g. `--batch_size 4`
2. use a smaller ASR model `--model base`
3. Use lighter compute type `--compute_type int8`

Transcription differences from openai's whisper:
1. Transcription without timestamps. To enable single pass batching, whisper inference is performed `--without_timestamps True`, this ensures 1 forward pass per sample in the batch. However, this can cause discrepancies the default whisper output.
2. VAD-based segment transcription, unlike the buffered transcription of openai's. In Wthe WhisperX paper we show this reduces WER, and enables accurate batched inference
3.  `--condition_on_prev_text` is set to `False` by default (reduces hallucination)





<a href="https://www.buymeacoffee.com/maxhbain" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


<h2 align="left" id="acks">Acknowledgements 🙏</h2>

This work, and my PhD, is supported by the [VGG (Visual Geometry Group)](https://www.robots.ox.ac.uk/~vgg/) and the University of Oxford.

Of course, this is builds on [openAI's whisper](https://github.com/openai/whisper).
Borrows important alignment code from [PyTorch tutorial on forced alignment](https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html)
And uses the wonderful pyannote VAD / Diarization https://github.com/pyannote/pyannote-audio


Valuable VAD & Diarization Models from [pyannote audio][https://github.com/pyannote/pyannote-audio]

Great backend from [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)

Those who have [supported this work financially](https://www.buymeacoffee.com/maxhbain) 🙏

Finally, thanks to the OS [contributors](https://github.com/m-bain/whisperX/graphs/contributors) of this project, keeping it going and identifying bugs.

<h2 align="left" id="cite">Citation</h2>
If you use this in your research, please cite the paper:

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
