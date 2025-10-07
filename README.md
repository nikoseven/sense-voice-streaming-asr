# 🎙️ SenseVoice Streaming ASR

[![PyPI](https://img.shields.io/pypi/v/sense-voice-streaming-asr)](https://pypi.org/project/sense-voice-streaming-asr/)
[![Python](https://img.shields.io/pypi/pyversions/sense-voice-streaming-asr)](https://pypi.org/project/sense-voice-streaming-asr/)
[![License](https://img.shields.io/pypi/l/sense-voice-streaming-asr)](LICENSE)

A lightweight, real-time streaming speech recognition engine powered by SenseVoiceSmall.


## ✨ Features

- **Streaming ASR**: lightweight, real-time streaming speech recognition engine, runs on CPU with realtime recongnition.
- **Integrated VAD**: Built-in Voice Activity Detection (VAD) to detect speech segments.
- **Multilingual**: Supports `Chinese`, `English`, `Japanese`, `Korean`, `Cantonese` out of the box.
- **Self-contained**: Models bundled inside the package — no external downloads needed.
- **Pure Python + ONNX**: No heavy dependencies; runs on CPU.

## 🚀 Installation

```bash
pip install sense-voice-streaming-asr
```

> Requires Python ≥ 3.8.



## 🛠️ Development

To install in development mode:

```bash
git clone https://github.com/yourname/sense-voice-streaming-asr.git
cd sense-voice-streaming-asr
git submodule init # for SenseVoice models
pip install -e .
```


## 📄 License

Apache 2.0 License.

This project incorporates code from [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) which is licensed under Apache 2.0. The project as a whole is therefore distributed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for full license text.

## 🙏 Acknowledgements

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) by FunAudioLLM
- [SenseVoice ONNX Models](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- [FSMN VAD](https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-onnx)
- [Kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
- ONNX Runtime
