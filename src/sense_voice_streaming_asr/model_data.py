from importlib import resources

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime
import json

from .cmvn_utils import load_cmvn

# Define package paths for models
VAD_MODEL_PACKAGE = (
    "sense_voice_streaming_asr.models.speech_fsmn_vad_zh-cn-16k-common-onnx"
)
SENSEVOICE_MODEL_PACKAGE = "sense_voice_streaming_asr.models.SenseVoiceSmall"

# Create context managers for each model file path
VAD_CMVN_PATH = resources.path(VAD_MODEL_PACKAGE, "am.mvn")
VAD_MODEL_PATH = resources.path(VAD_MODEL_PACKAGE, "model_quant.onnx")

SENSEVOICE_CMVN_PATH = resources.path(SENSEVOICE_MODEL_PACKAGE, "am.mvn")
SENSEVOICE_TOKENS_PATH = resources.path(SENSEVOICE_MODEL_PACKAGE, "tokens.json")
SENSEVOICE_MODEL_PATH = resources.path(SENSEVOICE_MODEL_PACKAGE, "model_quant.onnx")


class VadModel:
    def __init__(self):
        with VAD_CMVN_PATH as cmvn_path, VAD_MODEL_PATH as model_path:
            self.cmvn = load_cmvn(str(cmvn_path))
            self.model_inference_session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        self.fbank_opts = knf.FbankOptions()
        self.fbank_opts.frame_opts.samp_freq = 16000
        self.fbank_opts.frame_opts.dither = 0.0
        self.fbank_opts.frame_opts.window_type = "hamming"
        self.fbank_opts.frame_opts.frame_shift_ms = 10
        self.fbank_opts.frame_opts.frame_length_ms = 25
        self.fbank_opts.mel_opts.num_bins = 80
        self.fbank_opts.energy_floor = 0
        self.fbank_opts.frame_opts.snip_edges = True
        self.fbank_opts.mel_opts.debug_mel = False
        self.lfr_m = 5
        self.lfr_n = 1
        self.vad_cache = [np.zeros((1, 128, 19, 1), dtype=np.float32) for _ in range(4)]


class SenseVoiceModel:
    def __init__(self, use_cuda=False):
        with (
            SENSEVOICE_CMVN_PATH as cmvn_path,
            SENSEVOICE_TOKENS_PATH as tokens_json_path,
            SENSEVOICE_MODEL_PATH as model_path,
        ):
            self.cmvn = load_cmvn(str(cmvn_path))
            self.sensevoice_tokens = json.load(open(tokens_json_path))

            execuction_providers = ["CPUExecutionProvider"]
            if use_cuda:
                execuction_providers.append("CUDAExecutionProvider")
            self.model_inference_session = onnxruntime.InferenceSession(
                str(model_path), providers=execuction_providers
            )

        self.fbank_opts = knf.FbankOptions()
        self.fbank_opts.frame_opts.samp_freq = 16000
        self.fbank_opts.frame_opts.dither = 0.0
        self.fbank_opts.frame_opts.window_type = "hamming"
        self.fbank_opts.frame_opts.frame_shift_ms = 10
        self.fbank_opts.frame_opts.frame_length_ms = 25
        self.fbank_opts.mel_opts.num_bins = 80
        self.fbank_opts.energy_floor = 0
        self.fbank_opts.frame_opts.snip_edges = True
        self.fbank_opts.mel_opts.debug_mel = False
        self.lfr_m = 7
        self.lfr_n = 6
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }

    def decode(self, logits):
        """
        Decodes logits to text with CTC post-processing.
        Args:
            logits: A numpy array of shape (batch_size, sequence_length, vocab_size).
        Returns:
            A list of decoded strings.
        """
        predictions = np.argmax(logits, axis=-1)

        decoded_texts = []
        for pred in predictions:
            # Merge repeated tokens
            merged = [p for i, p in enumerate(pred) if i == 0 or p != pred[i - 1]]
            parts = []
            for token_id in merged:
                if token_id > len(self.sensevoice_tokens):
                    continue
                piece = self.sensevoice_tokens[token_id]
                if not piece.startswith("<|") and piece != "<unk>":
                    parts.append(piece)
            # 3. Decode to text
            decoded_texts.append("".join(parts).replace("▁", " "))
        return decoded_texts
