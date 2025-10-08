import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Callable
import kaldi_native_fbank as knf
from dataclasses import dataclass
from enum import Enum


from .cmvn_utils import apply_cmvn
from .model_data import SenseVoiceModel, VadModel
from .rolling_buffer import RollingBuffer


@dataclass
class ASRState:
    start_frame_idx: Optional[int] = None
    last_decoded_text: str = ""
    last_processed_frame: int = -1

    def is_speech_active(self) -> bool:
        return self.start_frame_idx is not None

    def start_speech(self, frame_idx: int):
        self.start_frame_idx = frame_idx

    def end_speech(self):
        self.start_frame_idx = None
        self.last_decoded_text = ""
        self.last_processed_frame = -1


class StreamingASREventType(str, Enum):
    """Enum for audio processor event types."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    PARTIAL_RESULT = "partial_result"
    FINAL_RESULT = "final_result"
    ERROR = "error"


# Type alias for event callback
StreamingASRCallback = Callable[[StreamingASREventType, str], None]

logger = logging.getLogger(__name__)


@dataclass
class StreamingASRConfig:
    """Configuration for streaming ASR with VAD.

    All time values are in milliseconds (ms) unless noted.

    Attributes:
        lang: Language code "zh", "en", "yue", "ja", "ko" or 'auto' for auto-detection. Default: 'auto'.
        itn_min_length: Minimum character length of previous result to enable ITN for next recognition.
                             If set to 0, ITN is always enabled.
                             If set to -1, ITN is always disabled.
                             If set to positive value, ITN is enabled when previous result length >= this value. Default: -1.
        vad_start_threshold: VAD threshold to start speech (range: 0–1). Default: 0.7.
        vad_end_threshold: VAD threshold to end speech (should be < start threshold). Default: 0.3.
        vad_start_persistence_ms: Min ms above start threshold to trigger speech start. Default: 200.
        vad_end_persistence_ms: Min ms below end threshold to trigger speech end. Default: 500.
        vad_start_padding_ms: Audio (ms) to include before detected speech start. Default: 200.
        buffer_duration_sec: Max buffer duration in seconds (limits memory). Default: 600.
        asr_result_trigger_buffer_ms: Min speech length (ms) before first partial result. Default: 1000.
        asr_result_update_interval_ms: Min interval (ms) between partial results. Default: 500.
    """

    lang: str = "auto"
    itn_min_length: int = (
        10  # -1: no ITN; 0, ITN; >0: enable ITN if result length exceeded
    )
    vad_start_threshold: float = 0.6
    vad_end_threshold: float = 0.3
    vad_start_persistence_ms: int = 100
    vad_end_persistence_ms: int = 500
    vad_start_padding_ms: int = 100
    buffer_duration_sec: int = 600
    asr_result_trigger_buffer_ms: int = 1000
    asr_result_update_interval_ms: int = 500


def _range_lfr(
    buf: RollingBuffer, a: int, b: int, lfr_m: int, lfr_n: int
) -> np.ndarray:
    """
    Compute LFR features for the buffer segment [a, b].

    :param lfr_m: Window size, i.e., the number of frames concatenated per output frame.
    :param lfr_n: Step size, i.e., the number of frames to skip between output frames.
    :return: LFR features
    """
    i_left = a - lfr_m // 2
    i_right = b + lfr_m // 2 + 1
    x = buf[max(i_left, buf.head) : min(i_right, buf.tail)]
    pad_left = max(0, buf.head - i_left)
    pad_right = max(0, i_right - buf.tail)
    if pad_left > 0 or pad_right > 0:
        x = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="edge")
    raw_feat_dim = x.shape[1]  # type: ignore
    return sliding_window_view(x, (lfr_m, raw_feat_dim))[::lfr_n].reshape(
        -1, lfr_m * raw_feat_dim
    )


class SenseVoiceStreamingASR:
    # Audio config
    SAMPLE_RATE = 16000
    SAMPLES_PER_FRAME = 160  # 10ms at 16kHz
    ASR_FRAME_RATE = 100  # Hz
    MS_PER_FRAME = 10

    def ms_to_frames(self, ms: int) -> int:
        return ms // self.MS_PER_FRAME

    def frames_to_samples(self, frames: int) -> int:
        return frames * self.SAMPLES_PER_FRAME

    def __init__(
        self,
        asr_model: SenseVoiceModel,
        vad_model: VadModel,
        config: StreamingASRConfig,
    ):
        self.config = config
        self.fbank = knf.OnlineFbank(asr_model.fbank_opts)
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.on_event: StreamingASRCallback = lambda _, __: None

        self.lookback_frames = self.ms_to_frames(
            config.vad_start_persistence_ms + config.vad_start_padding_ms
        )

        self.audio_buffer = RollingBuffer(
            (),
            "float32",
            capacity=self.SAMPLE_RATE * self.config.buffer_duration_sec,
        )
        self.fbank_feature_buffer = RollingBuffer(
            (80,),
            "float32",
            capacity=self.ASR_FRAME_RATE * self.config.buffer_duration_sec,
        )
        self.speech_prob_buffer = RollingBuffer(
            (),
            "float32",
            capacity=self.ASR_FRAME_RATE * self.config.buffer_duration_sec,
        )

        lang = self.config.lang
        if self.config.lang not in self.asr_model.lid_dict:
            logger.error(
                "Language '%s' is not supported. Falling back to 'auto'. Supported languages: %s",
                lang,
                list(self.asr_model.lid_dict.keys()),
            )
            lang = "auto"
        self.lang_token_np = np.array(
            [self.asr_model.lid_dict.get(lang)], dtype=np.int32
        )

        self.vad_cache = [np.zeros((1, 128, 19, 1), dtype=np.float32) for _ in range(4)]
        self.asr_state = ASRState()

    def accept_audio(self, audio: np.ndarray[tuple[int], np.dtype[np.float32]]) -> None:
        self.audio_buffer.extend(audio)
        self._extract_features(audio)
        self._run_vad_inference_if_needed()

        # Handle VAD start detection
        speech_ended = False
        if self.asr_state.is_speech_active():
            speech_ended = self.detect_vad_speech_ended()
        else:
            if self.detect_vad_speech_start():
                self.asr_state.start_speech(
                    self.speech_prob_buffer.tail - self.lookback_frames
                )
                self.on_event(StreamingASREventType.SPEECH_START, "")

        # Handle ASR inference if speech is active
        if self.asr_state.is_speech_active():
            decoded_text = self._run_asr_inference_if_needed(force_run=speech_ended)
            if decoded_text is not None:
                if decoded_text != self.asr_state.last_decoded_text:
                    self.asr_state.last_decoded_text = decoded_text
                    self.on_event(StreamingASREventType.PARTIAL_RESULT, decoded_text)

        # Handle VAD end detection
        if speech_ended:
            self.on_event(
                StreamingASREventType.FINAL_RESULT, self.asr_state.last_decoded_text
            )
            self.on_event(StreamingASREventType.SPEECH_END, "")
            self.asr_state.end_speech()

        self.move_buffer_head()

    def move_buffer_head(self):
        vad_min_frame_i = (
            self.speech_prob_buffer.tail
            - self.vad_model.lfr_m // 2
            - self.lookback_frames
        )
        frame_i = vad_min_frame_i
        if self.asr_state.is_speech_active():
            assert self.asr_state.start_frame_idx is not None
            asr_min_frame_i = (
                self.asr_state.start_frame_idx
                - self.asr_model.lfr_m // 2
                - self.lookback_frames
            )
            frame_i = min(asr_min_frame_i, vad_min_frame_i)

        self.audio_buffer.drop_buffer_before(self.frames_to_samples(frame_i))
        self.fbank_feature_buffer.drop_buffer_before(frame_i)
        self.speech_prob_buffer.drop_buffer_before(frame_i)

    def finalize_utterance(self):
        if self.asr_state.is_speech_active():
            self.on_event(
                StreamingASREventType.FINAL_RESULT, self.asr_state.last_decoded_text
            )
            self.on_event(StreamingASREventType.SPEECH_END, "")
            self.asr_state.end_speech()

    def set_on_event_callback(self, on_event_callback: StreamingASRCallback):
        self.on_event = on_event_callback

    def _run_vad_inference_if_needed(self):
        """Run VAD inference if there are new features to process"""
        vad_feat_i_min = self.speech_prob_buffer.tail
        vad_feat_i_max = self.fbank_feature_buffer.tail - self.vad_model.lfr_m // 2
        if vad_feat_i_max >= vad_feat_i_min:
            self._run_vad_inference(vad_feat_i_min, vad_feat_i_max)

    def _run_asr_inference_if_needed(self, force_run=False) -> Optional[str]:
        """Run ASR inference if conditions are met. Returns (decoded_text) or None."""
        if not self.asr_state.is_speech_active():
            return None
        assert self.asr_state.start_frame_idx is not None

        try:
            asr_i_min: int = self.asr_state.start_frame_idx
            asr_i_max = self.fbank_feature_buffer.tail - self.asr_model.lfr_m // 2

            if force_run or self._should_update_asr_result(asr_i_min, asr_i_max):
                self.asr_state.last_processed_frame = asr_i_max
                decoded_text = self._run_asr_inference(asr_i_min, asr_i_max)
                return decoded_text

            return None
        except Exception as e:
            logger.error(f"Error in ASR inference: {e}", exc_info=True)
            self.on_event(StreamingASREventType.ERROR, str(e))
            return None

    def _should_update_asr_result(self, asr_i_min: int, asr_i_max: int) -> bool:
        """Determine if ASR inference should be run"""
        if asr_i_max < asr_i_min:
            return False

        is_pending_result = asr_i_max - asr_i_min > self.ms_to_frames(
            self.config.asr_result_trigger_buffer_ms
        ) and asr_i_max - self.asr_state.last_processed_frame > self.ms_to_frames(
            self.config.asr_result_update_interval_ms
        )

        return is_pending_result

    def detect_vad_speech_start(self) -> bool:
        """
        Detect if speech has started based on VAD probabilities.

        Returns:
            bool: True if speech should start, False otherwise
        """
        persistence_frames = self.ms_to_frames(self.config.vad_start_persistence_ms)
        if self.speech_prob_buffer.count < persistence_frames:
            return False
        last_n_probs = self.speech_prob_buffer.get_all()[-persistence_frames:]
        return all(p > self.config.vad_start_threshold for p in last_n_probs)

    def detect_vad_speech_ended(self) -> bool:
        """
        Detect if speech has ended based on VAD probabilities.

        Returns:
            bool: True if speech should stop, False otherwise
        """
        persistence_frames = self.ms_to_frames(self.config.vad_end_persistence_ms)
        if self.speech_prob_buffer.count < persistence_frames:
            return False
        last_n_probs = self.speech_prob_buffer.get_all()[-persistence_frames:]
        return all(p < self.config.vad_end_threshold for p in last_n_probs)

    def _extract_features(self, audio: np.ndarray[tuple[int], np.dtype[np.float32]]):
        self.fbank.accept_waveform(self.SAMPLE_RATE, audio.tolist())
        pop_count = self.fbank.num_frames_ready - self.fbank_feature_buffer.tail
        for i in range(self.fbank_feature_buffer.tail, self.fbank.num_frames_ready):
            self.fbank_feature_buffer.append(self.fbank.get_frame(i))
        self.fbank.pop(pop_count)

    def _text_norm(self):
        ITN_TOKEN = 14
        NO_ITN_TOKEN = 15
        if self.config.itn_min_length < 0:
            return np.array([NO_ITN_TOKEN])
        if self.config.itn_min_length == 0:
            return np.array([ITN_TOKEN])
        enable_itn = len(self.asr_state.last_decoded_text) > self.config.itn_min_length
        return np.array([ITN_TOKEN if enable_itn else NO_ITN_TOKEN], dtype=np.int32)

    def _run_asr_inference(self, start_frame: int, end_frame: int) -> str:
        """Run ASR on [start_frame, end_frame], return decoded text."""
        asr_lfr_feat = _range_lfr(
            self.fbank_feature_buffer,
            start_frame,
            end_frame,
            self.asr_model.lfr_m,
            self.asr_model.lfr_n,
        )
        asr_feat = apply_cmvn(asr_lfr_feat, self.asr_model.cmvn)
        input_feed = {
            "speech": np.expand_dims(asr_feat, axis=0),
            "speech_lengths": np.array([asr_feat.shape[0]], dtype=np.int32),
            "language": self.lang_token_np,
            "textnorm": self._text_norm(),
        }
        logits = self.asr_model.model_inference_session.run(None, input_feed)[0]
        decoded = self.asr_model.decode(logits)[0]
        return decoded

    def _run_vad_inference(
        self, start_frame: int, end_frame: int
    ) -> Optional[np.ndarray]:
        """Run VAD inference on [start_frame, end_frame], update cache, return speech probs."""
        if end_frame < start_frame:
            return None

        vad_lfr_feat = _range_lfr(
            self.fbank_feature_buffer,
            start_frame,
            end_frame,
            self.vad_model.lfr_m,
            self.vad_model.lfr_n,
        )
        vad_feat = apply_cmvn(vad_lfr_feat, self.vad_model.cmvn)
        results = self.vad_model.model_inference_session.run(
            None,
            {
                "speech": np.expand_dims(vad_feat, axis=0),
                "in_cache0": self.vad_cache[0],
                "in_cache1": self.vad_cache[1],
                "in_cache2": self.vad_cache[2],
                "in_cache3": self.vad_cache[3],
            },
        )
        logits, *self.vad_cache = results
        speech_probs = 1.0 - logits[0, :, 0]  # type: ignore
        self.speech_prob_buffer.extend(speech_probs)
        return speech_probs
