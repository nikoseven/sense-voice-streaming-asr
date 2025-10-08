import logging
import sounddevice as sd
import sys
import numpy as np
from sense_voice_streaming_asr.sense_voice_streaming_asr import (
    SenseVoiceStreamingASR,
    StreamingASRConfig,
)
from sense_voice_streaming_asr.model_data import SenseVoiceModel, VadModel


def main():
    """Simple example of using the SenseVoice streaming ASR."""
    processor = None

    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # 1. Load models
        print("Loading models...")
        vad_model = VadModel()
        asr_model = SenseVoiceModel()
        print("Models loaded successfully.")

        # 2. Setup audio stream
        sample_rate = 16000
        blocksize = int(sample_rate * 500 / 1000)  # 500ms buffer

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
        )

        # 3. Initialize the streaming ASR processor
        config = StreamingASRConfig()
        processor = SenseVoiceStreamingASR(
            asr_model=asr_model,
            vad_model=vad_model,
            config=config,
        )

        # Set up event callback to handle results
        def on_event(event_type: str, message: str):
            if event_type == "speech_start":
                print("> speech started")
            elif event_type == "speech_end":
                print("> speech ended")
            elif event_type == "partial_result":
                sys.stdout.write(f"\rPartial: {message}")
                sys.stdout.flush()
            elif event_type in ["RECOGNITION_FINAL_RESULT", "final_result"]:
                print(f"\nFinal: {message}")
            elif event_type == "error":
                print(f"Error: {message}")

        processor.set_on_event_callback(on_event)

        # 4. Start streaming
        print("Starting audio stream...")
        print("Speak into your microphone. Press Ctrl+C to stop.")

        with stream:
            stream.start()

            while True:
                # Read available audio frames
                read_frame_num = min(
                    stream.read_available,
                    processor.audio_buffer.space_left,
                )

                if read_frame_num > 0:
                    frames, overflowed = stream.read(read_frame_num)
                    if overflowed:
                        print("Warning: Audio overflow detected")

                    audio = np.squeeze(frames).astype(np.float32).reshape(-1)
                    processor.accept_audio(audio)

                # Add small delay to prevent excessive CPU usage
                sd.sleep(10)

    except KeyboardInterrupt:
        print("\\nStopping...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if processor:
            processor.finalize_utterance()
            print("Processor stopped.")


if __name__ == "__main__":
    main()
