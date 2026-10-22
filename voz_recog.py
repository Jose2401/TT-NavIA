import json
import os
import queue
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel


class VoiceRecognizer:
    #   Reconocimiento de voz offline 

    def __init__(self, model_path, sample_rate=16000, device=None, blocksize=4000):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo Vosk en '{model_path}'.   https://alphacephei.com/vosk/models"
                
            )

        SetLogLevel(-1)  # silencia los logs de Kaldi

        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize  # 4000 muestras = 0.25 s a 16 kHz

        self.audio_queue = queue.Queue()
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

    @staticmethod
    def list_devices():
        print(sd.query_devices())

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] {status}")
        self.audio_queue.put(bytes(indata))

    def _flush_queue(self):
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    def listen(self, timeout=None):
        # Limpia la cola de audio y reinicia el reconocedor
        self._flush_queue()
        self.recognizer.Reset()

        print("\nEscuchando...")
        start = time.monotonic()

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
        ):
            while True:
                if timeout is not None and time.monotonic() - start > timeout:
                    return ""

                try:
                    data = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"Usuario: {text}")
                        return text

    def reset(self):
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)