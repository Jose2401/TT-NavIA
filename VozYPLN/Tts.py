import asyncio
import hashlib
import os

try:
    import edge_tts
    import pygame
    _HAS_EDGE = True
except ImportError:
    _HAS_EDGE = False

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False


class Speaker:
    def __init__(
        self,
        voice="es-MX-DaliaNeural",   # alternativas
        rate="-5%",                  # un poco más pausada = más natural y clara
        pitch="+0Hz",
        volume="+0%",
        cache_dir="tts_cache",
        enabled=True,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._fallback = None

        if _HAS_EDGE:
            os.makedirs(cache_dir, exist_ok=True)
            pygame.mixer.init()
        else:
            print("[TTS] Instala 'edge-tts' y 'pygame' para una voz natural "
                  "(pip install edge-tts pygame). Usando voz del sistema.")

    #  API  
    def say(self, text):
        print(f"Sistema: {text}")
        if not self.enabled or not text:
            return
        if _HAS_EDGE:
            try:
                self._say_edge(text)
                return
            except Exception as error:
                print(f"[TTS] edge-tts falló ({error}); usando voz del sistema.")
        self._say_fallback(text)

    # edge
    def _cache_path(self, text):
        key = f"{self.voice}|{self.rate}|{self.pitch}|{self.volume}|{text}"
        name = hashlib.md5(key.encode("utf-8")).hexdigest() + ".mp3"
        return os.path.join(self.cache_dir, name)

    def _say_edge(self, text):
        path = self._cache_path(text)
        if not os.path.exists(path):
            async def synthesize():
                communicate = edge_tts.Communicate(
                    text, self.voice,
                    rate=self.rate, pitch=self.pitch, volume=self.volume,
                )
                await communicate.save(path)
            asyncio.run(synthesize())

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():   # bloquea hasta terminar de hablar
            pygame.time.wait(50)
        pygame.mixer.music.unload()

    # respaldo 
    def _say_fallback(self, text):
        if not _HAS_PYTTSX3:
            return
        if self._fallback is None:
            self._fallback = pyttsx3.init()
            self._fallback.setProperty("rate", 160)
            for v in self._fallback.getProperty("voices"):
                if "spanish" in v.name.lower() or "es" in str(v.languages).lower():
                    self._fallback.setProperty("voice", v.id)
                    break
        self._fallback.say(text)
        self._fallback.runAndWait()