"""Alertas de obstáculos repentinos (carro, moto, etc.) desde sensores, visión o ambos.

- report() es seguro para llamarse desde cualquier hilo (sensores, visión, ...).
- El pitido suena de inmediato en un hilo propio, aunque el programa principal
  esté escuchando o hablando.
- El pitido es estéreo: izquierda / derecha / frente / atrás se distinguen por
  el canal (y el tono, en el caso de atrás).
- Si sensor y visión reportan lo mismo en la misma dirección casi a la vez,
  se fusionan ("ambos") y la alerta sube un nivel.

Prueba rápida sin voz:  python alertas.py
"""
import math
import queue
import threading
import time
from array import array
from dataclasses import dataclass

import sounddevice as sd

from PLN_oneimport import normalize_text

# ---------------- niveles ----------------
INFO, WARNING, CRITICAL = 1, 2, 3

# ---------------- parámetros ajustables ----------------
SAMPLE_RATE = 44100
COOLDOWN_S = 2.5          # no repetir la misma alerta (misma dirección) antes de esto...
FUSION_WINDOW_S = 1.0     # sensor + visión dentro de esta ventana = "ambos"

VEHICLE_CRITICAL_M = 5.0          # vehículo a esta distancia o menos -> crítico
VEHICLE_APPROACHING_CRITICAL_M = 8.0   # vehículo que se acerca
OBSTACLE_CRITICAL_M = 1.0         # cualquier obstáculo a esta distancia -> crítico
OBSTACLE_WARNING_M = 2.5

VEHICLES = {"carro", "moto", "camion", "autobus", "bicicleta", "vehiculo"}

TYPE_ALIASES = {
    "auto": "carro", "coche": "carro", "automovil": "carro", "car": "carro",
    "motocicleta": "moto", "motorcycle": "moto",
    "truck": "camion", "bus": "autobus", "microbus": "autobus",
    "bicycle": "bicicleta", "bici": "bicicleta",
    "person": "persona", "dog": "perro",
    "vehicle": "vehiculo",
}

DIRECTION_ALIASES = {
    "frente": "frente", "adelante": "frente", "enfrente": "frente", "centro": "frente",
    "izquierda": "izquierda", "izq": "izquierda", "left": "izquierda",
    "derecha": "derecha", "der": "derecha", "right": "derecha",
    "atras": "atras", "detras": "atras", "back": "atras",
}

# (frecuencia Hz, ms encendido, ms silencio)
PATTERNS = {
    CRITICAL: [(2100, 110, 50), (1500, 110, 50), (2100, 110, 50), (1500, 110, 0)],
    WARNING: [(1200, 180, 110), (1200, 180, 0)],
    INFO: [(900, 200, 0)],
}

# ganancia (izquierda, derecha) por dirección
PAN = {
    "izquierda": (1.0, 0.2),
    "derecha": (0.2, 1.0),
    "frente": (1.0, 1.0),
    "atras": (1.0, 1.0),
}


# ---------------- utilidades ----------------
def normalize_type(tipo):
    text = normalize_text(tipo) if tipo else ""
    if not text:
        return "obstaculo"
    return TYPE_ALIASES.get(text, text)


def normalize_direction(direccion):
    text = normalize_text(direccion) if direccion else ""
    return DIRECTION_ALIASES.get(text, "frente")


def normalize_source(fuente):
    text = normalize_text(fuente) if fuente else ""
    return text if text in ("sensor", "vision", "ambos") else "sensor"


def compute_level(tipo, distancia_m, se_acerca, fuente):
    vehicle = tipo in VEHICLES

    if distancia_m is None:
        if vehicle and se_acerca:
            level = CRITICAL
        else:
            level = WARNING
    elif vehicle:
        if distancia_m <= VEHICLE_CRITICAL_M or (
            se_acerca and distancia_m <= VEHICLE_APPROACHING_CRITICAL_M
        ):
            level = CRITICAL
        else:
            level = WARNING
    else:
        if distancia_m <= OBSTACLE_CRITICAL_M:
            level = CRITICAL
        elif distancia_m <= OBSTACLE_WARNING_M:
            level = WARNING
        else:
            level = INFO

    if fuente == "ambos" and level < CRITICAL:
        level += 1          # corroborado por dos canales
    return level


def _tone(freq, ms, left_gain, right_gain, amplitude=0.8):
    n = int(SAMPLE_RATE * ms / 1000)
    fade = int(SAMPLE_RATE * 0.005)      # 5 ms de rampa: evita "clics"
    samples = array("h")
    for i in range(n):
        envelope = 1.0
        if i < fade:
            envelope = i / fade
        elif i > n - fade:
            envelope = (n - i) / fade
        value = math.sin(2 * math.pi * freq * i / SAMPLE_RATE) * amplitude * envelope
        value = int(value * 32767)
        samples.append(int(value * left_gain))
        samples.append(int(value * right_gain))
    return samples


def _silence(ms):
    return array("h", [0]) * (int(SAMPLE_RATE * ms / 1000) * 2)


@dataclass
class ObstacleEvent:
    tipo: str
    distancia_m: float
    direccion: str
    fuente: str
    se_acerca: bool
    nivel: int
    t: float


# ---------------- gestor ----------------
class AlertManager:
    def __init__(self, on_interrupt=None, on_speak=None, device=None):
        """
        on_interrupt(): se llama al inicio de una alerta WARNING+ (para callar la voz).
        on_speak(event): se llama después del pitido (para decir el aviso hablado).
        """
        self.on_interrupt = on_interrupt
        self.on_speak = on_speak
        self.device = device

        self.active = threading.Event()   # activo mientras se atiende una alerta
        self.last_end = 0.0

        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._recent = {}        # direccion -> (t, fuente, tipo, distancia, se_acerca)  [crudo]
        self._last_alert = {}    # direccion -> (t, nivel)
        self._buffers = self._build_buffers()
        self._running = False
        self._thread = None

    # ----- ciclo de vida -----
    def start(self):
        if self._thread:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)

    # ----- API para sensores / visión -----
    def report(self, tipo=None, distancia_m=None, direccion="frente",
               fuente="sensor", se_acerca=False):
        """Llamar apenas llegue una señal. Devuelve el evento o None si se descartó."""
        now = time.monotonic()
        tipo = normalize_type(tipo)
        direccion = normalize_direction(direccion)
        raw_source = normalize_source(fuente)
        raw = (now, raw_source, tipo, distancia_m, se_acerca)

        with self._lock:
            merged_type, merged_dist = tipo, distancia_m
            merged_close, source = se_acerca, raw_source

            prev = self._recent.get(direccion)
            if (prev and now - prev[0] <= FUSION_WINDOW_S
                    and prev[1] != raw_source and raw_source != "ambos"):
                _, _, p_type, p_dist, p_close = prev
                if merged_type == "obstaculo":
                    merged_type = p_type
                known = [d for d in (merged_dist, p_dist) if d is not None]
                merged_dist = min(known) if known else None
                merged_close = merged_close or p_close
                source = "ambos"

            self._recent[direccion] = raw

            level = compute_level(merged_type, merged_dist, merged_close, source)
            last = self._last_alert.get(direccion)
            if last and now - last[0] < COOLDOWN_S and level <= last[1]:
                return None      # misma alerta repetida: no spamear
            self._last_alert[direccion] = (now, level)

        event = ObstacleEvent(merged_type, merged_dist, direccion, source,
                              merged_close, level, now)
        self._queue.put(event)
        return event

    # ----- internos -----
    def _build_buffers(self):
        buffers = {}
        for level, pattern in PATTERNS.items():
            for direction, (left, right) in PAN.items():
                factor = 0.7 if direction == "atras" else 1.0   # atrás = tono más grave
                buf = array("h")
                for freq, on_ms, off_ms in pattern:
                    buf.extend(_tone(freq * factor, on_ms, left, right))
                    if off_ms:
                        buf.extend(_silence(off_ms))
                buffers[(level, direction)] = buf.tobytes()
        return buffers

    def _play(self, data):
        try:
            with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=2,
                                    dtype="int16", device=self.device,
                                    latency="low") as stream:
                stream.write(data)
        except Exception as error:
            print(f"[ALERTA] No se pudo reproducir el pitido: {error}")
            try:
                import winsound
                winsound.Beep(1500, 300)
            except Exception:
                print("\a")

    def _worker(self):
        while self._running:
            event = self._queue.get()
            if event is None:
                break

            # Si se acumularon varias, se atiende solo la más grave.
            pending = [event]
            while True:
                try:
                    extra = self._queue.get_nowait()
                except queue.Empty:
                    break
                if extra is not None:
                    pending.append(extra)
            event = max(pending, key=lambda e: e.nivel)

            self.active.set()
            try:
                print(f"[ALERTA] nivel={event.nivel} {event.tipo} "
                      f"({event.direccion}, {event.fuente}, dist={event.distancia_m})")
                if self.on_interrupt and event.nivel >= WARNING:
                    self.on_interrupt()
                self._play(self._buffers[(event.nivel, event.direccion)])
                if self.on_speak:
                    self.on_speak(event)
            except Exception as error:
                print(f"[ALERTA] Error atendiendo alerta: {error}")
            finally:
                self.last_end = time.monotonic()
                self.active.clear()

    def recently_active(self, seconds=2.0):
        return self.active.is_set() or (time.monotonic() - self.last_end) < seconds


# ---------------- prueba sin voz ----------------
if __name__ == "__main__":
    manager = AlertManager(on_speak=lambda e: print("  (aquí iría el aviso hablado)"))
    manager.start()

    print("1) Obstáculo lejano por sensor (INFO, frente)")
    manager.report(None, 4.0, "frente", "sensor")
    time.sleep(2.5)

    print("2) Persona cerca a la derecha por visión (WARNING)")
    manager.report("person", 2.0, "derecha", "vision")
    time.sleep(2.5)

    print("3) Carro por la izquierda: sensor y visión casi a la vez (CRITICAL)")
    manager.report(None, 3.0, "izquierda", "sensor", se_acerca=True)
    manager.report("car", None, "izquierda", "vision", se_acerca=True)
    time.sleep(3)

    print("4) Moto que se acerca por atrás (CRITICAL, tono grave)")
    manager.report("moto", 6.0, "atras", "vision", se_acerca=True)
    time.sleep(3)

    manager.stop()