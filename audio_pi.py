"""
audio_pi.py — deja listo el audio de la pi SIN sudo.

medido en la pi 5 (2026-11-10): la salida por pygame (SDL -> pipewire)
es estable (un pitido de 0.3 s tarda 0.30 s); en cambio sounddevice
sobre alsa tardaba entre 3 y 49 s en sacar el mismo pitido. por eso en
la pi la voz y los pitidos salen por pygame y el microfono entra por
pw-record (pipewire nativo, remuestrea solo a 16 kHz). sounddevice
queda de respaldo (laptop) y pq los modulos de VozYPLN lo importan:

  - portaudio: la pi no lo trae y no hay sudo; instalar_pi.sh lo baja
    con `apt-get download` a ~/.local/lib/navia y aqui se parcha
    ctypes.util.find_library pa que sounddevice lo encuentre
  - la diadema usb: si hay una conectada se pone como salida y
    microfono por defecto de pipewire (wpctl)

llamar prepararAudio() ANTES de importar sounddevice. en la laptop (o
con todo instalado por apt) no hace nada.
"""
import json
import os
import shutil
import subprocess

DIR_LIBS = os.path.expanduser("~/.local/lib/navia")
_listo = None


def _portaudioLocal():
    import ctypes.util
    try:
        if ctypes.util.find_library("portaudio"):
            return "sistema"
    except Exception:
        pass
    ruta = os.path.join(DIR_LIBS, "libportaudio.so.2")
    if not os.path.exists(ruta):
        return None
    original = ctypes.util.find_library
    ctypes.util.find_library = (
        lambda nombre: ruta if nombre == "portaudio" else original(nombre))
    return ruta


def nodosAudio():
    """[(id, clase, nombre, descripcion)] de pipewire; [] si no hay."""
    if not shutil.which("pw-dump"):
        return []
    try:
        crudo = subprocess.run(["pw-dump"], capture_output=True, timeout=5,
                               text=True).stdout
        objetos = json.loads(crudo)
    except Exception:
        return []
    out = []
    for o in objetos:
        props = (o.get("info") or {}).get("props") or {}
        clase = props.get("media.class", "")
        if clase in ("Audio/Sink", "Audio/Source"):
            out.append((o["id"], clase, props.get("node.name", ""),
                        props.get("node.description", "")))
    return out


def _esUsb(nombre, descripcion):
    t = (nombre + " " + descripcion).lower()
    return "usb" in t or "headset" in t or "headphone" in t


def elegirDiadema(nodos=None, aplicar=True):
    """pone la diadema usb como salida y entrada por defecto.
    -> {"salida": desc|None, "entrada": desc|None}"""
    nodos = nodosAudio() if nodos is None else nodos
    elegido = {"salida": None, "entrada": None}
    for clase, llave in (("Audio/Sink", "salida"), ("Audio/Source", "entrada")):
        usb = [n for n in nodos if n[1] == clase and _esUsb(n[2], n[3])]
        if not usb:
            continue
        nid, _c, _nombre, desc = usb[0]
        if aplicar and shutil.which("wpctl"):
            subprocess.run(["wpctl", "set-default", str(nid)],
                           capture_output=True, timeout=5)
        elegido[llave] = desc
    return elegido


def hayMicrofono(nodos=None):
    nodos = nodosAudio() if nodos is None else nodos
    return any(n[1] == "Audio/Source" for n in nodos)


def microfono(nodos=None):
    """nombre (node.name) del microfono a usar, la diadema usb primero;
    None si no hay ninguno. ojo: SIN microfono pipewire le conecta a
    pw-record el monitor de la bocina (medido 2026-11-10) y el sistema
    se oye a si mismo; por eso se graba pidiendo el microfono por
    nombre y, si no hay, no se graba."""
    nodos = nodosAudio() if nodos is None else nodos
    mics = [n for n in nodos if n[1] == "Audio/Source"]
    if not mics:
        return None
    usb = [n for n in mics if _esUsb(n[2], n[3])]
    return (usb or mics)[0][2]


def prepararAudio(verbose=True):
    global _listo
    if _listo is not None:
        return _listo
    pa = _portaudioLocal()
    nodos = nodosAudio()
    diadema = elegirDiadema(nodos)
    _listo = {"portaudio": pa, "pipewire": bool(nodos),
              "microfono": hayMicrofono(nodos), **diadema}
    if verbose and nodos:
        print(f"[AUDIO] salida: {diadema['salida'] or 'la del sistema'} | "
              f"micrófono: "
              f"{diadema['entrada'] or ('el del sistema' if _listo['microfono'] else 'NO HAY')}")
    return _listo


_tonos = {}


def tonito(nombre, volumen=0.35):
    """tonitos cortos de interfaz por pygame: "escucha_on" (sube),
    "escucha_off" (baja), "ok". no truena si no hay audio."""
    patrones = {"escucha_on": ((660, 70), (990, 90)),
                "escucha_off": ((990, 70), (660, 90)),
                "ok": ((880, 80),)}
    try:
        import numpy as np
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(44100, -16, 2, 512)
        fs, _fmt, canales = pygame.mixer.get_init()
        if nombre not in _tonos:
            partes = []
            for hz, ms in patrones[nombre]:
                t = np.arange(int(fs * ms / 1000.0)) / fs
                env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.008)
                partes.append(np.sin(2 * np.pi * hz * t) * env)
            onda = (np.concatenate(partes) * 32767 * 0.8).astype(np.int16)
            if canales == 2:
                onda = np.ascontiguousarray(np.stack([onda, onda], 1))
            _tonos[nombre] = pygame.mixer.Sound(buffer=onda.tobytes())
        _tonos[nombre].set_volume(volumen)
        _tonos[nombre].play()
    except Exception:
        pass


if __name__ == "__main__":
    print(prepararAudio())
    for n in nodosAudio():
        print(" ", n)
