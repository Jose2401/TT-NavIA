#!/bin/bash
# instalar_pi.sh — deja lista la Raspberry Pi 5 de las gafas SIN sudo.
# correr desde la raiz del repo:  bash instalar_pi.sh
# (se puede correr varias veces; lo que ya esta no se vuelve a bajar)
set -e
cd "$(dirname "$0")"

echo "== 1/6 entorno de python (.venv con los paquetes del sistema) =="
# --system-site-packages: picamera2, gpiozero/lgpio, pyserial y pygame
# vienen con raspberry pi os y no se instalan bien por pip
[ -d .venv ] || python3 -m venv --system-site-packages .venv
NUMPY=$(python3 -c "import numpy; print(numpy.__version__)")
echo "numpy==$NUMPY" > /tmp/navia_constraints.txt   # el mismo de picamera2
.venv/bin/python -m pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    -c /tmp/navia_constraints.txt \
    torch torchvision ultralytics transformers gymnasium stable-baselines3 \
    matplotlib vosk sounddevice pyttsx3 edge-tts onnx onnxruntime ncnn piper-tts

echo "== 2/6 portaudio sin sudo (lo ocupa sounddevice) =="
LIBS="$HOME/.local/lib/navia"
mkdir -p "$LIBS"
if ! ldconfig -p | grep -q libportaudio && [ ! -f "$LIBS/libportaudio.so.2" ]; then
    TMP=$(mktemp -d)
    (cd "$TMP" && apt-get download libportaudio2 >/dev/null && dpkg -x libportaudio2_*.deb x \
        && cp x/usr/lib/*/libportaudio.so.2.0.0 "$LIBS/libportaudio.so.2")
    rm -rf "$TMP"
fi

echo "== 3/6 modelo de voz a texto (vosk, español) =="
if [ ! -d models/vosk-model-small-es-0.42 ]; then
    mkdir -p models
    (cd models && curl -sL -o vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip \
        && unzip -q -o vosk.zip && rm vosk.zip)
fi

echo "== 4/6 voz sin internet (piper, español de México) =="
mkdir -p models/piper
[ -f models/piper/es_MX-claude-high.onnx.json ] || \
    .venv/bin/python -m piper.download_voices --download-dir models/piper es_MX-claude-high

echo "== 5/6 yolo en formato ncnn (4x mas rapido en la pi) =="
.venv/bin/python exportar_modelos.py

echo "== 6/6 firmware del rp2040 de las gafas =="
if [ -d /media/$USER/CIRCUITPY ]; then
    if ! cmp -s firmware/code.py /media/$USER/CIRCUITPY/code.py; then
        cp firmware/code.py /media/$USER/CIRCUITPY/code.py && sync
        echo "firmware copiado a CIRCUITPY (se reinicia solo)"
    else
        echo "firmware ya esta al dia"
    fi
else
    echo "no veo la unidad CIRCUITPY: conecta el rp2040 y copia firmware/code.py a mano"
fi

echo
echo "listo. siguiente:"
echo "  .venv/bin/python sensores.py            # ver imu y ultrasonicos en vivo"
echo "  .venv/bin/python calibrar_gafas.py      # una vez, con los lentes puestos"
echo "  .venv/bin/python main_nav_pi.py         # el sistema completo"
