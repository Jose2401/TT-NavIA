"""
evaluacion del pln y de la voz pa la doc del TT2.
  - pln: ~110 frases escritas a mano, NO sacadas de las semillas del pln
  - asr: voces en español de macOS (say) -> vosk -> pln, con ruido
correr (en mac):
    python -m tests.eval_interaccion --out docs/TT_2026_B045/Capturas_TT2/voz
ojo: corre el pln en una carpeta temporal pa no usar/ensuciar la
memoria aprendida (navigation_nlp_memory.json)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (frase, intencion esperada, etiqueta de destino esperada o None)
FRASES = [
    ("llévame a la salida", "DESTINO", "door"),
    ("quiero ir a la puerta", "DESTINO", "door"),
    ("me puedes llevar a la entrada por favor", "DESTINO", "door"),
    ("ayúdame a llegar a la salida", "DESTINO", "door"),
    ("guíame hacia la silla", "DESTINO", "chair"),
    ("vamos a la mesa", "DESTINO", "table"),
    ("quiero llegar al baño", "DESTINO", "toilet"),
    ("llévame al refri", "DESTINO", "fridge"),
    ("necesito ir a las escaleras", "DESTINO", "escalera"),
    ("dirígeme a la ventana", "DESTINO", "window"),
    ("ve a la planta", "DESTINO", "potted plant"),
    ("llévame al escritorio", "DESTINO", "desk"),
    ("quiero ir al bote de basura", "DESTINO", "trash can"),
    ("me llevas a la banca", "DESTINO", "bench"),
    ("navega hacia el elevador", "DESTINO", "elevador"),
    ("oye llévame a la salida de emergencia", "DESTINO", "door"),
    ("quiero que me lleves al sillón", "DESTINO", "sofa"),
    ("llevame a la puerta", "DESTINO", "door"),
    ("ir a la cafetería", "DESTINO", "cafeteria"),
    ("vamos para la biblioteca", "DESTINO", "biblioteca"),
    ("me puedes guiar a la mesa", "DESTINO", "table"),
    ("quiero ir a donde está la puerta", "DESTINO", "door"),
    ("llévame", "DESTINO", None),

    ("espérame tantito", "PAUSA", None),
    ("pausa un momento", "PAUSA", None),
    ("pon pausa", "PAUSA", None),
    ("detente un momento por favor", "PAUSA", None),
    ("espera voy a amarrarme las agujetas", "PAUSA", None),
    ("haz una pausa", "PAUSA", None),
    ("dame un momento", "PAUSA", None),
    ("espérate ahí", "PAUSA", None),

    ("ya podemos seguir", "REANUDAR", None),
    ("continúa por favor", "REANUDAR", None),
    ("sigue guiándome", "REANUDAR", None),
    ("reanuda la ruta", "REANUDAR", None),
    ("vamos sigue", "REANUDAR", None),
    ("ok continúa con la navegación", "REANUDAR", None),
    ("avanza", "REANUDAR", None),
    ("ya estoy listo sigamos", "REANUDAR", None),

    ("detente", "DETENER", None),
    ("alto ahí", "DETENER", None),
    ("para ya", "DETENER", None),
    ("frena", "DETENER", None),
    ("ya no quiero seguir", "DETENER", None),
    ("deja de guiarme", "DETENER", None),
    ("termina la navegación", "DETENER", None),
    ("stop", "DETENER", None),

    ("cancela la ruta", "CANCELAR", None),
    ("olvídalo", "CANCELAR", None),
    ("ya no quiero ir", "CANCELAR", None),
    ("cancela el destino", "CANCELAR", None),
    ("abandona la ruta", "CANCELAR", None),

    ("busca otro camino", "RUTA_ALTERNATIVA", None),
    ("hay otra ruta", "RUTA_ALTERNATIVA", None),
    ("por aquí no se puede cambia la ruta", "RUTA_ALTERNATIVA", None),
    ("recalcula", "RUTA_ALTERNATIVA", None),
    ("dame una ruta diferente", "RUTA_ALTERNATIVA", None),
    ("llévame por otro lado", "RUTA_ALTERNATIVA", None),

    ("auxilio", "EMERGENCIA", None),
    ("ayuda me caí", "EMERGENCIA", None),
    ("es una emergencia", "EMERGENCIA", None),
    ("me siento mal necesito ayuda", "EMERGENCIA", None),
    ("socorro", "EMERGENCIA", None),
    ("estoy en peligro", "EMERGENCIA", None),
    ("llama a alguien rápido", "EMERGENCIA", None),

    ("dónde estoy", "UBICACION", None),
    ("en qué parte estoy", "UBICACION", None),
    ("dime mi ubicación", "UBICACION", None),
    ("en qué cuarto estoy", "UBICACION", None),
    ("dónde me encuentro", "UBICACION", None),

    ("qué hay enfrente", "ENTORNO", None),
    ("qué tengo adelante", "ENTORNO", None),
    ("describe lo que hay alrededor", "ENTORNO", None),
    ("hay obstáculos", "ENTORNO", None),
    ("qué ves", "ENTORNO", None),
    ("qué hay a mi alrededor", "ENTORNO", None),

    ("repite", "REPETIR", None),
    ("no te entendí repítelo", "REPETIR", None),
    ("otra vez por favor", "REPETIR", None),
    ("qué dijiste", "REPETIR", None),
    ("vuelve a decirlo", "REPETIR", None),

    ("ayuda", "AYUDA", None),
    ("qué puedo decirte", "AYUDA", None),
    ("cuáles son los comandos", "AYUDA", None),
    ("cómo te uso", "AYUDA", None),
    ("qué opciones tengo", "AYUDA", None),

    ("sí", "CONFIRMAR", None),
    ("sí correcto", "CONFIRMAR", None),
    ("está bien", "CONFIRMAR", None),
    ("claro que sí", "CONFIRMAR", None),
    ("afirmativo", "CONFIRMAR", None),
    ("ok", "CONFIRMAR", None),

    ("no", "NEGAR", None),
    ("no gracias", "NEGAR", None),
    ("negativo", "NEGAR", None),
    ("eso no", "NEGAR", None),
    ("no es correcto", "NEGAR", None),

    ("cuánto falta", "ESTADO", None),
    ("ya casi llegamos", "ESTADO", None),
    ("cómo voy", "ESTADO", None),
    ("falta mucho para llegar", "ESTADO", None),
    ("cuánto me falta para la salida", "ESTADO", None),
]

# lo que de verdad hace el sistema: detener y cancelar acaban igual
ACCION_EFECTIVA = {"DETENER": "detener", "CANCELAR": "detener"}

# subconjunto pa la prueba de voz (comandos de uso diario)
FRASES_ASR = [
    "llévame a la salida", "quiero ir a la puerta", "guíame hacia la silla",
    "vamos a la mesa", "quiero llegar al baño", "necesito ir a las escaleras",
    "llévame al escritorio", "pausa un momento", "haz una pausa",
    "continúa por favor", "reanuda la ruta", "detente", "alto ahí",
    "frena", "cancela la ruta", "olvídalo", "busca otro camino",
    "recalcula", "auxilio", "es una emergencia", "socorro", "dónde estoy",
    "dime mi ubicación", "qué hay enfrente", "qué tengo adelante",
    "repite", "ayuda", "cuánto falta", "sí correcto", "no gracias",
]

VOCES = ["Paulina", "Mónica", "Eddy (Spanish (Mexico))",
         "Eddy (Spanish (Spain))", "Flo (Spanish (Mexico))",
         "Flo (Spanish (Spain))", "Grandma (Spanish (Mexico))",
         "Grandpa (Spanish (Mexico))", "Grandma (Spanish (Spain))",
         "Grandpa (Spanish (Spain))"]


def _armarPln():
    sys.path.insert(0, os.path.join(RAIZ, "VozYPLN"))
    sys.path.insert(0, RAIZ)
    viejo = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    from PLN_oneimport import NavigationNLP
    nlp = NavigationNLP()
    os.chdir(viejo)
    return nlp


def _etiquetaDestino(destino):
    if destino is None:
        return None
    from interaccion import _limpiarDestino
    from navigation.navigator import DEST_ALIASES
    d = _limpiarDestino(destino)
    return DEST_ALIASES.get(d, d)


def evaluarPln(nlp, frases=FRASES):
    filas = []
    for texto, esperada, dest in frases:
        t0 = time.perf_counter()
        r = nlp.process(texto)
        dt = time.perf_counter() - t0
        pred = r["intencion"]
        okDest = None
        if esperada == "DESTINO" and pred == "DESTINO":
            if dest is None:
                okDest = bool(r["requiere_aclaracion"])
            else:
                okDest = _etiquetaDestino(r["destino"]) == dest
        filas.append(dict(texto=texto, esperada=esperada, pred=pred,
                          conf=r["confianza"], destino=r["destino"],
                          okDest=okDest, ms=1000 * dt))
    return filas


def resumenPln(filas):
    n = len(filas)
    ok = sum(f["pred"] == f["esperada"] for f in filas)
    efe = sum(ACCION_EFECTIVA.get(f["pred"], f["pred"])
              == ACCION_EFECTIVA.get(f["esperada"], f["esperada"])
              for f in filas)
    dests = [f["okDest"] for f in filas if f["okDest"] is not None]
    return dict(n=n, exactitud=ok / n, exactitud_accion=efe / n,
                destino_ok=sum(dests), destino_n=len(dests),
                ms_prom=float(np.mean([f["ms"] for f in filas])),
                ms_max=float(np.max([f["ms"] for f in filas])))


def figConfusion(filas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted({f["esperada"] for f in filas}
                    | {f["pred"] for f in filas})
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for f in filas:
        m[idx[f["esperada"]], idx[f["pred"]]] += 1
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(m, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.lower() for l in labels], rotation=60,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.lower() for l in labels], fontsize=8)
    ax.set_xlabel("intención detectada")
    ax.set_ylabel("intención esperada")
    ax.set_title("Matriz de confusión del PLN (frases de prueba)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------- voz ----------------

def _sintetizar(texto, voz, wavPath):
    aiff = wavPath[:-4] + ".aiff"
    subprocess.run(["say", "-v", voz, "-o", aiff, texto], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c",
                    "1", aiff, wavPath], check=True,
                   stdout=subprocess.DEVNULL)
    os.remove(aiff)


def _leerWav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def _ruidoRosa(n, rng):
    # rosa por filtrado 1/f en frecuencia, se parece mas a ruido de calle
    blanco = rng.standard_normal(n)
    f = np.fft.rfft(blanco)
    esc = 1.0 / np.sqrt(np.maximum(np.arange(len(f)), 1))
    return np.fft.irfft(f * esc, n)


def _mezclar(voz, snrDb, rng):
    x = voz.astype(np.float64)
    # medio segundo de silencio antes y despues en TODAS las condiciones:
    # sin eso vosk se come la primera palabra y el "limpio" sale peor
    pad = np.zeros(8000)
    x = np.concatenate([pad, x, pad])
    if snrDb is None:
        return x.astype(np.int16)
    ruido = _ruidoRosa(len(x), rng)
    pVoz = np.mean(voz.astype(np.float64) ** 2)
    pRuido = np.mean(ruido ** 2)
    ruido *= np.sqrt(pVoz / (pRuido * 10 ** (snrDb / 10.0)))
    y = x + ruido
    y = np.clip(y, -32768, 32767)
    return y.astype(np.int16)


def _wer(ref, hip):
    from PLN_oneimport import normalize_text
    r, h = normalize_text(ref).split(), normalize_text(hip).split()
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)], len(r)


def evaluarAsr(nlp, voskDir, snrs=(None, 20, 10, 5), trabajo=None):
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    modelo = Model(voskDir)
    esperado = {t: (e, d) for t, e, d in FRASES}
    trabajo = trabajo or tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    audios = {}
    for voz in VOCES:
        for i, texto in enumerate(FRASES_ASR):
            p = os.path.join(trabajo, f"{abs(hash(voz)) % 9999}_{i}.wav")
            if not os.path.exists(p):
                _sintetizar(texto, voz, p)
            audios[(voz, texto)] = _leerWav(p)

    filas = []
    for snr in snrs:
        for (voz, texto), audio in audios.items():
            señal = _mezclar(audio, snr, rng)
            rec = KaldiRecognizer(modelo, 16000)
            t0 = time.perf_counter()
            for k in range(0, len(señal), 4000):
                rec.AcceptWaveform(señal[k:k + 4000].tobytes())
            tFin = time.perf_counter()
            hip = json.loads(rec.FinalResult()).get("text", "")
            tFinal = time.perf_counter() - tFin
            dur = len(señal) / 16000.0
            t1 = time.perf_counter()
            r = nlp.process(hip) if hip else {"intencion": "DESCONOCIDO",
                                              "destino": None,
                                              "requiere_aclaracion": False}
            tPln = time.perf_counter() - t1
            errores, palabras = _wer(texto, hip)
            esp, dest = esperado[texto]
            ok = (ACCION_EFECTIVA.get(r["intencion"], r["intencion"])
                  == ACCION_EFECTIVA.get(esp, esp))
            if ok and esp == "DESTINO" and dest is not None:
                ok = _etiquetaDestino(r["destino"]) == dest
            filas.append(dict(voz=voz, snr=snr, texto=texto, hip=hip,
                              errores=int(errores), palabras=int(palabras),
                              ok=bool(ok), dur=dur,
                              rtf=(tFin - t0 + tFinal) / dur,
                              ms_final=1000 * tFinal, ms_pln=1000 * tPln))
    return filas


def resumenAsr(filas):
    out = []
    for snr in sorted({f["snr"] for f in filas},
                      key=lambda s: -999 if s is None else -s):
        fs = [f for f in filas if f["snr"] == snr]
        out.append(dict(
            snr="limpio" if snr is None else snr, n=len(fs),
            wer=sum(f["errores"] for f in fs) / sum(f["palabras"] for f in fs),
            exactitud=sum(f["ok"] for f in fs) / len(fs),
            ms_final=float(np.mean([f["ms_final"] for f in fs])),
            ms_final_p95=float(np.percentile([f["ms_final"] for f in fs], 95)),
            ms_pln=float(np.mean([f["ms_pln"] for f in fs])),
            rtf=float(np.mean([f["rtf"] for f in fs]))))
    return out


def figAsr(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [str(r["snr"]) + ("" if r["snr"] == "limpio" else " dB")
          for r in res]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.38
    i = np.arange(len(res))
    b1 = ax.bar(i - w / 2, [100 * (1 - r["wer"]) for r in res], w,
                label="palabras correctas (1 - WER)", color="#4C78A8")
    b2 = ax.bar(i + w / 2, [100 * r["exactitud"] for r in res], w,
                label="comando interpretado bien", color="#F58518")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=8)
    ax.axhline(92, ls="--", color="tab:red", lw=1)
    ax.text(len(res) - 0.5, 93.5, "RNF-07 (92 %)", color="tab:red",
            fontsize=8, ha="right")
    ax.set_xticks(i)
    ax.set_xticklabels(xs)
    ax.set_xlabel("relación señal a ruido")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Voz sintetizada → Vosk → PLN (10 voces, 30 comandos)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/TT_2026_B045/Capturas_TT2/voz")
    p.add_argument("--vosk", default=os.path.join(
        RAIZ, "models", "vosk-model-small-es-0.42"))
    p.add_argument("--sin-asr", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nlp = _armarPln()
    filas = evaluarPln(nlp)
    res = resumenPln(filas)
    print("PLN:", res)
    for f in filas:
        if f["pred"] != f["esperada"] or f["okDest"] is False:
            print("  X", f["texto"], "->", f["pred"], f["destino"])
    figConfusion(filas, os.path.join(args.out, "pln_confusion.png"))
    json.dump({"resumen": res, "filas": filas},
              open(os.path.join(args.out, "pln_eval.json"), "w"),
              indent=1, ensure_ascii=False)

    if args.sin_asr:
        return
    trabajo = os.path.join(tempfile.gettempdir(), "tt_asr_wavs")
    os.makedirs(trabajo, exist_ok=True)
    fasr = evaluarAsr(nlp, args.vosk, trabajo=trabajo)
    rasr = resumenAsr(fasr)
    for r in rasr:
        print("ASR:", r)
    figAsr(rasr, os.path.join(args.out, "asr_ruido.png"))
    json.dump({"resumen": rasr, "filas": fasr},
              open(os.path.join(args.out, "asr_eval.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
