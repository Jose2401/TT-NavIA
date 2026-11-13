"""
grabar_sesion.py — graba un recorrido con las gafas puestas: los cuadros
de la camara + TODO lo que manda el rp2040 (imu a 50 Hz y ultrasonicos),
con tiempos. no corre vision ni nada pesado, solo guarda.

pa que sirve la grabacion:
  - sesion_a_mapa.py la pasa por el pipeline real (vision + imu +
    ultrasonido) y saca un mapa: con esos mapas se evalua/reentrena la
    navegacion (python -m navigation.train --real-maps ...)
  - ajustar el contador de pasos y revisar el ruido real de los sensores
  - sacar imagenes pa etiquetar y afinar el yolo (vision/finetune.py)

uso:
    python grabar_sesion.py                      (sesiones/AAAAMMDD_HHMMSS)
    python grabar_sesion.py sesiones/pasillo1    (ctrl-c pa parar)
    python grabar_sesion.py sesiones/pasillo1 --segundos 120 --fps 5
    python grabar_sesion.py --boton parar        (un click la termina)

sin monitor: al prender la pi, DOS clicks del boton arrancan esto solo
(~/gafas/arranque/arranque_boton.py, aparte del proyecto) y un click lo
para; al acabar dice cuanto se camino.

la distancia sale sola: pasos x largo segun la cadencia, y donde hubo
una pared enfrente tambien por ultrasonido, de comprobacion (queda en
meta.json -> "caminata"). ya no hace falta medirla con cinta.

el boton de la pi deja una MARCA en la grabacion (p.ej. al pasar por una
puerta); con --boton parar, la termina.
"""
import argparse
import json
import os
import signal
import threading
import time


def _decirResumen(texto):
    """lo dice por la voz del sistema (sin monitor es la unica forma de
    saber que si se guardo). si no hay tts no pasa nada."""
    try:
        from interaccion import ColaVoz
        voz = ColaVoz(motor="auto", echo=False)
        voz.decir(texto)
        voz.cerrar()
    except Exception as e:
        print(f"(sin voz: {e})")


def main():
    p = argparse.ArgumentParser(description="grabar un recorrido con las gafas")
    p.add_argument("carpeta", nargs="?", default=None,
                   help="donde guardar (por defecto sesiones/fecha_hora)")
    p.add_argument("--segundos", type=float, default=None)
    p.add_argument("--max-min", type=float, default=20.0,
                   help="tope de duración, por si se olvida pararla")
    p.add_argument("--fps", type=float, default=5.0,
                   help="cuadros por segundo que se guardan")
    p.add_argument("--calidad", type=int, default=85, help="jpeg 1-100")
    p.add_argument("--boton", choices=("marca", "parar"), default="marca",
                   help="qué hace el botón: dejar una marca o terminar")
    p.add_argument("--voz", action="store_true",
                   help="decir el resumen al terminar (pa usarla sin monitor)")
    p.add_argument("--nota", default="", help="texto libre pal meta.json")
    args = p.parse_args()
    if args.carpeta is None:
        args.carpeta = os.path.join("sesiones",
                                    time.strftime("%Y%m%d_%H%M%S"))
    tope = args.max_min * 60.0
    if args.segundos is not None:
        tope = min(tope, args.segundos)

    import cv2
    from camara import abrirCamara
    from sensores import SensoresGafas, resumenCaminata, textoResumen
    from alertas import BotonPullUp
    import audio_pi

    # systemctl stop (SIGTERM) tambien cierra bien los archivos
    def _sigterm(_s, _f):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except ValueError:
        pass

    dirCuadros = os.path.join(args.carpeta, "cuadros")
    os.makedirs(dirCuadros, exist_ok=True)

    cap, perfil = abrirCamara(0)
    if not cap.isOpened():
        raise SystemExit(f"sin cámara ({getattr(cap, 'razon', '')})")
    sens = SensoresGafas()
    if not sens.available:
        raise SystemExit(f"sin sensores de las gafas ({sens.razon})")

    t0 = time.time()
    lock = threading.Lock()
    rutaSerie = os.path.join(args.carpeta, "serie.txt")
    fSerie = open(rutaSerie, "w", encoding="utf-8")
    fCuadros = open(os.path.join(args.carpeta, "cuadros.csv"), "w",
                    encoding="utf-8")
    fCuadros.write("indice,t\n")
    marcas = []
    parar = threading.Event()

    def alLinea(linea, t):
        linea = linea.strip()
        if linea[:2] in ("I,", "U,", "E,", "H,"):
            with lock:
                if not fSerie.closed:
                    fSerie.write(f"{t - t0:.4f} {linea}\n")
    sens.alLinea = alLinea

    def marca():
        t = time.time() - t0
        marcas.append(round(t, 3))
        with lock:
            if not fSerie.closed:
                fSerie.write(f"{t:.4f} M,{len(marcas)}\n")
        audio_pi.tonito("ok")
        print(f"\n[marca {len(marcas)}] t={t:.1f} s")

    alBoton = parar.set if args.boton == "parar" else marca
    boton = BotonPullUp(alClick=alBoton, alCambiar=alBoton, ventanaS=0.05)
    if args.boton == "parar" and not boton.disponible:
        # sin boton no habria como terminarla (sin monitor ni teclado)
        fSerie.close()
        fCuadros.close()
        cap.release()
        sens.cerrar()
        raise SystemExit(f"el botón no está disponible ({boton.razon})")
    boton.led(True)                    # prendido = grabando

    meta = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "perfil": perfil,
            "fps": args.fps, "firmware": sens.version, "nota": args.nota}
    queHace = {"marca": "botón = marca", "parar": "botón = terminar"}
    print(f"grabando en {args.carpeta} (ctrl-c pa parar"
          + (f", {queHace[args.boton]}" if boton.disponible else "") + ")")
    audio_pi.tonito("escucha_on")

    n = 0
    periodo = 1.0 / args.fps
    siguiente = time.time()
    try:
        while time.time() - t0 < tope and not parar.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            ahora = time.time()
            if ahora < siguiente:
                continue
            siguiente = max(siguiente + periodo, ahora - periodo)
            cv2.imwrite(os.path.join(dirCuadros, f"{n:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.calidad])
            fCuadros.write(f"{n},{cap.tFrame - t0:.4f}\n")
            n += 1
            if n % 5 == 0:
                ecos = {u.lado: (f"{u.range_m:.2f}" if u.valid else "---")
                        for u in sens.lecturasUS()}
                e = sens.euler or (0, 0, 0)
                print(f"t={ahora - t0:5.1f}s cuadros={n} "
                      f"pasos={sens.pasos.stepsTotal} "
                      f"metros={sens.metros:5.1f} "
                      f"rumbo={e[0]:5.1f} cabeceo={e[2]:+5.1f} "
                      f"izq={ecos.get('izq', '?')} der={ecos.get('der', '?')}   ",
                      end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dur = time.time() - t0
        sens.alLinea = None
        with lock:
            fSerie.close()
        fCuadros.close()
        cap.release()
        try:
            # una grabacion es una caminata: sirve pa afinar el largo de
            # paso de quien trae los lentes (queda pa la siguiente corrida)
            aprendido = sens.aprenderPaso()
        except Exception:
            aprendido = None
        sens.cerrar()
        boton.led(False)
        boton.cerrar()
        audio_pi.tonito("escucha_off")
        # cuanto se camino, con el mismo contador que usa el sistema
        try:
            caminata = resumenCaminata(rutaSerie)
        except Exception as e:
            caminata = {"pasos": sens.pasos.stepsTotal,
                        "metros": round(sens.metros, 2), "cadenciaPpm": None,
                        "eco": None, "error": str(e)}
        if aprendido is not None:
            meta["largoPasoAprendidoM"] = aprendido[0]
        meta.update(firmware=sens.version,
                    duracionS=round(dur, 1), cuadros=n, marcas=marcas,
                    pasosDetectados=caminata["pasos"],
                    metrosEstimados=caminata["metros"], caminata=caminata,
                    muestrasImu=sens.muestrasImu, brincosRumbo=sens.brincos)
        with open(os.path.join(args.carpeta, "meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        time.sleep(0.3)
        print(f"\nlisto: {n} cuadros, {dur:.0f} s, {len(marcas)} marcas "
              f"-> {args.carpeta}")
        print("caminata: " + textoResumen(caminata))
        if args.voz:
            minutos = int(dur // 60)
            tiempo = (f"{minutos} minutos" if minutos >= 2 else
                      f"{int(round(dur))} segundos")
            _decirResumen(f"Recorrido guardado. {tiempo}, "
                          f"{caminata['pasos']} pasos, unos "
                          f"{int(round(caminata['metros']))} metros.")


if __name__ == "__main__":
    main()
    # igual que main_nav_pi: salida directa pa que ningun hilo nativo
    # (libcamera, onnxruntime de la voz) aborte al cerrar
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
