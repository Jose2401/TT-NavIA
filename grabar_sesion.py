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


def carpetaNueva(prefijo=""):
    """sesiones/[prefijo]AAAAMMDD_HHMMSS, sin reusar nunca una carpeta.
    ojo: la pi no tiene reloj con pila; sin internet arranca con la hora
    del ultimo apagado y dos grabaciones pueden caer en el mismo segundo."""
    base = os.path.join("sesiones", prefijo + time.strftime("%Y%m%d_%H%M%S"))
    carpeta, k = base, 1
    while os.path.exists(carpeta):
        k += 1
        carpeta = f"{base}_{k}"
    return carpeta


class GrabadorCorrida:
    """graba una corrida del programa principal (main_nav_pi --grabar)
    con el mismo formato que una sesion: cuadros + lo del rp2040, y ademas
    eventos.txt con lo que se oyo, lo que el sistema dijo, las alertas y
    la pose. asi una prueba de NAVEGACION se puede repasar despues
    (sesion_a_mapa.py / sesion_a_video.py) en vez de quedar en "guio raro".
    los cuadros se escriben en otro hilo pa no frenar el ciclo y todo se
    baja a la tarjeta cada 2 s (la pi se apaga desconectandola)."""

    def __init__(self, sensores=None, carpeta=None, calidad=80, meta=None,
                 ancho=640):
        """ancho: los cuadros se guardan a ese ancho (640 = lo mismo que
        ve la vision; a 1280 grabar le quitaba ~15 % de velocidad al
        programa)."""
        import queue
        self.ancho = ancho
        self.carpeta = carpeta or carpetaNueva("nav_")
        os.makedirs(os.path.join(self.carpeta, "cuadros"), exist_ok=True)
        self.t0 = time.time()
        self.calidad = calidad
        self.n = 0
        self._lock = threading.Lock()
        self._fSerie = open(os.path.join(self.carpeta, "serie.txt"), "w",
                            encoding="utf-8")
        self._fCuadros = open(os.path.join(self.carpeta, "cuadros.csv"), "w",
                              encoding="utf-8")
        self._fCuadros.write("indice,t\n")
        self._fEventos = open(os.path.join(self.carpeta, "eventos.txt"), "w",
                              encoding="utf-8")
        self.meta = {"tipo": "navegacion",
                     "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "perfil": "picam3wide"}
        self.meta.update(meta or {})
        self._guardarMeta()
        self._cola = queue.Queue(maxsize=4)
        self._fin = threading.Event()
        self._sens = sensores
        if sensores is not None:
            sensores.alLinea = self._linea
        threading.Thread(target=self._escritor, daemon=True).start()
        threading.Thread(target=self._guardian, daemon=True).start()

    def _guardarMeta(self):
        with open(os.path.join(self.carpeta, "meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=1, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())

    def _linea(self, linea, t):
        linea = linea.strip()
        if linea[:2] in ("I,", "U,", "E,", "H,"):
            with self._lock:
                if not self._fSerie.closed:
                    self._fSerie.write(f"{t - self.t0:.4f} {linea}\n")

    def cuadro(self, frame, tFrame=None):
        """encola el cuadro; si el disco no alcanza, se salta (nunca
        frena al programa)."""
        try:
            self._cola.put_nowait((frame, (tFrame or time.time()) - self.t0))
        except Exception:
            pass

    def evento(self, tipo, texto):
        with self._lock:
            if not self._fEventos.closed:
                self._fEventos.write(
                    f"{time.time() - self.t0:.2f} {tipo} {texto}\n")

    def _escritor(self):
        import cv2
        while not self._fin.is_set() or not self._cola.empty():
            try:
                frame, t = self._cola.get(timeout=0.5)
            except Exception:
                continue
            try:
                if self.ancho and frame.shape[1] > self.ancho:
                    alto = int(frame.shape[0] * self.ancho / frame.shape[1])
                    frame = cv2.resize(frame, (self.ancho, alto),
                                       interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(self.carpeta, "cuadros",
                                         f"{self.n:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.calidad])
                with self._lock:
                    if not self._fCuadros.closed:
                        self._fCuadros.write(f"{self.n},{t:.4f}\n")
                self.n += 1
            except Exception:
                pass

    def _guardian(self):
        while not self._fin.wait(2.0):
            try:
                with self._lock:
                    for fh in (self._fSerie, self._fCuadros, self._fEventos):
                        if not fh.closed:
                            fh.flush()
                os.sync()
            except (OSError, ValueError):
                pass

    def cerrar(self, **extra):
        if self._sens is not None:
            self._sens.alLinea = None
        self._fin.set()
        time.sleep(0.6)                    # que el escritor vacie la cola
        with self._lock:
            for fh in (self._fSerie, self._fCuadros, self._fEventos):
                fh.close()
        self.meta.update(duracionS=round(time.time() - self.t0, 1),
                         cuadros=self.n, **extra)
        self._guardarMeta()
        os.sync()


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
        args.carpeta = carpetaNueva()
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
    rutaMeta = os.path.join(args.carpeta, "meta.json")

    def guardarMeta():
        with open(rutaMeta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
    guardarMeta()                      # desde ya, por si la apagan a medias

    # ojo: sin monitor la pi se apaga DESCONECTANDOLA, y lo escrito en
    # los ultimos ~30 s todavia no estaba en la tarjeta: en los recorridos
    # del 2026-11-12 se perdieron los ultimos 15-20 s de cuadros, los
    # tiempos y el meta.json. cada 2 s se baja todo al disco
    finGuardian = threading.Event()

    def guardian():
        while not finGuardian.wait(2.0):
            try:
                with lock:
                    if not fSerie.closed:
                        fSerie.flush()
                if not fCuadros.closed:
                    fCuadros.flush()
                os.sync()
            except (OSError, ValueError):
                pass
    threading.Thread(target=guardian, daemon=True).start()
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
        finGuardian.set()
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
        guardarMeta()
        os.sync()          # ya todo en la tarjeta ANTES de decir "guardado"
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
