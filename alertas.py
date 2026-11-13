"""
alertas.py — el guardian del sistema (alg. 7 del TT), SIEMPRE corriendo
pida rutas o no. decide QUE avisar leyendo el NavigationFrame (umbrales
del doc, personas en movimiento, hoyos/escaleras) y lo entrega con el
modulo de pitidos del equipo (VozYPLN/alertas.py): precaucion = puro
pip estereo (izq/der/frente se oyen distinto), critica = pip + frase
corta hablada que interrumpe al tts. asi no se confunde con las
indicaciones de navegacion. la pausa/replanificacion sigue igual via
el (alertas, recalcular) que consume el navegador.
tambien recibe señales directas de sensores (reportarSensor): ahi se
conectan los ULTRASONICOS de las gafas, desde su propio hilo, pa que lo
muy cercano pite sin esperar a la camara.
"""
import math
import threading
import time
from dataclasses import replace

from navigation_feed import (safetyAlerts, alertasDeObstaculo,
                             _sectorOf as _sectorDe, CRITICAL_DISTANCE_M)
from nombres import nombreEs

# a que distancia SUENA un obstaculo fijo (pared, mueble, lo que vea el
# ultrasonido). el documento del TT pone 0.80 / 1.50 m, pero en
# interiores eso sonaba todo el rato (un pasillo mide 1 m de ancho) y
# la voz no paraba. pedido del equipo (2026-11-10): la alerta hablada
# solo ya encima, a 25 cm; antes de eso va nomas el pip de precaucion.
# ojo: a 0.6 m/s (lo que se midio caminando en casa) 25 cm son 0.4 s:
# la voz ya no alcanza a frenar a nadie, pa eso esta el pip.
# se cambian con --umbral-alerta / --umbral-precaucion en los mains.
# lo que se MUEVE (personas) y los peligros del piso (escaleras, hoyos)
# conservan sus distancias, y el navegador sigue recalculando a 0.80 m
CRITICA_M = 0.25
PRECAUCION_M = 0.60

_COOLDOWN = {0: 2.5, 1: 8.0}
# si la cosa sigue A LA MISMA distancia (parado frente a una pared, o
# sentado en un escritorio) la espera se va duplicando hasta este tope:
# avisar cada 2.5 s de algo que no se mueve nomas cansa
COOLDOWN_MAX_S = 30.0
QUIETO_M = 0.15
# una vez en critica, sigue siendolo hasta alejarse esto mas: el eco de
# algo quieto brinca 2-5 cm y, justo en el umbral, la misma cosa sonaba
# como "critica" y como "precaucion" por turnos
HISTERESIS_M = 0.08
VOL_PIP = 0.7
CONF_MIN_MAPA = 0.35    # debajo de esto el mapa no dispara alertas
_NOMBRE_SECTOR = {"front": "al frente", "left": "a la izquierda",
                  "right": "a la derecha"}
_SECTOR_A_DIR = {"front": "frente", "left": "izquierda",
                 "right": "derecha"}


def _vozypln():
    import sys, os
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "VozYPLN")
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
    try:
        import audio_pi
        audio_pi.prepararAudio()       # antes de importar sounddevice
    except Exception:
        pass


def _armarSalidaPip(hablarFn=None, interrumpirFn=None):
    """envuelve el AlertManager del equipo pa tocar pips con el nivel
    que decide ESTE modulo (sin recalcular niveles ni cooldowns: eso ya
    lo hicimos aca). devuelve (salida, razon_si_no)."""
    try:
        _vozypln()
        import importlib
        pipmod = importlib.import_module("VozYPLN.alertas")
        import dialogo
    except Exception as e:
        return None, str(e)

    class SalidaPip(pipmod.AlertManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # en la pi los pips salen por pygame (el mismo mezclador de
            # la voz): por sounddevice/alsa tardaban segundos en sonar
            self._sonidos = None
            self._nivelDe = {datos: nivel for (nivel, _d), datos
                             in self._buffers.items()}
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(pipmod.SAMPLE_RATE, -16, 2, 512)
                if pygame.mixer.get_init() == (pipmod.SAMPLE_RATE, -16, 2):
                    self._sonidos = {}
                    for datos in self._buffers.values():
                        snd = pygame.mixer.Sound(buffer=datos)
                        snd.set_volume(VOL_PIP)
                        self._sonidos[datos] = snd
            except Exception:
                self._sonidos = None

        def _play(self, data):
            # solo el pip CRITICO calla a la voz (y la voz decide: una
            # alerta que ya va sonando no se corta). el de precaucion
            # suena encima: cortar una respuesta por un pip suave era
            # perder la frase completa
            if interrumpirFn and self._nivelDe.get(data, 0) >= pipmod.CRITICAL:
                try:
                    interrumpirFn()
                except Exception:
                    pass
            snd = self._sonidos.get(data) if self._sonidos else None
            if snd is None:
                return super()._play(data)
            canal = snd.play()
            while canal is not None and canal.get_busy():
                time.sleep(0.01)

        def pip(self, nivel, direccion="frente", tipo="obstaculo",
                distanciaM=None):
            ev = pipmod.ObstacleEvent(tipo, distanciaM, direccion,
                                      "sistema", False, nivel,
                                      time.monotonic())
            self._queue.put(ev)

    def _hablar(ev):
        # solo la critica se habla, cortito (dialogo del equipo)
        if ev.nivel >= pipmod.CRITICAL and hablarFn:
            hablarFn(dialogo.obstacle_warning(ev.tipo, ev.direccion,
                                              ev.distancia_m, ev.nivel))

    # el dialogo del equipo solo conoce carros/personas: le enseñamos el
    # nombre en español de todas nuestras clases (con articulo indef)
    from nombres import _ES
    for label in _ES:
        dialogo.OBSTACLE_NAMES.setdefault(label, nombreEs(label, "indef"))
    salida = SalidaPip(on_interrupt=None, on_speak=_hablar)
    salida.start()
    salida.CRITICAL = pipmod.CRITICAL
    salida.WARNING = pipmod.WARNING
    return salida, None


class ModuloAlertas:
    def __init__(self, activo=True, hablarFn=None, interrumpirFn=None,
                 conPip=True, criticaM=None, precaucionM=None,
                 delMapa=False):
        """hablarFn(texto): voz pa la alerta critica (ColaVoz.alerta).
        conPip=False: sin audio (tests / --sin-pip), las criticas se
        hablan con hablarFn como antes.
        criticaM / precaucionM: a que distancia suena un obstaculo fijo
        (None = CRITICA_M / PRECAUCION_M de arriba).
        delMapa: que tambien SUENE lo que el mapa guardado dice que hay
        cerca. apagado: en los recorridos reales del 2026-11-12 las 4
        alertas habladas de interiores ("obstaculo a 13 cm, detengase")
        salieron del mapa mientras el ultrasonido media de 1.3 a 3.9 m o
        nada. la pose y las marcas del mapa traen error de medio metro
        o mas: pa avisar a 25-60 cm solo sirve lo que los sensores ven
        en ese momento (ultrasonido y la deteccion del cuadro actual).
        el navegador si sigue usando el mapa pa decidir."""
        self.activo = activo
        self.hablarFn = hablarFn
        self.criticaM = CRITICA_M if criticaM is None else criticaM
        self.precaucionM = max(self.criticaM, PRECAUCION_M
                               if precaucionM is None else precaucionM)
        # callable() -> True si el usuario va caminando (lo pone el main
        # con los pasos de la imu). parado junto a algo que no se mueve,
        # la alerta se dice UNA vez; se repite cuando vuelve a caminar
        self.caminandoFn = None
        self.delMapa = delMapa
        self._enCritica = False
        # el reloj de las esperas; al repasar una grabacion se le pone el
        # tiempo de la grabacion (sesion_a_mapa.py)
        self.reloj = time.time
        self._ultimo = {}
        self._pendSensor = []
        self._lock = threading.Lock()
        self._lockDosis = threading.Lock()
        self.salida = None
        if conPip:
            self.salida, razon = _armarSalidaPip(hablarFn, interrumpirFn)
            if self.salida is None:
                print(f"[alertas] pitidos no disponibles ({razon}); "
                      "las alertas criticas van por voz")

    def alternar(self):
        self.activo = not self.activo
        return self.activo

    def activar(self, encendido):
        self.activo = bool(encendido)
        return self.activo

    def sonando(self, margenS=0.5):
        return (self.salida is not None
                and self.salida.recently_active(margenS))

    # ---- señales directas de hardware (ultrasonido) ----
    def reportarSensor(self, direccion="frente", distanciaM=None,
                       tipo="obstaculo"):
        """llamable desde el hilo del sensor: pip INMEDIATO (no espera
        al ciclo de vision) y la alerta se entrega al navegador en el
        siguiente evaluar() pa que pause/recalcule. va dosificado con
        la misma llave que las alertas del mapa (la misma pared no
        suena dos veces)."""
        if not self.activo:
            return False
        critico = distanciaM is None or distanciaM <= self._umbralCritica()
        self._enCritica = critico
        prio = 0 if critico else 1
        if not self._pasa((prio, direccion, "obstaculo"), prio,
                          self.reloj(), distanciaM):
            return False
        if self.salida is not None:
            self.salida.pip(self.salida.CRITICAL if critico
                            else self.salida.WARNING,
                            direccion, tipo, distanciaM)
        dist = "" if distanciaM is None else f" a {int(distanciaM*100)} centímetros"
        texto = f"Ultrasonido: {tipo} {direccion}{dist}."
        if self.salida is None and critico and self.hablarFn:
            donde = {"frente": "al frente"}.get(direccion,
                                                f"a la {direccion}")
            self.hablarFn(f"Atención: {nombreEs(tipo, None)} {donde}"
                          f"{dist}. Deténgase.")
        with self._lock:
            self._pendSensor.append((prio, texto))
        return True

    # ---- evaluacion por ciclo ----
    def _umbralCritica(self):
        return self.criticaM + (HISTERESIS_M if self._enCritica else 0.0)

    def _caminando(self):
        try:
            return bool(self.caminandoFn())
        except Exception:
            return True

    def _pasa(self, tipo, prio, ahora, dist=None):
        with self._lockDosis:
            if dist is not None and len(tipo) == 3:
                # la MISMA cosa quieta a veces cambia de lado (los dos
                # ultrasonicos se turnan quien la ve mas cerca): si ya se
                # aviso de algo a esa distancia y sigue en su espera, este
                # aviso sobra. una precaucion nunca calla a una critica, y
                # entre criticas solo si es practicamente la misma lectura
                tol = QUIETO_M if prio == 1 else 0.06
                for otra, (t0, d0, esp) in self._ultimo.items():
                    if (otra != tipo and len(otra) == 3 and otra[0] <= prio
                            and otra[2] == tipo[2] and d0 is not None
                            and abs(dist - d0) < tol and ahora - t0 < esp):
                        return False
            ult = self._ultimo.get(tipo)
            espera = _COOLDOWN[prio]
            if ult is not None:
                t0, d0, esperaAnt = ult
                # las distancias de la camara brincan mas que las del
                # ultrasonido: de lejos se tolera el 25 %
                quieto = (dist is not None and d0 is not None
                          and abs(dist - d0) < max(QUIETO_M, 0.25 * d0))
                if quieto and self.caminandoFn is not None \
                        and not self._caminando():
                    # nadie se mueve: ya se dijo, repetirlo no ayuda
                    return False
                if ahora - t0 < (esperaAnt if quieto else _COOLDOWN[prio]):
                    return False
                if quieto:
                    espera = min(esperaAnt * 2.0, COOLDOWN_MAX_S)
            self._ultimo[tipo] = (ahora, dist, espera)
            return True

    def _recalcularNav(self, navFrame, frameAl, ahora):
        """¿el navegador tiene que decidir YA? eso va con los umbrales
        del documento (0.80 m), suene o no la alerta: bajarle a lo que
        se oye no debe hacer mas lento al navegador."""
        hay = False
        for prio, texto in safetyAlerts(frameAl):
            if prio == 0 and self._pasa(("nav", _dirDeTexto(texto)), 0,
                                        ahora):
                hay = True
        eco = _alertaEcoUltrasonico(navFrame)
        if eco is not None and eco["prio"] == 0 and self._pasa(
                ("nav", eco["dir"]), 0, ahora, eco["dist"]):
            hay = True
        return hay

    def evaluar(self, navFrame):
        """-> (lista de (prioridad, texto), recalcular). ademas ENTREGA
        el sonido: pip por cada alerta nueva (critica ademas se habla).
        recalcular = el navegador debe decidir ya (Interaccion.ciclo);
        va dosificado igual que las alertas."""
        if not self.activo:
            return [], False
        ahora = self.reloj()
        crudas = []        # dicts con estructura pa poder sonar bien
        recalcular = False

        if navFrame is not None:
            # validacion cruzada: lo que el ultrasonido CONTRADIJO no
            # dispara critica con la distancia de la camara; queda como
            # "posible" (rnf-09: avisar, pero sin gritar)
            contradichas = [d for d in navFrame.detections
                            if d.confirmado == "contradicho"]
            frameAl = navFrame
            if contradichas:
                frameAl = replace(navFrame, detections=[
                    d for d in navFrame.detections
                    if d.confirmado != "contradicho"])
            # con la pose poco confiable el mapa esta desalineado (la
            # pared del fondo aparece donde no es): ahi solo cuentan los
            # sensores EN VIVO, el mapa no dispara alertas
            if navFrame.slam.confidence < CONF_MIN_MAPA:
                frameAl = replace(frameAl, d_front=float("inf"),
                                  d_left=float("inf"),
                                  d_right=float("inf"))
            recalcular = self._recalcularNav(navFrame, frameAl, ahora)
            if not self.delMapa:
                # lo que SUENA: sin la memoria del mapa (el navegador,
                # arriba, si la uso)
                frameAl = replace(frameAl, d_front=float("inf"),
                                  d_left=float("inf"), d_right=float("inf"))
            ecoUS = _alertaEcoUltrasonico(navFrame, self._umbralCritica(),
                                          self.precaucionM)
            for prio, texto, sector, dist in alertasDeObstaculo(
                    frameAl, self.criticaM, self.precaucionM):
                # con la distancia, lo que no se mueve tambien se va
                # espaciando (antes las del mapa repetian cada 2.5 / 8 s)
                crudas.append({"prio": prio, "texto": texto,
                               "tipo": "obstaculo",
                               "dir": _SECTOR_A_DIR[sector],
                               "dist": dist, "grupo": "obstaculo"})

            sectoresMovil = set()
            moviles = []
            for det in navFrame.detections:
                if not det.moving or det.range_est is None:
                    continue
                sector = _sectorDe(det.bearing)
                if sector is None or det.range_est > 2.0:
                    continue
                quien = "una persona" if det.label == "person" else "algo"
                donde = _NOMBRE_SECTOR[sector]
                sectoresMovil.add(donde)
                # "alto" solo si viene de FRENTE: alguien que camina a tu
                # lado (quien te acompaña) no va a chocar contigo y decia
                # "Alto. Se acerca una persona..." a cada rato
                prio = 0 if det.range_est < 1.0 and sector == "front" else 1
                txt = (f"Alto. Se acerca {quien} {donde}, muy cerca."
                       if prio == 0 else
                       f"Precaución: {quien} en movimiento {donde}.")
                moviles.append({"prio": prio, "texto": txt,
                                "tipo": det.label,
                                "dir": _SECTOR_A_DIR[sector],
                                "dist": det.range_est, "grupo": "movil"})
            crudas = [c for c in crudas
                      if not any(d in c["texto"] for d in sectoresMovil)]
            crudas += moviles

            if ecoUS is not None:
                # el ultrasonido mide mas fino que el mapa: su aviso
                # sustituye al generico de ese lado
                crudas = [c for c in crudas if c["dir"] != ecoUS["dir"]
                          or c["tipo"] not in ("obstaculo",)]
                crudas.append(ecoUS)

            for det in contradichas:
                if det.range_est is None or det.range_est > self.precaucionM:
                    continue
                if _sectorDe(det.bearing) != "front":
                    continue
                crudas.append({"prio": 1,
                               "texto": f"Posible {nombreEs(det.label, None)} "
                                        "al frente (solo cámara).",
                               "tipo": det.label, "dir": "frente",
                               "dist": det.range_est})

            for det in navFrame.detections:
                if not det.hazard or det.range_est is None:
                    continue
                if _sectorDe(det.bearing) != "front" or det.range_est > 2.5:
                    continue
                pasos = max(1, round(det.range_est / 0.7))
                nombre = {"hoyo": "un hoyo", "agua": "agua",
                          "escalera": "una escalera"}.get(det.label,
                                                          det.label)
                prio = 0 if det.range_est < 1.3 else 1
                txt = (f"Cuidado: {nombre} justo adelante. Deténgase."
                       if prio == 0 else
                       f"Atención: {nombre} adelante, a unos {pasos} pasos.")
                crudas.append({"prio": prio, "texto": txt,
                               "tipo": det.label, "dir": "frente",
                               "dist": det.range_est, "grupo": "peligro"})

        salida = []
        for c in sorted(crudas, key=lambda c: c["prio"]):
            # la llave NO sale del texto (acentos, cm que cambian, frases
            # distintas pa la misma pared): es que tan grave, de que lado
            # y que clase de cosa
            tipo = (c["prio"], c["dir"], c.get("grupo", "obstaculo"))
            if self._pasa(tipo, c["prio"], ahora, c.get("dist")):
                salida.append((c["prio"], c["texto"]))
                if c["prio"] == 0:
                    recalcular = True
                if self.salida is not None:
                    self.salida.pip(self.salida.CRITICAL if c["prio"] == 0
                                    else self.salida.WARNING,
                                    c["dir"], c["tipo"], c["dist"])
                elif c["prio"] == 0 and self.hablarFn:
                    self.hablarFn(c["texto"])   # sin pips: voz como antes

        with self._lock:
            pendientes, self._pendSensor = self._pendSensor, []
        for prio, texto in pendientes:
            salida.append((prio, texto))
            if prio == 0:
                recalcular = True
        return salida, recalcular


def _dirDeTexto(texto):
    for sec, frase in _NOMBRE_SECTOR.items():
        if frase in texto:
            return _SECTOR_A_DIR[sec]
    return "frente"


def _ladoYTipo(lecturas):
    """con los 2 frontales: (rango, direccion, tipo) del eco mas
    cercano. ambos parejos = superficie de frente (pared/puerta/vidrio);
    solo uno = algo cargado a ese lado."""
    from montaje import bearingEstereo
    porLado = {u.lado: u for u in lecturas
               if getattr(u, "valid", False) and u.lado}
    if not porLado:
        sueltas = [u for u in lecturas if getattr(u, "valid", False)]
        if not sueltas:
            return None
        u = min(sueltas, key=lambda u: u.range_m)
        return u.range_m, "frente", "obstaculo"
    if "izq" in porLado and "der" in porLado:
        rI, rD = porLado["izq"].range_m, porLado["der"].range_m
        if abs(rI - rD) < 0.3:
            b = bearingEstereo(rI, rD)
            direccion = ("frente" if abs(b) < math.radians(5)
                         else "izquierda" if b > 0 else "derecha")
            tipo = "pared" if abs(rI - rD) < 0.08 else "obstaculo"
            return min(rI, rD), direccion, tipo
        lado = "izq" if rI < rD else "der"
    else:
        lado = next(iter(porLado))
    return (porLado[lado].range_m,
            "izquierda" if lado == "izq" else "derecha", "obstaculo")


def _alertaEcoUltrasonico(navFrame, critica=None, precaucion=None):
    """eco cercano que NINGUNA deteccion confirmada explica: es algo
    fisico que la camara no ve (vidrio, mampara, pared lisa). sin
    distancias = las del documento (lo que usa el navegador)."""
    from navigation_feed import CAUTION_DISTANCE_M
    critica = CRITICAL_DISTANCE_M if critica is None else critica
    precaucion = CAUTION_DISTANCE_M if precaucion is None else precaucion
    lect = navFrame.ultrasonic or []
    info = _ladoYTipo(lect)
    if info is None:
        return None
    r, direccion, tipo = info
    if r > precaucion:
        return None
    for d in navFrame.detections:
        if d.confirmado == "ambos" and d.range_est is not None \
                and abs(d.range_est - r) < 0.4:
            return None            # ya lo explica la vision (confirmado)
    nombre = {"pared": "pared", "obstaculo": "obstáculo"}[tipo]
    donde = {"frente": "al frente", "izquierda": "adelante a la izquierda",
             "derecha": "adelante a la derecha"}[direccion]
    cm = int(r * 100)
    if r <= critica:
        return {"prio": 0, "texto": f"Atención: {nombre} {donde} a {cm} "
                                    "centímetros. Deténgase.",
                "tipo": tipo, "dir": direccion, "dist": r}
    return {"prio": 1, "texto": f"Precaución: {nombre} {donde}.",
            "tipo": tipo, "dir": direccion, "dist": r}


def conectarUltrasonido(modAlertas, lecturas, umbralM=None, pitchDeg=None):
    """refuerzo inmediato con los MB1232 (mas rapidos que la camara):
    un eco mas cerca que umbralM (None = la distancia de precaucion del
    modulo) pita AL MOMENTO, sin esperar al ciclo de vision. ojo con el
    montaje frontal: con la cabeza agachada el haz rebota en el piso,
    eso se descarta con el pitch de la imu."""
    from montaje import esEcoDePiso
    if umbralM is None:
        umbralM = modAlertas.precaucionM
    utiles = [u for u in lecturas if getattr(u, "valid", False)
              and not esEcoDePiso(u.range_m, pitchDeg)]
    info = _ladoYTipo(utiles)
    if info is None:
        return
    r, direccion, tipo = info
    if r <= umbralM:
        modAlertas.reportarSensor(direccion, r, tipo=tipo)


class BotonPullUp:
    """boton fisico de la pi: GPIO17 a tierra con el pull-up INTERNO
    (hay que pedirlo aqui, el sistema no lo deja puesto) y su led en
    GPIO27 (docs/GAFAS_HARDWARE.md). UN click = alClick (empezar o
    dejar de escuchar); DOBLE click = alCambiar (prender o apagar las
    alertas). el click sencillo se confirma hasta que pasa la ventana
    del doble, por eso tarda ventanaS en hacer efecto."""

    def __init__(self, pin=17, alCambiar=None, ventanaS=0.45,
                 alClick=None, pinLed=27, usarGpio=True):
        self.pin = pin
        self.alCambiar = alCambiar
        self.alClick = alClick
        self.ventanaS = ventanaS
        self._tUltimo = 0.0
        self._timer = None
        self._lock = threading.Lock()
        self.disponible = False
        self.razon = ""
        self._boton = self._led = None
        if not usarGpio:
            return
        try:
            from gpiozero import Button, LED
            self._boton = Button(pin, pull_up=True, bounce_time=0.05)
            self._boton.when_pressed = self.pulsado
            self.disponible = True
            if pinLed is not None:
                self._led = LED(pinLed)
        except Exception as e:      # sin gpiozero o no es una pi
            self.razon = str(e)

    def led(self, encendido):
        if self._led is not None:
            try:
                self._led.on() if encendido else self._led.off()
            except Exception:
                pass

    def parpadear(self, veces=2, cadaS=0.12, apagadoS=None):
        """veces=None: sigue parpadeando hasta led(True/False)."""
        if self._led is not None:
            try:
                self._led.blink(on_time=cadaS, off_time=apagadoS or cadaS,
                                n=veces)
            except Exception:
                pass

    def pulsado(self):
        ahora = time.time()
        with self._lock:
            doble = (ahora - self._tUltimo) < self.ventanaS
            self._tUltimo = 0.0 if doble else ahora
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if not doble and self.alClick is not None:
                self._timer = threading.Timer(self.ventanaS, self.alClick)
                self._timer.daemon = True
                self._timer.start()
        if doble and self.alCambiar:
            self.alCambiar()

    def cerrar(self):
        for d in (self._boton, self._led):
            try:
                if d is not None:
                    d.close()
            except Exception:
                pass
