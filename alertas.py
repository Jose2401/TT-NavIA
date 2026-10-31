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
conectan los INFRARROJOS de la pi pa lo muy cercano entre frames.
"""
import math
import threading
import time

from navigation_feed import safetyAlerts, _sectorOf as _sectorDe

_COOLDOWN = {0: 2.5, 1: 8.0}
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

    salida = SalidaPip(on_interrupt=interrumpirFn, on_speak=_hablar)
    salida.start()
    salida.CRITICAL = pipmod.CRITICAL
    salida.WARNING = pipmod.WARNING
    return salida, None


class ModuloAlertas:
    def __init__(self, activo=True, hablarFn=None, interrumpirFn=None,
                 conPip=True):
        """hablarFn(texto): voz pa la alerta critica (p.ej. ColaVoz p0).
        conPip=False: sin audio (tests / --sin-pip), las criticas se
        hablan con hablarFn como antes."""
        self.activo = activo
        self.hablarFn = hablarFn
        self._ultimo = {}
        self._pendSensor = []
        self._lock = threading.Lock()
        self.salida = None
        if conPip:
            self.salida, razon = _armarSalidaPip(hablarFn, interrumpirFn)
            if self.salida is None:
                print(f"[alertas] pitidos no disponibles ({razon}); "
                      "las alertas criticas van por voz")

    def alternar(self):
        self.activo = not self.activo
        return self.activo

    def sonando(self, margenS=0.5):
        return (self.salida is not None
                and self.salida.recently_active(margenS))

    # ---- señales directas de hardware (infrarrojos, ultrasonido...) ----
    def reportarSensor(self, direccion="frente", distanciaM=None,
                       tipo="obstaculo"):
        """llamable desde el hilo del sensor: pip INMEDIATO (no espera
        al ciclo de vision) y la alerta se entrega al navegador en el
        siguiente evaluar() pa que pause/recalcule."""
        if not self.activo:
            return
        critico = distanciaM is None or distanciaM < 1.0
        if self.salida is not None:
            self.salida.pip(self.salida.CRITICAL if critico
                            else self.salida.WARNING,
                            direccion, tipo, distanciaM)
        dist = "" if distanciaM is None else f" a {int(distanciaM*100)} centímetros"
        texto = f"Ultrasonido: {tipo} {direccion}{dist}."
        with self._lock:
            self._pendSensor.append((0 if critico else 1, texto))

    # ---- evaluacion por ciclo ----
    def _pasa(self, tipo, prio, ahora):
        if ahora - self._ultimo.get(tipo, 0.0) < _COOLDOWN[prio]:
            return False
        self._ultimo[tipo] = ahora
        return True

    def evaluar(self, navFrame):
        """-> (lista de (prioridad, texto), recalcular). ademas ENTREGA
        el sonido: pip por cada alerta nueva (critica ademas se habla)."""
        if not self.activo:
            return [], False
        ahora = time.time()
        crudas = []        # dicts con estructura pa poder sonar bien

        if navFrame is not None:
            for prio, texto in safetyAlerts(navFrame):
                direccion = "frente"
                for sec, frase in _NOMBRE_SECTOR.items():
                    if frase in texto:
                        direccion = _SECTOR_A_DIR[sec]
                crudas.append({"prio": prio, "texto": texto,
                               "tipo": "obstaculo", "dir": direccion,
                               "dist": None})

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
                prio = 0 if det.range_est < 1.0 else 1
                txt = (f"Alto. Se acerca {quien} {donde}, muy cerca."
                       if prio == 0 else
                       f"Precaución: {quien} en movimiento {donde}.")
                moviles.append({"prio": prio, "texto": txt,
                                "tipo": det.label,
                                "dir": _SECTOR_A_DIR[sector],
                                "dist": det.range_est})
            crudas = [c for c in crudas
                      if not any(d in c["texto"] for d in sectoresMovil)]
            crudas += moviles

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
                               "dist": det.range_est})

        salida = []
        recalcular = False
        for c in sorted(crudas, key=lambda c: c["prio"]):
            tipo = (c["prio"], c["texto"].split(":")[0].split(".")[0])
            if self._pasa(tipo, c["prio"], ahora):
                salida.append((c["prio"], c["texto"]))
                if self.salida is not None:
                    self.salida.pip(self.salida.CRITICAL if c["prio"] == 0
                                    else self.salida.WARNING,
                                    c["dir"], c["tipo"], c["dist"])
                elif c["prio"] == 0 and self.hablarFn:
                    self.hablarFn(c["texto"])   # sin pips: voz como antes
            if c["prio"] == 0:
                recalcular = True

        with self._lock:
            pendientes, self._pendSensor = self._pendSensor, []
        for prio, texto in pendientes:
            salida.append((prio, texto))
            if prio == 0:
                recalcular = True
        return salida, recalcular


def conectarUltrasonido(modAlertas, lecturas, umbralM=1.5):
    """refuerzo con los ULTRASONICOS MB1232 de la pi: una lectura
    valida mas cerca que umbralM dispara pip de inmediato (los MB1232
    van a 40 Hz, mucho mas rapido que la camara). llamar en el loop
    del main de la pi con lo que devuelva Mb1232Array.read(); en
    cuanto el hardware exista esto queda andando solo."""
    for u in lecturas:
        if not getattr(u, "valid", False) or u.range_m > umbralM:
            continue
        if u.sensor_bearing > 0.3:
            direccion = "izquierda"
        elif u.sensor_bearing < -0.3:
            direccion = "derecha"
        else:
            direccion = "frente"
        modAlertas.reportarSensor(direccion, u.range_m,
                                  tipo="obstaculo")


class BotonPullUp:
    """boton fisico de la pi (gpio pull-up): DOBLE click alterna las
    alertas. TODO hardware: gpiozero Button(pin, pull_up=True) y
    when_pressed = self.pulsado (pin sugerido: GPIO27)."""

    def __init__(self, pin=27, alCambiar=None, ventanaS=0.6):
        self.pin = pin
        self.alCambiar = alCambiar
        self.ventanaS = ventanaS
        self._tUltimo = 0.0
        self.disponible = False

    def pulsado(self):
        ahora = time.time()
        doble = (ahora - self._tUltimo) < self.ventanaS
        self._tUltimo = ahora
        if doble and self.alCambiar:
            self.alCambiar()
