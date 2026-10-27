"""
alertas.py — modulo de alertas del TT (alg. 7): el guardian que SIEMPRE
esta corriendo, pida rutas o no. avisa si vas a chocar, si hay
hoyo/escalera adelante o si se te atraviesa alguien muy cerca, y pide
recalcular ruta cuando algo critico aparece. dosifica por TIPO de
alerta (umbral prudente, que avise sin estar chingue y chingue) y se
puede apagar/prender: tecla en la lap, doble click de boton en la pi.
"""
import math
import time

from navigation_feed import safetyAlerts, _sectorOf as _sectorDe


# cooldowns por tipo de alerta (s): critica repite rapido, el resto no
_COOLDOWN = {0: 2.5, 1: 8.0}

_NOMBRE_SECTOR = {"front": "al frente", "left": "a la izquierda",
                  "right": "a la derecha"}


class ModuloAlertas:
    def __init__(self, activo=True):
        self.activo = activo
        self._ultimo = {}      # tipo -> timestamp

    def alternar(self):
        self.activo = not self.activo
        return self.activo

    def _pasa(self, tipo, prio, ahora):
        if ahora - self._ultimo.get(tipo, 0.0) < _COOLDOWN[prio]:
            return False
        self._ultimo[tipo] = ahora
        return True

    def evaluar(self, navFrame):
        """-> (lista de (prioridad, texto), recalcular). prioridad 0
        interrumpe el tts y fuerza decision inmediata del navegador."""
        if not self.activo or navFrame is None:
            return [], False
        ahora = time.time()
        crudas = list(safetyAlerts(navFrame))

        # persona u objeto EN MOVIMIENTO muy cerca (y quitamos la alerta
        # generica de ese sector, esta es mas util)
        moverAlertas = []
        sectoresMovil = set()
        for det in navFrame.detections:
            if not det.moving or det.range_est is None:
                continue
            sector = _sectorDe(det.bearing)
            if sector is None or det.range_est > 2.0:
                continue
            quien = "una persona" if det.label == "person" else "algo"
            donde = _NOMBRE_SECTOR[sector]
            sectoresMovil.add(donde)
            if det.range_est < 1.0:
                moverAlertas.append((0, f"Alto. Se acerca {quien} {donde}, "
                                        "muy cerca."))
            else:
                moverAlertas.append((1, f"Precaución: {quien} en "
                                        f"movimiento {donde}."))
        crudas = [(p, t) for (p, t) in crudas
                  if not any(d in t for d in sectoresMovil)]
        crudas += moverAlertas

        # peligro de piso adelante (hoyo/agua/escalera), con nombre
        for det in navFrame.detections:
            if not det.hazard or det.range_est is None:
                continue
            if _sectorDe(det.bearing) != "front" or det.range_est > 2.5:
                continue
            pasos = max(1, round(det.range_est / 0.7))
            nombre = {"hoyo": "un hoyo", "agua": "agua",
                      "escalera": "una escalera"}.get(det.label,
                                                      det.label)
            if det.range_est < 1.3:
                crudas.append((0, f"Cuidado: {nombre} justo adelante. "
                                  "Deténgase."))
            else:
                crudas.append((1, f"Atención: {nombre} adelante, a unos "
                                  f"{pasos} pasos."))

        # dosificar por tipo (los textos traen numeros que cambian cada
        # frame, asi que el antispam por texto exacto no servia)
        salida = []
        recalcular = False
        for prio, texto in sorted(crudas):
            tipo = (prio, texto.split(":")[0].split(".")[0])
            if self._pasa(tipo, prio, ahora):
                salida.append((prio, texto))
            if prio == 0:
                recalcular = True
        return salida, recalcular


class BotonPullUp:
    """boton fisico de la pi (gpio con pull-up): DOBLE click = alternar
    alertas. TODO al conectar el hardware: descomentar gpiozero y
    elegir el pin (el diseño usa GPIO17 pal boton de sesion; este puede
    ir en otro, p.ej. GPIO27)."""

    def __init__(self, pin=27, alCambiar=None, ventanaS=0.6):
        self.pin = pin
        self.alCambiar = alCambiar
        self.ventanaS = ventanaS
        self._tUltimo = 0.0
        self.disponible = False
        # TODO hardware:
        # from gpiozero import Button
        # self._btn = Button(pin, pull_up=True)
        # self._btn.when_pressed = self.pulsado
        # self.disponible = True

    def pulsado(self):
        ahora = time.time()
        doble = (ahora - self._tUltimo) < self.ventanaS
        self._tUltimo = ahora
        if doble and self.alCambiar:
            self.alCambiar()
