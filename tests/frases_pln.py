# set de evaluacion del pln: (frase, intencion esperada). incluye las
# frases que fallaban en la evaluacion del documento (rnf-07: >= 92 %)
FRASES = [
    # destino
    ("llevame a la salida", "DESTINO"), ("quiero ir a la biblioteca", "DESTINO"),
    ("vamos a la cafeteria", "DESTINO"), ("guiame al baño", "DESTINO"),
    ("necesito llegar a la entrada", "DESTINO"), ("ve a la puerta", "DESTINO"),
    ("me puedes llevar al laboratorio", "DESTINO"), ("quiero ir al elevador", "DESTINO"),
    ("dirigeme a las escaleras", "DESTINO"), ("llevame a la cocina", "DESTINO"),
    # pausa
    ("pausa", "PAUSA"), ("espera un momento", "PAUSA"), ("haz una pausa", "PAUSA"),
    ("esperate tantito", "PAUSA"), ("pausa la navegacion", "PAUSA"),
    ("espera", "PAUSA"), ("dame un segundo", "PAUSA"),
    # reanudar
    ("continua", "REANUDAR"), ("sigue", "REANUDAR"), ("reanuda la navegacion", "REANUDAR"),
    ("seguimos", "REANUDAR"), ("ya puedes seguir", "REANUDAR"),
    ("sigue con la ruta", "REANUDAR"), ("vamos de nuevo", "REANUDAR"),
    # detener
    ("detente", "DETENER"), ("alto", "DETENER"), ("ya quiero parar", "DETENER"),
    ("ya no quiero seguir", "DETENER"), ("termina la navegacion", "DETENER"),
    ("deja de guiarme", "DETENER"), ("para ya", "DETENER"),
    ("no quiero continuar", "DETENER"), ("basta", "DETENER"),
    ("ya no sigas", "DETENER"), ("detener la navegacion", "DETENER"),
    ("finaliza el recorrido", "DETENER"),
    # ruta alternativa
    ("busca otra ruta", "RUTA_ALTERNATIVA"), ("dame otro camino", "RUTA_ALTERNATIVA"),
    ("recalcula la ruta", "RUTA_ALTERNATIVA"), ("por ahi no, busca otra ruta", "RUTA_ALTERNATIVA"),
    ("cambia la ruta", "RUTA_ALTERNATIVA"), ("quiero otra ruta", "RUTA_ALTERNATIVA"),
    ("hay otro camino", "RUTA_ALTERNATIVA"),
    # emergencia
    ("auxilio", "EMERGENCIA"), ("ayuda, me caí", "EMERGENCIA"), ("me lastimé", "EMERGENCIA"),
    ("estoy herido", "EMERGENCIA"), ("emergencia", "EMERGENCIA"),
    ("necesito ayuda urgente", "EMERGENCIA"), ("estoy en peligro", "EMERGENCIA"),
    ("me caí", "EMERGENCIA"), ("socorro", "EMERGENCIA"), ("me duele mucho", "EMERGENCIA"),
    # ubicacion
    ("donde estoy", "UBICACION"), ("en que lugar estoy", "UBICACION"),
    ("cual es mi ubicacion", "UBICACION"), ("donde me encuentro", "UBICACION"),
    ("dime donde estoy", "UBICACION"),
    # entorno
    ("que hay enfrente", "ENTORNO"), ("que tengo delante", "ENTORNO"),
    ("describe lo que hay alrededor", "ENTORNO"), ("hay obstaculos", "ENTORNO"),
    ("que hay a mi alrededor", "ENTORNO"), ("que ves", "ENTORNO"),
    # repetir
    ("repite", "REPETIR"), ("que dijiste", "REPETIR"), ("dilo otra vez", "REPETIR"),
    ("no te escuche", "REPETIR"), ("repitelo por favor", "REPETIR"),
    ("como dijiste", "REPETIR"),
    # cancelar
    ("cancela la ruta", "CANCELAR"), ("olvidalo", "CANCELAR"),
    ("cancelar", "CANCELAR"), ("abandona la ruta", "CANCELAR"),
    # ayuda
    ("ayuda", "AYUDA"), ("como te uso", "AYUDA"), ("que puedo hacer", "AYUDA"),
    ("que comandos hay", "AYUDA"), ("como funciona", "AYUDA"),
    ("que puedo decir", "AYUDA"), ("ayudame con los comandos", "AYUDA"),
    # confirmar
    ("si", "CONFIRMAR"), ("ok", "CONFIRMAR"), ("correcto", "CONFIRMAR"),
    ("esta bien", "CONFIRMAR"), ("de acuerdo", "CONFIRMAR"),
    ("claro que si", "CONFIRMAR"), ("vale", "CONFIRMAR"), ("afirmativo", "CONFIRMAR"),
    # negar
    ("no", "NEGAR"), ("no es correcto", "NEGAR"), ("incorrecto", "NEGAR"),
    ("eso no", "NEGAR"), ("no es asi", "NEGAR"), ("negativo", "NEGAR"),
    ("esta mal", "NEGAR"), ("no gracias", "NEGAR"),
    # estado
    ("cuanto falta", "ESTADO"), ("como voy", "ESTADO"), ("falta mucho", "ESTADO"),
    ("cual es mi progreso", "ESTADO"), ("ya casi llego", "ESTADO"),
    ("estado de la ruta", "ESTADO"),
]

# las que NO pueden fallar: cambian el sentido de la orden o son seguridad
CRITICAS = {
    "ya no quiero seguir", "termina la navegacion", "no es correcto",
    "ayuda", "ayuda, me caí", "deja de guiarme", "qué dijiste",
    "que dijiste", "como te uso", "ok", "detente", "alto", "auxilio",
    "me caí", "no quiero continuar", "ya no sigas",
}

# set APARTADO: frases que NO estan en la semilla (mide si generaliza o
# solo memorizo). se usa pa la evaluacion honesta.
APARTADO = [
    ("oye llevame hacia la entrada principal", "DESTINO"),
    ("quisiera ir a la biblioteca por favor", "DESTINO"),
    ("acompañame a la salida", "DESTINO"),
    ("espérame tantito", "PAUSA"), ("un segundito", "PAUSA"),
    ("aguanta tantito", "PAUSA"),
    ("ya podemos seguir", "REANUDAR"), ("continúa por favor", "REANUDAR"),
    ("sigamos", "REANUDAR"),
    ("ya no quiero que me guíes", "DETENER"), ("ya no sigas guiándome", "DETENER"),
    ("terminar la ruta", "DETENER"), ("detén la navegación por favor", "DETENER"),
    ("no quiero seguir caminando", "DETENER"), ("deja de darme indicaciones", "DETENER"),
    ("ya párale", "DETENER"),
    ("busca un camino diferente", "RUTA_ALTERNATIVA"),
    ("por aquí no puedo, otra ruta", "RUTA_ALTERNATIVA"),
    ("ayúdame me caí", "EMERGENCIA"), ("me tropecé y me lastimé", "EMERGENCIA"),
    ("auxilio por favor", "EMERGENCIA"), ("me duele la pierna", "EMERGENCIA"),
    ("en dónde estoy ahorita", "UBICACION"), ("dime mi ubicación", "UBICACION"),
    ("qué hay frente a mí", "ENTORNO"), ("hay algo enfrente", "ENTORNO"),
    ("perdón, qué dijiste", "REPETIR"), ("me lo repites", "REPETIR"),
    ("no te oí", "REPETIR"),
    ("cómo se usa esto", "AYUDA"), ("qué cosas puedo pedirte", "AYUDA"),
    ("okey", "CONFIRMAR"), ("sí, está bien", "CONFIRMAR"), ("perfecto", "CONFIRMAR"),
    ("no, no es correcto", "NEGAR"), ("así no es", "NEGAR"), ("nel", "NEGAR"),
    ("cuánto me falta para llegar", "ESTADO"), ("ya mero llegamos", "ESTADO"),
]
