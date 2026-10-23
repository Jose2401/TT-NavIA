from PLN_oneimport import DESCONOCIDO, SimpleRandom

ASSISTANT_NAME = "Guía"      # aun pensando en el nombre xd
STEP_LENGTH_M = 0.7          # longitud aproximada de un paso

_rng = SimpleRandom()


def pick(options):
    return options[int(_rng.random() * len(options))]


def _capitalize(text):
    return text[:1].upper() + text[1:] if text else text



# Presentación
def introduction():
    return (
        f"¡Hola! Soy {ASSISTANT_NAME}, tu guía de navegación. "
        "Voy a acompañarte en tu recorrido y a contarte lo que hay a tu alrededor. "
        "Puedes pedirme cosas como: llévame a la biblioteca, ¿qué hay enfrente?, "
        "o ¿dónde estoy? Y si necesitas parar, solo di: detente. "
        "Cuando quieras, dime a dónde vamos."
    )


def goodbye():
    return pick(["Hasta pronto, cuídate mucho.", "Fue un gusto acompañarte. ¡Hasta luego!"])



# Nombres de lugares -el PLN los guarda 

DISPLAY_NAMES = {
    "biblioteca": "la biblioteca",
    "laboratorio": "el laboratorio",
    "cafeteria": "la cafetería",
    "entrada": "la entrada",
    "salida": "la salida",
    "banos": "los baños",
    "escaleras": "las escaleras",
    "elevador": "el elevador",
    "escom": "ESCOM",
    "ipn_politecnico": "el Poli",
    "metro_politecnico": "el metro Politécnico",
    "alameda": "la Alameda",
    "angel_de_la_independencia": "el Ángel de la Independencia",
}


def display_name(destination):
    if not destination:
        return "ese lugar"
    return DISPLAY_NAMES.get(destination, destination.replace("_", " "))


# Respuestas a comandos de voz

def friendly_response(result):
    intent = result["intencion"]
    place = display_name(result.get("destino"))

    if intent == "DESTINO":
        return pick([
            f"¡Claro! Vamos hacia {place}. Yo te voy diciendo por dónde.",
            f"Perfecto, nos dirigimos a {place}. Camina con confianza, yo te guío.",
            f"Muy bien, te llevo a {place}. Vamos paso a paso.",
        ])
    if intent == "PAUSA":
        return pick(["Va, hacemos una pausa. Avísame cuando quieras seguir.",
                     "Listo, nos detenemos un momento. Aquí te espero."])
    if intent == "REANUDAR":
        return pick(["¡Seguimos! Retomo la ruta.", "Perfecto, continuamos con el recorrido."])
    if intent == "DETENER":
        return pick(["Listo, me detengo. Quédate tranquilo.",
                     "Hecho, detuve la navegación."])
    if intent == "RUTA_ALTERNATIVA":
        return pick(["Claro, déjame buscar otro camino.",
                     "Sin problema, voy a calcular otra ruta para ti."])
    if intent == "EMERGENCIA":
        return "Tranquilo, estoy contigo. Ya estoy avisando que necesitas ayuda."
    if intent == "UBICACION":
        return pick(["Déjame ver dónde estamos.", "Un momento, reviso tu ubicación."])
    if intent == "ENTORNO":
        return pick(["Voy a fijarme en lo que tienes a tu alrededor.",
                     "Dame un segundo y te cuento qué hay cerca."])
    if intent == "REPETIR":
        return pick(["Claro, te lo repito.", "Con gusto, te lo digo otra vez."])
    if intent == "CANCELAR":
        return pick(["Está bien, cancelé la ruta. Dime si quieres ir a otro lado.",
                     "Listo, ruta cancelada."])
    if intent == "AYUDA":
        return ("Puedes decirme: llévame a la biblioteca, qué hay enfrente, dónde estoy, "
                "pausa, continúa, detente o busca otra ruta. Aquí estoy para lo que necesites.")
    if intent == "ESTADO":
        return pick(["Déjame revisar cómo vas.", "Un momento, consulto el avance de tu ruta."])
    if intent == "CONFIRMAR":
        return pick(["Perfecto.", "Entendido."])
    if intent == "NEGAR":
        return pick(["De acuerdo, no lo haré.", "Está bien, lo dejamos así."])
    return pick([
        "Perdona, no te entendí bien. ¿Me lo repites?",
        "Mmm, no alcancé a captarlo. ¿Puedes decirlo de otra forma?",
    ])


def ask_destination():
    return pick(["¿A dónde te gustaría ir?", "Claro, ¿a qué lugar quieres que vayamos?"])


def no_audio_message():
    return "No te escuché. Cuando quieras, vuelve a intentarlo."



def describe_turn(degrees):
    side = "derecha" if degrees > 0 else "izquierda"
    a = abs(degrees)
    if a < 10:
        return None
    if a < 35:
        return pick([f"desvíate un poquito hacia tu {side}",
                     f"gira muy suavemente a tu {side}"])
    if a < 75:
        return pick([f"da una vuelta suave a tu {side}",
                     f"gira sin hacerlo muy cerrado hacia tu {side}"])
    if a < 120:
        return pick([f"gira a tu {side}, como al doblar una esquina",
                     f"da vuelta completa a tu {side}, como una esquina"])
    if a < 165:
        return f"da una vuelta amplia hacia tu {side}, casi regresando"
    return "date la vuelta completamente"


def describe_distance(meters):
    if meters is None:
        return ""
    steps = int(round(meters / STEP_LENGTH_M))
    if steps <= 2:
        return "un par de pasos"
    if meters < 15:
        if steps > 10:
            steps = int(round(steps / 5.0)) * 5
        return f"unos {steps} pasos"
    return f"unos {int(round(meters))} metros"


OPENERS = ["Muy bien", "Perfecto", "Vas muy bien", "Listo", "Excelente"]


def navigation_instruction(turn_degrees=0, distance_m=None, landmark=None):
    turn = describe_turn(turn_degrees)
    distance = describe_distance(distance_m)

    if turn is None:
        if distance:
            body = pick([f"sigue derechito {distance}", f"continúa de frente {distance}"])
        else:
            body = "sigue derechito"
    else:
        body = f"{turn} y luego camina {distance}" if distance else turn

    text = f"{pick(OPENERS)}, {body}."
    if landmark:
        text += f" Ahí vas a encontrar {landmark}."
    return text


def arrival_message(destination):
    place = display_name(destination)
    return pick([f"¡Llegamos! Ya estás en {place}.",
                 f"Lo lograste, esta es {place}." if place.startswith("la ") else f"¡Listo, llegamos a {place}!"])


def obstacle_warning(description="un obstáculo"):
    return pick([f"Cuidado, hay {description} justo enfrente. Frena despacito.",
                 f"Atento, tienes {description} adelante. Avanza con calma."])