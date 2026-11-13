"""
nombres.py — de la etiqueta interna (clases de yolo coco/open images en
ingles + las nuestras) al nombre en español que se le DICE al usuario.
todo lo hablado pasa por aqui: llegada, "que hay enfrente", alertas.
"""

# etiqueta -> (articulo, nombre). articulo: el/la/los/las
_ES = {
    # coco (con los alias de obstacle_logic ya aplicados)
    "person": ("la", "persona"), "bicycle": ("la", "bicicleta"),
    "car": ("el", "carro"), "motorcycle": ("la", "moto"),
    "airplane": ("el", "avión"), "bus": ("el", "autobús"),
    "train": ("el", "tren"), "truck": ("el", "camión"),
    "boat": ("el", "bote"), "traffic light": ("el", "semáforo"),
    "fire hydrant": ("el", "hidrante"), "stop sign": ("el", "alto"),
    "parking meter": ("el", "parquímetro"), "bench": ("la", "banca"),
    "bird": ("el", "pájaro"), "cat": ("el", "gato"),
    "dog": ("el", "perro"), "horse": ("el", "caballo"),
    "sheep": ("la", "oveja"), "cow": ("la", "vaca"),
    "elephant": ("el", "elefante"), "bear": ("el", "oso"),
    "zebra": ("la", "cebra"), "giraffe": ("la", "jirafa"),
    "backpack": ("la", "mochila"), "umbrella": ("el", "paraguas"),
    "handbag": ("la", "bolsa"), "tie": ("la", "corbata"),
    "suitcase": ("la", "maleta"), "frisbee": ("el", "frisbee"),
    "skis": ("los", "esquís"), "snowboard": ("la", "tabla"),
    "sports ball": ("la", "pelota"), "kite": ("el", "papalote"),
    "baseball bat": ("el", "bate"), "baseball glove": ("el", "guante"),
    "skateboard": ("la", "patineta"), "surfboard": ("la", "tabla de surf"),
    "tennis racket": ("la", "raqueta"), "bottle": ("la", "botella"),
    "wine glass": ("la", "copa"), "cup": ("la", "taza"),
    "fork": ("el", "tenedor"), "knife": ("el", "cuchillo"),
    "spoon": ("la", "cuchara"), "bowl": ("el", "tazón"),
    "banana": ("el", "plátano"), "apple": ("la", "manzana"),
    "sandwich": ("el", "sándwich"), "orange": ("la", "naranja"),
    "broccoli": ("el", "brócoli"), "carrot": ("la", "zanahoria"),
    "hot dog": ("el", "hot dog"), "pizza": ("la", "pizza"),
    "donut": ("la", "dona"), "cake": ("el", "pastel"),
    "chair": ("la", "silla"), "sofa": ("el", "sillón"),
    "potted plant": ("la", "planta"), "bed": ("la", "cama"),
    "table": ("la", "mesa"), "toilet": ("el", "baño"),
    "monitor": ("la", "pantalla"), "laptop": ("la", "laptop"),
    "mouse": ("el", "mouse"), "remote": ("el", "control"),
    "keyboard": ("el", "teclado"), "cell phone": ("el", "celular"),
    "microwave": ("el", "microondas"), "oven": ("el", "horno"),
    "toaster": ("el", "tostador"), "sink": ("el", "lavabo"),
    "fridge": ("el", "refrigerador"), "book": ("el", "libro"),
    "clock": ("el", "reloj"), "vase": ("el", "florero"),
    "scissors": ("las", "tijeras"), "teddy bear": ("el", "peluche"),
    "hair drier": ("la", "secadora"), "toothbrush": ("el", "cepillo"),
    # detector extra (open images v7)
    "exit sign": ("el", "letrero de salida"),
    "door": ("la", "puerta"), "window": ("la", "ventana"),
    "trash can": ("el", "bote de basura"), "street light": ("el", "poste"),
    "sign": ("el", "letrero"), "ladder": ("la", "escalera de mano"),
    "shelf": ("el", "estante"), "tree": ("el", "árbol"),
    "fountain": ("la", "fuente"), "desk": ("el", "escritorio"),
    "cabinet": ("el", "archivero"), "counter": ("el", "mostrador"),
    "whiteboard": ("el", "pizarrón"),
    # segmentacion / sensores / internas
    "escalera": ("la", "escalera"), "hoyo": ("el", "hoyo"),
    "agua": ("el", "agua"), "pared": ("la", "pared"),
    "obstaculo": ("el", "obstáculo"), "eco_us": ("el", "obstáculo"),
    "punto": ("el", "punto indicado"),
}

_INDEF = {"el": "un", "la": "una", "los": "unos", "las": "unas"}


def nombreEs(label, articulo="def"):
    """articulo: "def" (la puerta), "indef" (una puerta) o None (puerta).
    etiqueta desconocida: se regresa tal cual (mejor eso que inventar)."""
    if not label:
        return ""
    art, nombre = _ES.get(label, ("", label.replace("_", " ")))
    if articulo is None or not art:
        return nombre
    if articulo == "indef":
        # "un agua" no: el agua es femenino con "el" (a tonica)
        return ("agua" if nombre == "agua" else f"{_INDEF[art]} {nombre}")
    return f"{art} {nombre}"


def conArticulo(palabra):
    """pa lo que DIJO el usuario ("salida" -> "la salida"): si coincide
    con un nombre conocido usa su articulo, si no adivina por la
    terminacion (lo comun en español)."""
    p = (palabra or "").strip().lower()
    if not p:
        return ""
    if p.split()[0] in ("el", "la", "los", "las", "al"):
        return p
    for art, nombre in _ES.values():
        if nombre == p:
            return f"{art} {p}"
    if p.endswith(("as",)):
        return f"las {p}"
    if p.endswith(("os", "es")):
        return f"los {p}"
    if p.endswith(("a", "ión", "ion", "dad", "tad")):
        return f"la {p}"
    return f"el {p}"
