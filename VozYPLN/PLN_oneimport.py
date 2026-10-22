"""
Módulo de Procesamiento de Lenguaje Natural (PLN).
Proyecto Terminal IPN ESCOM — TT 2026-B045.

  1. Recibir el texto que entrega el reconocimiento de voz.
  2. Normaliza ese texto (minúsculas, tokenización, lematización/stemmingm y
     elimina de palabras sin información) para que en caso de que el usuario tenga
     distintas formas de decir lo mismo se traten de manera equivalente.
  3. Determina la intención del usuario dentro de un conjunto cerrado y
     predecible de acciones de navegación.
  4. Extrae la información relevante de la frase (el destino, cuando la
     intención lo requiere).
  5. Convierte esa interpretación en una acción estructurada
     (intención - entidad - acción) que se entrega al módulo de navegación.

Este módulo PLN no ejecuta funciones físicas del sistema, es decir que  
no mueve nada, no calcula nada de navegacion, no controla cámara, no detecta obstáculos, 
no calcula la ruta. 
Produce la instrucción estructurada. Por eso `process()` siempre
entrega como resultado principal de un proceso donde se impliecan 3 partes 
(intención, destino/entidad, acción); el texto de respuesta hablada que también 
se incluye es solo una sugerencia de conveniencia para pruebas por consola, 
ya que el "módulo de salida" (audio) todavía no existe como módulo independiente.

Intenciones que el usuario puede manejar:
    DESTINO, PAUSA, REANUDAR, DETENER, RUTA_ALTERNATIVA, DESCONOCIDO, EMERGENCIA
Más algunas intenciones de utilidad que EXTIENDEN el diseño base (el
documento aclara que no hace falta reconocer "cualquier intención
imaginable", pero no prohíbe funciones adicionales útiles para la
accesibilidad): UBICACION, ENTORNO, REPETIR, CANCELAR, AYUDA, CONFIRMAR,
NEGAR, ESTADO. Estas están claramente separadas del conjunto núcleo en el
código de abajo.

Aprendizaje: el documento plantea usar TF-IDF para representar el comando
antes de clasificar la intención. Aquí se implementa TF-IDF real (término-
frecuencia por inverso-de-frecuencia-de-documento) desde cero, y además se
añade una red neuronal feedforward (perceptrón multicapa con retro-
propagación) también implementada desde cero — sin numpy ni ninguna
librería externa, solo la biblioteca estándar de Python — que aprende con
esos vectores TF-IDF y se sigue entrenando con la retroalimentación del
usuario.

Nota sobre "sin librerías": ningún `import` de este archivo instala nada
con pip — todos son módulos incluidos con el propio intérprete de Python.
Aun así, se sustituyó cada uno por código propio salvo el único caso en el
que hacerlo tiene un costo real:

  - `unicodedata` -> reemplazado por una tabla de acentos escrita a mano.
  - `os`          -> reemplazado por try/except al abrir el archivo.
  - `collections` -> reemplazado por diccionarios normales con .get()/.setdefault().
  - `random`      -> reemplazado por un generador congruencial lineal (LCG) propio.
  - `re`          -> reemplazado por recorrido manual de caracteres y
                     búsqueda de subcadenas (sin expresiones regulares).
  - `json`        -> reemplazado por un serializador propio, estilo
                     "bencode" (cadenas con longitud explícita + repr()
                     para los flotantes), que guarda/lee la memoria sin
                     depender del módulo json.
  - `math`        -> se conserva. Se midió: reimplementar tanh/exp a mano
                     en Python puro es ~7 veces más lento, y la red
                     neuronal los invoca decenas de millones de veces
                     durante el entrenamiento (una vez por cada neurona,
                     en cada ejemplo, en cada época). Eso convertiría un
                     entrenamiento inicial de ~1 segundo en más de un
                     minuto, e incluso peor en la Raspberry Pi. Ahí sí
                     hay un impacto negativo real, así que se mantiene.
"""

import math



# Manejo de acentos mediante tabla (unicodedata)


_ACCENT_TABLE = str.maketrans(
    "áéíóúüñÁÉÍÓÚÜÑ",
    "aeiouunAEIOUUN",
)


def strip_accents_manual(text):
    return text.translate(_ACCENT_TABLE)


# ============================================================
# REEMPLAZO DE `random` — generador congruencial lineal (LCG)
# ============================================================
# No se necesita calidad criptográfica aquí: el LCG solo se usa para
# romper la simetría al inicializar los pesos de la red y para mezclar
# el orden de los ejemplos de entrenamiento (SGD). Parámetros del LCG de
# 48 bits que usa glibc/POSIX drand48 (bien documentados y con buen
# comportamiento estadístico para este tipo de uso).

class SimpleRandom:

    def __init__(self, seed=None):
        if seed is None:
            # semilla a partir del reloj, sin usar el módulo random/time:
            # se toma de id() de un objeto recién creado, que en CPython
            # depende de la dirección de memoria (cambia entre corridas)
            seed = id(object()) ^ id([])
        self._state = seed & ((1 << 48) - 1)

    def _next_raw(self):
        # constantes del LCG de 48 bits (mismas que POSIX drand48)
        self._state = (25214903917 * self._state + 11) & ((1 << 48) - 1)
        return self._state

    def random(self):
        """Flotante uniforme en [0, 1)."""
        return self._next_raw() / (1 << 48)

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def shuffle(self, sequence):
        """Baraja `sequence` en su lugar (algoritmo de Fisher-Yates)."""
        for i in range(len(sequence) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            if j > i:
                j = i
            sequence[i], sequence[j] = sequence[j], sequence[i]

    def sample(self, population, k):
        """Selecciona k elementos distintos de population, sin reemplazo."""
        pool = list(population)
        k = min(k, len(pool))
        result = []
        for _ in range(k):
            index = int(self.random() * len(pool))
            if index >= len(pool):
                index = len(pool) - 1
            result.append(pool.pop(index))
        return result


_rng = SimpleRandom(seed=20260829)  # semilla fija: corridas reproducibles


# ============================================================
# CONFIGURACIÓN
# ============================================================

LEARNING_FILE = "navigation_nlp_memory.json"

MIN_CONFIDENCE = 0.42
LEARNING_CONFIDENCE = 0.75

MAX_LEARNED_PHRASES = 5000

DESCONOCIDO = "DESCONOCIDO"  # tal como lo nombra el documento

# --- hiperparámetros de la red neuronal ---
HIDDEN_SIZE = 24
LEARNING_RATE = 0.08
L2_REGULARIZATION = 1e-4

BOOTSTRAP_EPOCHS = 80          # entrenamiento inicial (una sola vez)
FEEDBACK_EPOCHS = 25           # entrenamiento al recibir retroalimentación
REPLAY_SAMPLE_SIZE = 30        # ejemplos de "repaso" para no olvidar lo aprendido
NN_TRUST_SATURATION = 400      # ejemplos vistos a partir de los cuales confiamos casi al máximo en la red


# ============================================================
# UTILIDADES DE TEXTO  (normalización / tokenización — responsabilidad #2)
# ============================================================

def remove_accents(text):
    """Elimina acentos manteniendo el significado textual (tabla manual,
    sin unicodedata)."""
    return strip_accents_manual(text)


def normalize_text(text):
    """Normalización general del texto: minúsculas, sin acentos, sin signos.
    Recorre los caracteres a mano en vez de usar expresiones regulares."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = remove_accents(text)

    # sustituye todo lo que no sea letra/dígito/espacio por un espacio
    # (equivalente a re.sub(r"[^a-z0-9\s]", " ", text), pero carácter por
    # carácter en vez de con una expresión regular)
    characters = []
    for char in text:
        if ("a" <= char <= "z") or ("0" <= char <= "9") or char.isspace():
            characters.append(char)
        else:
            characters.append(" ")
    text = "".join(characters)

    # colapsa espacios múltiples: separar por espacios en blanco ya
    # descarta cualquier cantidad de espacios repetidos, así que unir con
    # un solo espacio logra lo mismo que re.sub(r"\s+", " ", text)
    return " ".join(text.split())


def tokenize(text):
    """Convierte texto en tokens (palabras)."""
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


def _count_occurrences(tokens):
    """Cuenta ocurrencias de cada elemento (reemplazo de collections.Counter)."""
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


# ============================================================
# DISTANCIA LEVENSHTEIN (tolerancia a errores del reconocimiento de voz)
# ============================================================

def levenshtein(a, b):
    """Distancia de Levenshtein implementada desde cero (sin librerías)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def similarity(a, b):
    """Similaridad entre 0 y 1 basada en Levenshtein."""
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    distance = levenshtein(a, b)
    maximum = max(len(a), len(b))
    return 1.0 - (distance / maximum)


# ============================================================
# STEMMING SIMPLE (lematización aproximada que pide el documento)
# ============================================================

class SimpleStemmer:

    def __init__(self):
        self.suffixes = [
            "amientos", "imientos", "aciones", "iciones", "mente",
            "ando", "iendo", "ados", "idos", "adas", "idas",
            "es", "os", "as", "o", "a",
        ]

    def stem(self, word):
        word = normalize_text(word)
        if len(word) <= 4:
            return word
        for suffix in self.suffixes:
            if word.endswith(suffix):
                candidate = word[:-len(suffix)]
                if len(candidate) >= 3:
                    return candidate
        return word


# ============================================================
# STOPWORDS (palabras sin información relevante que se descartan)
# ============================================================

STOPWORDS = {
    "a", "al", "el", "la", "los", "las", "de", "del", "en", "por", "para",
    "con", "que", "me", "te", "se", "mi", "un", "una", "unos", "unas",
    "quiero", "necesito", "puedes", "puede", "favor", "porfavor",
    "hacia", "como", "donde", "pues",
}


# ============================================================
# INTENCIONES  (responsabilidad #3: conjunto cerrado y predecible)
# ============================================================

INTENTS = {

    # -------- conjunto núcleo exigido por el documento del TT --------

    "DESTINO": [
        "ir", "llegar", "llevar", "navegar", "dirigir", "caminar", "mover",
        "destino", "ruta", "quiero ir", "llevame", "lleva", "ve a",
        "vamos a", "vamos para", "dirigeme", "navega a", "quiero llegar",
        "me puedes llevar", "quiero que me lleves", "llevame para",
        "puedes llevarme",
    ],

    "PAUSA": [
        "pausa", "pausar", "pausa la navegacion", "espera", "esperate",
        "espera un momento", "haz una pausa", "detente un momento",
    ],

    "REANUDAR": [
        "continua", "continuar", "avanza", "avanzar", "sigue", "seguir",
        "reanuda", "reanudar", "reanuda la navegacion", "sigue con la ruta",
    ],

    "DETENER": [
        "detente", "detener", "parar", "para", "alto", "parate",
        "deten", "frenar", "detener la navegacion", "ya quiero parar",
    ],

    "RUTA_ALTERNATIVA": [
        "busca otra ruta", "otra ruta", "cambia la ruta", "recalcula la ruta",
        "quiero otra ruta", "ruta alternativa", "otro camino",
        "cambiame la ruta", "recalcula", "busca otro camino",
    ],

    # -------- intención adicional recomendada por el documento --------

    "EMERGENCIA": [
        "necesito ayuda", "hay una emergencia", "auxilio", "emergencia",
        "necesito salir", "estoy en peligro", "ayuda urgente", "socorro",
    ],

    # -------- extensiones fuera del mínimo del documento (accesibilidad) --------

    "UBICACION": [
        "donde estoy", "ubicacion", "posicion", "localizacion",
        "localizarme", "en que lugar estoy",
    ],

    "ENTORNO": [
        "que hay", "enfrente", "adelante", "alrededor", "entorno",
        "ambiente", "objetos", "obstaculos", "que tengo delante",
        "que hay enfrente",
    ],

    "REPETIR": [
        "repite", "repetir", "otra vez", "nuevamente", "dilo otra vez",
        "vuelve a decir",
    ],

    "CANCELAR": [
        "cancela", "cancelar", "cancela ruta", "cancelar ruta",
        "olvidalo", "abandona la ruta",
    ],

    "AYUDA": [
        "ayuda", "ayudame", "que puedo hacer", "comandos", "opciones",
        "como funciona",
    ],

    "CONFIRMAR": [
        "si", "correcto", "afirmativo", "exacto", "confirmo", "esta bien",
    ],

    "NEGAR": [
        "no", "incorrecto", "negativo", "no quiero", "cancela eso",
    ],

    "ESTADO": [
        "estado", "como voy", "como va", "progreso", "cuanto falta",
        "falta mucho", "donde voy", "ruta actual",
    ],
}

INTENT_LABELS = list(INTENTS.keys())

# Intenciones críticas de seguridad: si el usuario dice literalmente una
# de estas palabras, se respetan sin pasar por el modelo (ni por reglas
# difusas ni por la red, que puede equivocarse mientras aprende). Esto es
# importante porque el sistema guía a una persona con discapacidad visual,
# y el documento dedica una sección aparte a las situaciones de
# emergencia: no queremos que "emergencia" o "detente" dependan de qué
# tan entrenada esté la red en un momento dado.
SAFETY_KEYWORDS = {
    # ojo: "para" queda fuera a propósito — es a la vez el comando de
    # detener y la preposición más común del español ("para estudiar",
    # "para llegar"...), así que forzarla por coincidencia exacta
    # generaba falsos "detener" en frases que no tenían nada que ver.
    # Sigue reconociéndose como DETENER a través del motor por reglas,
    # que sí considera el resto del contexto de la frase.
    "DETENER": {"detente", "detener", "parar", "alto", "parate", "deten", "frenar", "stop"},
    "EMERGENCIA": {"auxilio", "emergencia", "peligro", "socorro"},
    "CANCELAR": {"cancela", "cancelar", "olvidalo"},
}

# La intención de emergencia se marca aquí, pero -tal como aclara el
# documento- el PLN NO decide qué hacer con ella: solo la reporta como
# "accion": "notificar_emergencia" para que el módulo correspondiente de
# la arquitectura del sistema (todavía por definir) tome el control.


# ============================================================
# ACCIÓN ESTRUCTURADA  (responsabilidad #5)
# ============================================================
# Mapeo intención -> acción tal como lo ejemplifica el documento:
# "intención: destino, destino: Metro Politécnico, acción: navegar"
# "genera una acción como 'detener sistema'" / "genera 'reanudar navegación'"

ACTION_MAP = {
    "DESTINO": "navegar",
    "PAUSA": "pausar_navegacion",
    "REANUDAR": "reanudar_navegacion",
    "DETENER": "detener_navegacion",
    "RUTA_ALTERNATIVA": "recalcular_ruta",
    "EMERGENCIA": "notificar_emergencia",
    "UBICACION": "consultar_ubicacion",
    "ENTORNO": "describir_entorno",
    "REPETIR": "repetir_ultima_instruccion",
    "CANCELAR": "cancelar_ruta",
    "AYUDA": "mostrar_ayuda",
    "CONFIRMAR": "confirmar",
    "NEGAR": "denegar",
    "ESTADO": "consultar_estado_ruta",
    DESCONOCIDO: "solicitar_repeticion",
}


def build_structured_action(intent, destination):
    """
    Construye la ÚNICA salida que, según el documento, el PLN le debe
    a los demás módulos del sistema: intención + entidad (destino) +
    acción. El PLN no ejecuta nada por sí mismo (no mueve al usuario, no
    consulta el GPS, no calcula la ruta, no controla la cámara): solo
    entrega esta instrucción estructurada para que el módulo de
    navegación, el de emergencias, o el de salida por audio actúen.
    """
    return {
        "intencion": intent,
        "destino": destination,
        "accion": ACTION_MAP.get(intent, "solicitar_repeticion"),
    }


# ============================================================
# SINÓNIMOS
# ============================================================

SYNONYMS = {
    "ir": ["marchar", "dirigirse", "desplazarse", "caminar"],
    "llegar": ["alcanzar", "arribar"],
    "detente": ["alto", "para", "frena", "deten"],
    "continua": ["sigue", "avanza", "reanuda"],
    "pausa": ["espera", "esperate"],
    "enfrente": ["adelante", "delante", "frente"],
    "donde": ["ubicacion", "posicion"],
    "repite": ["nuevamente", "otra vez"],
    "auxilio": ["emergencia", "socorro"],
}


# ============================================================
# LUGARES / DESTINOS  (responsabilidad #4: extracción de entidades)
# ============================================================
# Incluye los lugares genéricos del diseño original más los ejemplos
# concretos que usa el propio documento del TT (Metro Politécnico,
# ESCOM, el "Poli"/IPN, la Alameda, el Ángel de la Independencia), para
# que esos ejemplos puntuales del documento se resuelvan correctamente.

DEFAULT_DESTINATIONS = {
    "biblioteca": ["biblioteca", "libros", "biblioteca central"],
    "laboratorio": ["laboratorio", "lab", "laboratorio de computo"],
    "cafeteria": ["cafeteria", "comedor", "comida"],
    "entrada": ["entrada", "acceso", "puerta principal"],
    "salida": ["salida", "salir", "puerta de salida"],
    "banos": ["banos", "bano", "sanitarios"],
    "escaleras": ["escaleras", "escalera"],
    "elevador": ["elevador", "ascensor"],
    "escom": ["escom"],
    "ipn_politecnico": ["poli", "el poli", "politecnico", "ipn"],
    "metro_politecnico": ["metro politecnico"],
    "alameda": ["alameda", "la alameda"],
    "angel_de_la_independencia": ["angel de la independencia", "el angel"],
}

# Frases disparadoras para extraer el destino que viene después (reemplazo
# de la expresión regular original). Se ordenan de la más larga a la más
# corta para que, si varias calzan en la misma posición del texto, gane
# la más específica ("quiero que me lleves a" antes que "a" por ejemplo).
DESTINATION_TRIGGER_PHRASES = sorted(
    [
        "ir a", "ir para", "ve a", "llevame a", "llevame para",
        "quiero ir a", "quiero ir para", "quiero llegar a",
        "me puedes llevar a", "puedes llevarme a", "quiero que me lleves a",
        "navega a", "navega hacia", "dirigeme a", "dirigete a",
        "vamos a", "vamos para",
    ],
    key=len,
    reverse=True,
)


# ============================================================
# CONJUNTO DE ENTRENAMIENTO "SEMILLA" (con aumento de datos)
# ============================================================

def build_seed_dataset():
    """
    Construye el conjunto de entrenamiento inicial de la red a partir de
    INTENTS, generando además variantes sintéticas sustituyendo palabras
    por sus sinónimos (aumento de datos). Esto le da a la red suficiente
    variedad para generalizar desde el primer arranque, sin depender
    todavía de retroalimentación real del usuario.
    """
    dataset = []
    for intent, phrases in INTENTS.items():
        for phrase in phrases:
            dataset.append((phrase, intent))
            words = phrase.split()
            for i, word in enumerate(words):
                if word in SYNONYMS:
                    for synonym in SYNONYMS[word]:
                        variant = list(words)
                        variant[i] = synonym
                        dataset.append((" ".join(variant), intent))
    return dataset


# ============================================================
# VOCABULARIO DINÁMICO + TF-IDF
# ============================================================
# El documento plantea explícitamente representar el comando con TF-IDF
# antes de clasificar la intención. Se implementa aquí desde cero:
#   - TF (frecuencia de término): qué tan seguido aparece una palabra
#     DENTRO de la frase.
#   - IDF (frecuencia inversa de documento): qué tan rara es esa palabra
#     a lo largo de TODO el conjunto de frases de entrenamiento — con la
#     fórmula suavizada estándar log((N+1)/(df+1)) + 1, para que ninguna
#     palabra tenga IDF de cero ni se divida entre cero.

class Vocabulary:
    """Vocabulario que crece dinámicamente: cada palabra (ya reducida por
    el stemmer) recibe un índice, y se le lleva la cuenta de en cuántas
    frases de entrenamiento distintas aparece (para el IDF)."""

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []
        self.document_frequency = []
        self.total_documents = 0

    def size(self):
        return len(self.index_to_word)

    def index_of(self, word, add_if_missing=True):
        if word in self.word_to_index:
            return self.word_to_index[word]
        if not add_if_missing:
            return None
        index = len(self.index_to_word)
        self.word_to_index[word] = index
        self.index_to_word.append(word)
        self.document_frequency.append(0)
        return index

    def reset_document_frequencies(self):
        self.document_frequency = [0] * self.size()
        self.total_documents = 0

    def register_document(self, word_indices):
        """Se llama UNA vez por cada frase de entrenamiento distinta."""
        self.total_documents += 1
        for index in word_indices:
            self.document_frequency[index] += 1

    def idf(self, index):
        if index is None or index >= len(self.document_frequency):
            return 1.0
        df = self.document_frequency[index]
        return math.log((self.total_documents + 1) / (df + 1)) + 1.0

    def to_dict(self):
        return {
            "words": self.index_to_word,
            "document_frequency": self.document_frequency,
            "total_documents": self.total_documents,
        }

    def load_dict(self, data):
        data = data or {}
        self.index_to_word = list(data.get("words", []))
        self.word_to_index = {word: i for i, word in enumerate(self.index_to_word)}
        self.document_frequency = list(data.get("document_frequency", []))
        while len(self.document_frequency) < len(self.index_to_word):
            self.document_frequency.append(0)
        self.total_documents = data.get("total_documents", 0)


def vectorize_tokens(tokens, vocabulary, add_if_missing):
    """
    Convierte una lista de tokens (ya con stemming) en un vector TF-IDF.
    Si add_if_missing es True (durante el ajuste/entrenamiento) las
    palabras nuevas se agregan al vocabulario; si es False (durante la
    predicción) las palabras desconocidas simplemente se ignoran.
    """
    counts = {}
    for token in tokens:
        index = vocabulary.index_of(token, add_if_missing=add_if_missing)
        if index is not None:
            counts[index] = counts.get(index, 0) + 1

    vector = [0.0] * vocabulary.size()
    total_terms = sum(counts.values())
    if total_terms == 0:
        return vector

    for index, count in counts.items():
        term_frequency = count / total_terms
        vector[index] = term_frequency * vocabulary.idf(index)

    return vector


# ============================================================
# RED NEURONAL (PERCEPTRÓN MULTICAPA) — SIN LIBRERÍAS EXTERNAS
# ============================================================

class NeuralIntentClassifier:
    """
    Red neuronal feedforward de una capa oculta:

        entrada (vector TF-IDF, tamaño dinámico)
            -> capa oculta (tanh)
            -> capa de salida (softmax sobre las intenciones)

    Entrenada con descenso de gradiente estocástico y retropropagación,
    implementados a mano con listas de Python y el módulo `math` de la
    biblioteca estándar (nada de numpy/torch/sklearn). El tamaño de la
    capa de entrada crece dinámicamente: cuando aparece una palabra que
    nunca se había visto, se le agrega una fila de pesos nueva
    inicializada al azar, sin tocar los pesos ya aprendidos.
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE,
                 l2=L2_REGULARIZATION, labels=None):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.labels = list(labels) if labels else list(INTENT_LABELS)
        self.output_size = len(self.labels)

        self.input_size = 0
        self.w1 = []  # una fila por palabra del vocabulario: [input_size][hidden_size]
        self.b1 = [0.0] * self.hidden_size
        self.w2 = [self._random_row(self.output_size) for _ in range(self.hidden_size)]
        self.b2 = [0.0] * self.output_size

        self.trained_examples = 0

    @staticmethod
    def _random_weight():
        return _rng.uniform(-0.5, 0.5)

    def _random_row(self, size):
        return [self._random_weight() for _ in range(size)]

    def ensure_input_size(self, new_size):
        """Agrega filas de pesos nuevas para palabras nunca vistas."""
        while self.input_size < new_size:
            self.w1.append(self._random_row(self.hidden_size))
            self.input_size += 1

    @staticmethod
    def _tanh(x):
        return math.tanh(x)

    @staticmethod
    def _softmax(values):
        maximum = max(values)
        exponentials = [math.exp(v - maximum) for v in values]
        total = sum(exponentials)
        if total == 0:
            return [1.0 / len(values)] * len(values)
        return [e / total for e in exponentials]

    def forward(self, vector):
        self.ensure_input_size(len(vector))

        hidden_raw = [0.0] * self.hidden_size
        for i, x in enumerate(vector):
            if x == 0.0:
                continue
            row = self.w1[i]
            for j in range(self.hidden_size):
                hidden_raw[j] += x * row[j]
        hidden = [self._tanh(hidden_raw[j] + self.b1[j]) for j in range(self.hidden_size)]

        output_raw = [0.0] * self.output_size
        for j in range(self.hidden_size):
            hj = hidden[j]
            if hj == 0.0:
                continue
            row = self.w2[j]
            for k in range(self.output_size):
                output_raw[k] += hj * row[k]
        output_raw = [output_raw[k] + self.b2[k] for k in range(self.output_size)]

        probabilities = self._softmax(output_raw)
        return hidden, probabilities

    def predict(self, vector):
        _, probabilities = self.forward(vector)
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return self.labels[best_index], probabilities[best_index], probabilities

    def train_step(self, vector, label):
        """Un paso de descenso de gradiente estocástico (retropropagación)."""
        if label not in self.labels:
            return None

        self.ensure_input_size(len(vector))
        target_index = self.labels.index(label)

        hidden, probabilities = self.forward(vector)

        d_output = list(probabilities)
        d_output[target_index] -= 1.0

        d_hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            grad = 0.0
            row = self.w2[j]
            for k in range(self.output_size):
                grad += d_output[k] * row[k]
            d_hidden[j] = grad * (1.0 - hidden[j] ** 2)  # derivada de tanh

        lr = self.learning_rate

        for j in range(self.hidden_size):
            hj = hidden[j]
            row = self.w2[j]
            for k in range(self.output_size):
                gradient = d_output[k] * hj + self.l2 * row[k]
                row[k] -= lr * gradient
        for k in range(self.output_size):
            self.b2[k] -= lr * d_output[k]

        for i, x in enumerate(vector):
            if x == 0.0:
                continue
            row = self.w1[i]
            for j in range(self.hidden_size):
                gradient = d_hidden[j] * x + self.l2 * row[j]
                row[j] -= lr * gradient
        for j in range(self.hidden_size):
            self.b1[j] -= lr * d_hidden[j]

        self.trained_examples += 1
        return -math.log(max(probabilities[target_index], 1e-9))

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "labels": self.labels,
            "input_size": self.input_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "trained_examples": self.trained_examples,
        }

    def load_dict(self, data):
        if not data:
            return

        stored_labels = data.get("labels")
        if stored_labels and stored_labels != self.labels:
            # El conjunto de intenciones cambió (por ejemplo, se agregó
            # RUTA_ALTERNATIVA o EMERGENCIA respecto a una versión previa
            # del módulo). Los pesos guardados ya no son compatibles con
            # esta arquitectura, así que se reinicia la red — el motor
            # por reglas conserva de todas formas las frases aprendidas.
            print("[NN] El conjunto de intenciones cambió respecto a la memoria "
                  "guardada; se reinicia la red neuronal.")
            return

        self.hidden_size = data.get("hidden_size", self.hidden_size)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.l2 = data.get("l2", self.l2)
        self.labels = data.get("labels", self.labels)
        self.output_size = len(self.labels)
        self.input_size = data.get("input_size", 0)
        self.w1 = data.get("w1", [])
        self.b1 = data.get("b1", [0.0] * self.hidden_size)
        self.w2 = data.get("w2") or [self._random_row(self.output_size) for _ in range(self.hidden_size)]
        self.b2 = data.get("b2", [0.0] * self.output_size)
        self.trained_examples = data.get("trained_examples", 0)


# ============================================================
# REEMPLAZO DE `json` — serializador propio (con longitud explícita)
# ============================================================
# Formato inspirado en bencode (el formato de BitTorrent): cada valor se
# escribe como  <tipo><longitud-del-contenido>:<contenido> . Como la
# longitud siempre va primero y en dígitos decimales, el lector sabe
# exactamente cuántos caracteres leer para cada valor — nunca hace falta
# escapar comillas, dos puntos, ni ningún otro carácter especial, ni
# siquiera dentro de las frases que el usuario haya escrito. Los números
# se escriben con repr(), que en Python garantiza una conversión exacta
# de ida y vuelta (float(repr(x)) == x), así que los pesos de la red
# neuronal se guardan sin perder precisión.

def _serialize(value):
    if value is None:
        return "n0:"
    if isinstance(value, bool):
        payload = "1" if value else "0"
        return f"b{len(payload)}:{payload}"
    if isinstance(value, int):
        payload = repr(value)
        return f"i{len(payload)}:{payload}"
    if isinstance(value, float):
        payload = repr(value)
        return f"f{len(payload)}:{payload}"
    if isinstance(value, str):
        return f"s{len(value)}:{value}"
    if isinstance(value, (list, tuple)):
        payload = "".join(_serialize(item) for item in value)
        return f"l{len(payload)}:{payload}"
    if isinstance(value, dict):
        payload = "".join(_serialize(key) + _serialize(val) for key, val in value.items())
        return f"d{len(payload)}:{payload}"
    raise TypeError(f"Tipo no soportado por el serializador de memoria: {type(value)}")


class _Deserializer:
    """Cursor de lectura para el formato propio: parse() consume un
    valor completo (con todo y sus hijos si es lista/diccionario) y
    avanza la posición interna."""

    def __init__(self, text):
        self.text = text
        self.pos = 0

    def parse(self):
        type_char = self.text[self.pos]
        self.pos += 1

        colon = self.text.index(":", self.pos)
        length = int(self.text[self.pos:colon])
        self.pos = colon + 1

        payload = self.text[self.pos:self.pos + length]
        self.pos += length

        if type_char == "n":
            return None
        if type_char == "b":
            return payload == "1"
        if type_char == "i":
            return int(payload)
        if type_char == "f":
            return float(payload)
        if type_char == "s":
            return payload
        if type_char == "l":
            items = []
            reader = _Deserializer(payload)
            while reader.pos < len(payload):
                items.append(reader.parse())
            return items
        if type_char == "d":
            result = {}
            reader = _Deserializer(payload)
            while reader.pos < len(payload):
                key = reader.parse()
                val = reader.parse()
                result[key] = val
            return result

        raise ValueError(f"Formato de memoria inválido: tipo desconocido {type_char!r}")


def save_data(value):
    """Equivalente propio de json.dumps()."""
    return _serialize(value)


def load_data(text):
    """Equivalente propio de json.loads()."""
    return _Deserializer(text).parse()


# ============================================================
# MEMORIA (frases aprendidas + destinos + vocabulario + red neuronal)
# ============================================================

class NLPMemory:

    def __init__(self, path):
        self.path = path
        self.learned_phrases = {}
        self.destinations = {key: list(values) for key, values in DEFAULT_DESTINATIONS.items()}
        self.vocabulary = Vocabulary()
        self.classifier = NeuralIntentClassifier(labels=INTENT_LABELS)
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                raw_text = file.read()
        except FileNotFoundError:
            return
        except OSError as error:
            print(f"[MEMORY] Error cargando memoria: {error}")
            return

        try:
            data = load_data(raw_text)

            for intent, phrases in data.get("learned_phrases", {}).items():
                self.learned_phrases[intent] = list(phrases)

            for name, values in data.get("destinations", {}).items():
                bucket = self.destinations.setdefault(name, [])
                for value in values:
                    if value not in bucket:
                        bucket.append(value)

            self.vocabulary.load_dict(data.get("vocabulary"))
            self.classifier.load_dict(data.get("neural_network"))

        except Exception as error:
            print(f"[MEMORY] Error interpretando el archivo de memoria: {error}")

    def save(self):
        data = {
            "learned_phrases": self.learned_phrases,
            "destinations": self.destinations,
            "vocabulary": self.vocabulary.to_dict(),
            "neural_network": self.classifier.to_dict(),
        }
        try:
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(save_data(data))
        except OSError as error:
            print(f"[MEMORY] Error guardando memoria: {error}")

    def learn_phrase(self, phrase, intent):
        phrase = normalize_text(phrase)
        if not phrase or intent not in INTENTS:
            return False
        bucket = self.learned_phrases.setdefault(intent, [])
        if phrase in bucket:
            return False
        total = sum(len(values) for values in self.learned_phrases.values())
        if total >= MAX_LEARNED_PHRASES:
            return False
        bucket.append(phrase)
        self.save()
        return True

    def learn_destination(self, canonical_name, phrase):
        canonical_name = normalize_text(canonical_name)
        phrase = normalize_text(phrase)
        if not canonical_name or not phrase:
            return False
        bucket = self.destinations.setdefault(canonical_name, [])
        if phrase in bucket:
            return False
        bucket.append(phrase)
        self.save()
        return True

    def all_learned_examples(self):
        """Todas las frases (semilla + aprendidas) como pares (texto, intención)."""
        dataset = build_seed_dataset()
        for intent, phrases in self.learned_phrases.items():
            for phrase in phrases:
                dataset.append((phrase, intent))
        return dataset


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

class NavigationNLP:

    def __init__(self):
        self.memory = NLPMemory(LEARNING_FILE)
        self.stemmer = SimpleStemmer()
        self.last_debug = {}

        self.context = {
            "last_intent": None,
            "last_destination": None,
            "last_command": None,
            "last_response": None,
        }

        # el IDF debe reflejar siempre el corpus actual (semilla + lo
        # aprendido), incluso si solo se recargó memoria vieja sin
        # entrenar nada nuevo en esta sesión
        self._fit_tfidf(self.memory.all_learned_examples())

        # si la red nunca se ha entrenado (primer arranque / archivo
        # nuevo / arquitectura de intenciones cambiada), se entrena una
        # vez con el conjunto semilla antes de usarse
        if self.memory.classifier.trained_examples == 0:
            self.retrain_full(epochs=BOOTSTRAP_EPOCHS, verbose=False)

    # ========================================================
    # NORMALIZACIÓN Y TOKENS  (responsabilidad #2)
    # ========================================================

    def preprocess(self, text):
        normalized = normalize_text(text)
        tokens = tokenize(normalized)
        return normalized, tokens

    def expand_tokens(self, tokens):
        expanded = list(tokens)
        for token in tokens:
            if token in SYNONYMS:
                expanded.extend(SYNONYMS[token])
        return expanded

    def meaningful_tokens(self, tokens):
        result = []
        for token in tokens:
            if token not in STOPWORDS:
                result.append(self.stemmer.stem(token))
        return result

    # ========================================================
    # AJUSTE DE TF-IDF
    # ========================================================

    def _fit_tfidf(self, dataset):
        """
        Recalcula las frecuencias de documento (IDF) a partir del corpus
        actual (semilla + frases aprendidas), tal como plantea el
        documento del TT: representar cada comando con TF-IDF antes de
        clasificar la intención. Se puede llamar cuantas veces se quiera:
        nunca reordena ni elimina palabras del vocabulario (así los pesos
        ya aprendidos de la red neuronal siguen alineados a sus índices),
        solo actualiza en cuántas frases distintas aparece cada palabra.
        """
        vocabulary = self.memory.vocabulary
        seen_phrases = set()
        tokenized_phrases = []

        for text, _ in dataset:
            key = normalize_text(text)
            if key in seen_phrases:
                continue
            seen_phrases.add(key)
            tokens = self.meaningful_tokens(self.expand_tokens(tokenize(text)))
            if not tokens:
                continue
            tokenized_phrases.append(tokens)
            for token in tokens:
                vocabulary.index_of(token, add_if_missing=True)

        vocabulary.reset_document_frequencies()
        for tokens in tokenized_phrases:
            indices = {vocabulary.index_of(t, add_if_missing=False) for t in tokens}
            indices.discard(None)
            if indices:
                vocabulary.register_document(indices)

    # ========================================================
    # MOTOR POR REGLAS / SIMILITUD (base robusta ante errores de voz)
    # ========================================================

    def phrase_similarity(self, text, phrase):
        text = normalize_text(text)
        phrase = normalize_text(phrase)

        if text == phrase:
            return 1.0
        if phrase in text:
            return 0.92

        text_tokens = tokenize(text)
        phrase_tokens = tokenize(phrase)
        if not phrase_tokens:
            return 0.0

        scores = []
        for target in phrase_tokens:
            best = 0.0
            for token in text_tokens:
                score = similarity(target, token)
                if score > best:
                    best = score
            scores.append(best)

        return sum(scores) / len(scores) if scores else 0.0

    def vector_similarity(self, tokens_a, tokens_b):
        a = _count_occurrences(tokens_a)
        b = _count_occurrences(tokens_b)
        vocabulary = set(a) | set(b)
        if not vocabulary:
            return 0.0

        dot = magnitude_a = magnitude_b = 0.0
        for word in vocabulary:
            value_a = a.get(word, 0)
            value_b = b.get(word, 0)
            dot += value_a * value_b
            magnitude_a += value_a ** 2
            magnitude_b += value_b ** 2

        denominator = math.sqrt(magnitude_a) * math.sqrt(magnitude_b)
        return dot / denominator if denominator else 0.0

    def _rule_based_scores(self, text, tokens):
        scores = {}
        expanded = self.expand_tokens(tokens)
        meaningful = self.meaningful_tokens(expanded)

        for intent, phrases in INTENTS.items():
            for phrase in phrases:
                score = self.phrase_similarity(text, phrase)
                scores[intent] = max(scores.get(intent, 0.0), score)

                phrase_tokens = self.meaningful_tokens(tokenize(phrase))
                vector_score = self.vector_similarity(meaningful, phrase_tokens)
                scores[intent] = max(scores[intent], vector_score * 0.85)

        for intent, phrases in self.memory.learned_phrases.items():
            for phrase in phrases:
                score = self.phrase_similarity(text, phrase)
                scores[intent] = max(scores[intent], score)

        return scores

    # ========================================================
    # VERIFICACIÓN DE SEGURIDAD (no depende del aprendizaje)
    # ========================================================

    def _safety_override(self, tokens):
        token_set = set(tokens)
        for intent, keywords in SAFETY_KEYWORDS.items():
            if token_set & keywords:
                return intent
        return None

    # ========================================================
    # FUSIÓN REGLAS + RED NEURONAL (ensemble)
    # ========================================================

    def _blend(self, rule_score, nn_score, coverage):
        """
        Combina la puntuación por reglas y la de la red neuronal.
        Al principio (pocos ejemplos vistos) se confía más en las reglas;
        conforme la red acumula ejemplos de entrenamiento, su peso crece.

        `coverage` es la fracción de palabras de la frase que la red
        reconoce en su vocabulario. Si la frase trae una palabra nunca
        vista (por ejemplo un error de tipeo), su vector de entrada queda
        en ceros y la red solo respondería con su sesgo (una "adivinanza"
        sin fundamento) — en ese caso se reduce su peso en la mezcla y se
        confía más en el motor por reglas, que sí tolera errores de tipeo
        mediante Levenshtein.
        """
        examples = self.memory.classifier.trained_examples
        base_weight = min(0.65, 0.15 + examples / NN_TRUST_SATURATION)
        nn_weight = base_weight * coverage
        rule_weight = 1.0 - nn_weight
        return rule_weight * rule_score + nn_weight * nn_score

    # ========================================================
    # CLASIFICAR INTENCIÓN  (responsabilidad #3)
    # ========================================================

    def classify_intent(self, text, tokens):
        safety_intent = self._safety_override(tokens)
        if safety_intent:
            return safety_intent, 0.99

        rule_scores = self._rule_based_scores(text, tokens)

        expanded = self.expand_tokens(tokens)
        meaningful = self.meaningful_tokens(expanded)
        vector = vectorize_tokens(meaningful, self.memory.vocabulary, add_if_missing=False)
        _, _, nn_probabilities = self.memory.classifier.predict(vector)
        nn_scores = dict(zip(self.memory.classifier.labels, nn_probabilities))

        if meaningful:
            known = sum(
                1 for token in meaningful
                if self.memory.vocabulary.index_of(token, add_if_missing=False) is not None
            )
            coverage = known / len(meaningful)
        else:
            coverage = 0.0

        combined = {}
        for intent in INTENTS.keys():
            combined[intent] = self._blend(
                rule_scores.get(intent, 0.0), nn_scores.get(intent, 0.0), coverage
            )

        self.last_debug = {"rule": dict(rule_scores), "nn": nn_scores, "combined": combined}

        if not combined:
            return DESCONOCIDO, 0.0

        ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ordered[0]

        if len(ordered) > 1:
            second_score = ordered[1][1]
            margin = best_score - second_score
            # si dos intenciones quedan casi empatadas, el documento es
            # explícito: no se debe adivinar, hay que marcar desconocido
            if best_score < 0.70 and margin < 0.08:
                return DESCONOCIDO, best_score

        if best_score < MIN_CONFIDENCE:
            return DESCONOCIDO, best_score

        return best_intent, best_score

    # ========================================================
    # EXTRACCIÓN DE DESTINO  (responsabilidad #4)
    # ========================================================

    def _find_destination_after_trigger(self, text):
        """
        Busca la primera (más a la izquierda) frase disparadora de
        destino dentro del texto y devuelve lo que queda después de
        ella. Reemplaza a la expresión regular original mediante
        búsqueda manual de subcadenas, con el mismo comportamiento:
        recorre el texto de izquierda a derecha y, en cada posición,
        prueba primero las frases disparadoras más largas.
        """
        for start in range(len(text)):
            for trigger in DESTINATION_TRIGGER_PHRASES:
                end = start + len(trigger)
                if text[start:end] == trigger and end < len(text) and text[end] == " ":
                    remainder = text[end + 1:].strip()
                    if remainder:
                        return remainder
        return None

    def extract_destination(self, text):
        text = normalize_text(text)

        best_destination = None
        best_score = 0.0
        best_alias_length = 0
        for canonical, aliases in self.memory.destinations.items():
            for alias in aliases:
                score = self.phrase_similarity(text, alias)
                # ante un empate (por ejemplo "poli" y "metro politecnico"
                # coinciden igual de bien dentro del texto), se prefiere el
                # alias más específico/largo — evita que un lugar genérico
                # (el IPN) le "gane" a uno más preciso (una estación puntual)
                # solo por aparecer antes en el diccionario.
                is_better = score > best_score or (
                    score == best_score and len(alias) > best_alias_length
                )
                if is_better:
                    best_score = score
                    best_destination = canonical
                    best_alias_length = len(alias)

        if best_destination and best_score >= 0.72:
            return best_destination, best_score

        destination = self._find_destination_after_trigger(text)
        if destination:
            return destination, 0.75

        # respaldo: si no hubo patrón ni alias conocido, se toma lo que
        # quede del texto tras quitar palabras de navegación y stopwords
        navigation_keywords = {"ir", "llegar", "llevar", "navegar", "dirigir",
                                "caminar", "mover", "destino", "ruta", "llevame",
                                "vamos", "dirigeme", "navega", "alla", "aca"}
        remaining = [
            token for token in tokenize(text)
            if token not in STOPWORDS and token not in navigation_keywords
        ]
        if remaining:
            return " ".join(remaining), 0.5

        return None, 0.0

    # ========================================================
    # CONTEXTO
    # ========================================================

    def resolve_context(self, intent, text):
        # Ejemplo:
        #   Usuario: "llévame a ESCOM"       -> DESTINO, entidad=escom
        #   Usuario: "¿cuánto falta?"        -> ESTADO
        #   Usuario: "cancela"               -> CANCELAR
        return text

    # ========================================================
    # RESPUESTA HABLADA DE CONVENIENCIA
    # ========================================================
    # IMPORTANTE: según el documento, esto NO es responsabilidad del PLN
    # (es del "módulo de salida"). Se incluye solo para poder probar el
    # módulo por consola mientras ese módulo no exista todavía.

    def suggested_spoken_response(self, intent, destination):
        if intent == "DESTINO":
            return f"Entendido. Voy a ayudarte a llegar a {destination}."
        if intent == "PAUSA":
            return "De acuerdo. Pausando la navegación."
        if intent == "REANUDAR":
            return "De acuerdo. Continuamos con la navegación."
        if intent == "DETENER":
            return "De acuerdo. He detenido la navegación."
        if intent == "RUTA_ALTERNATIVA":
            return "De acuerdo. Buscando una ruta alternativa."
        if intent == "EMERGENCIA":
            return "Entendido, esto es una emergencia. Notificando de inmediato."
        if intent == "UBICACION":
            return "Voy a consultar tu ubicación actual."
        if intent == "ENTORNO":
            return "Voy a revisar lo que tienes alrededor."
        if intent == "REPETIR":
            return "Voy a repetir la última indicación."
        if intent == "CANCELAR":
            return "De acuerdo. He cancelado la navegación."
        if intent == "AYUDA":
            return ("Puedes decir: llévame a la biblioteca, qué hay enfrente, "
                    "dónde estoy, pausa, continúa, detente, busca otra ruta.")
        if intent == "ESTADO":
            return "Voy a consultar el estado de tu ruta."
        if intent == "CONFIRMAR":
            return "Entendido."
        if intent == "NEGAR":
            return "De acuerdo. No realizaré esa acción."
        return "No estoy seguro de lo que necesitas. ¿Puedes repetir la instrucción?"

    # ========================================================
    # APRENDIZAJE — retroalimentación en línea
    # ========================================================

    def _sample_replay(self, k=REPLAY_SAMPLE_SIZE):
        """Toma una muestra al azar de ejemplos ya conocidos (semilla +
        aprendidos) para reforzarlos junto con el ejemplo nuevo y evitar
        que la red "olvide" lo que ya sabía (repaso / experience replay)."""
        dataset = self.memory.all_learned_examples()
        if len(dataset) <= k:
            return dataset
        return _rng.sample(dataset, k)

    def _train_examples(self, examples, epochs=1, shuffle=True):
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            if shuffle:
                _rng.shuffle(examples)
            for text, label in examples:
                tokens = self.meaningful_tokens(self.expand_tokens(tokenize(text)))
                vector = vectorize_tokens(tokens, self.memory.vocabulary, add_if_missing=True)
                loss = self.memory.classifier.train_step(vector, label)
                if loss is not None:
                    total_loss += loss
                    steps += 1
        return total_loss / steps if steps else 0.0

    def retrain_full(self, epochs=40, verbose=True):
        """Reentrena la red con TODO el conjunto conocido (semilla +
        aprendido). Útil como consolidación periódica, más robusta que
        un solo paso en línea."""
        dataset = self.memory.all_learned_examples()
        if not dataset:
            return
        self._fit_tfidf(dataset)
        average_loss = self._train_examples(dataset, epochs=epochs)
        self.memory.save()
        if verbose:
            print(f"[NN] Reentrenamiento completo: {len(dataset)} frases, "
                  f"{epochs} épocas, pérdida promedio final ≈ {average_loss:.4f}")

    def learn_from_feedback(self, phrase, correct_intent, destination=None):
        """
        Aprende de una corrección del usuario/operador:
          1. Guarda la frase en la memoria de reglas (para el motor por similitud).
          2. Re-ajusta el IDF con el corpus actualizado.
          3. Entrena la red neuronal en línea con esa frase repetida unas
             cuantas veces + un lote de repaso de ejemplos ya conocidos,
             para que aprenda el caso nuevo sin olvidar los anteriores.
        """
        phrase = normalize_text(phrase)
        correct_intent = (correct_intent or "").strip().upper()

        if correct_intent not in INTENTS or not phrase:
            return False

        self.memory.learn_phrase(phrase, correct_intent)

        if correct_intent == "DESTINO" and destination:
            self.memory.learn_destination(destination, destination)

        self._fit_tfidf(self.memory.all_learned_examples())

        replay_batch = self._sample_replay()
        training_batch = [(phrase, correct_intent)] * 4 + replay_batch

        self._train_examples(training_batch, epochs=FEEDBACK_EPOCHS)
        self.memory.save()

        return True

    # ========================================================
    # PROCESAMIENTO COMPLETO
    # ========================================================
    # Devuelve, ante todo, la salida estructurada que exige el documento:
    # intención / destino (entidad) / acción. El resto de los campos son
    # metadatos útiles (confianza, texto normalizado, aclaración
    # requerida) y una respuesta hablada de conveniencia para pruebas.

    def process(self, text):
        normalized, tokens = self.preprocess(text)

        if not normalized:
            result = build_structured_action(DESCONOCIDO, None)
            result.update({
                "confianza": 0.0,
                "texto": "",
                "requiere_aclaracion": False,
                "respuesta_sugerida": "No escuché ningún comando.",
            })
            return result

        intent, confidence = self.classify_intent(normalized, tokens)

        destination = None
        if intent == "DESTINO":
            destination, _ = self.extract_destination(normalized)
            if destination is None:
                result = build_structured_action("DESTINO", None)
                result.update({
                    "confianza": confidence,
                    "texto": normalized,
                    "requiere_aclaracion": True,
                    "respuesta_sugerida": "¿A qué lugar quieres ir?",
                })
                return result

        self.context["last_intent"] = intent
        self.context["last_destination"] = destination
        self.context["last_command"] = normalized

        response = self.suggested_spoken_response(intent, destination)
        self.context["last_response"] = response

        result = build_structured_action(intent, destination)
        result.update({
            "confianza": round(confidence, 3),
            "texto": normalized,
            "requiere_aclaracion": False,
            "respuesta_sugerida": response,
        })

        if self.last_debug:
            rule_best = max(self.last_debug["rule"].items(), key=lambda kv: kv[1], default=(None, 0.0))
            nn_best = max(self.last_debug["nn"].items(), key=lambda kv: kv[1], default=(None, 0.0))
            result["intencion_reglas"] = rule_best[0]
            result["confianza_reglas"] = round(rule_best[1], 3)
            result["intencion_red"] = nn_best[0]
            result["confianza_red"] = round(nn_best[1], 3)

        return result


# ============================================================
# INTERFAZ DE PRUEBA
# ============================================================

def main():
    nlp = NavigationNLP()

    print("=" * 68)
    print(" PLN DE NAVEGACIÓN - TT (motor por reglas + red neuronal, TF-IDF)")
    print("=" * 68)
    print(f"Frases aprendidas: {sum(len(v) for v in nlp.memory.learned_phrases.values())}")
    print(f"Ejemplos vistos por la red neuronal: {nlp.memory.classifier.trained_examples}")
    print(f"Tamaño del vocabulario: {nlp.memory.vocabulary.size()}")
    print()
    print("Escribe un comando. 'salir' para terminar, 'entrenar' para "
          "forzar un reentrenamiento completo de la red.")
    print()

    while True:
        try:
            text = input("Usuario > ")

            if normalize_text(text) == "salir":
                break

            if normalize_text(text) == "entrenar":
                nlp.retrain_full(epochs=40)
                continue

            result = nlp.process(text)

            print()
            print("Intención:", result["intencion"], f"(confianza: {result['confianza']})")
            print("Destino/entidad:", result["destino"])
            print("Acción:", result["accion"])
            if "intencion_reglas" in result:
                print(f"  motor por reglas -> {result['intencion_reglas']} ({result['confianza_reglas']})")
                print(f"  red neuronal     -> {result['intencion_red']} ({result['confianza_red']})")
            print("Respuesta sugerida (módulo de salida):", result["respuesta_sugerida"])

            if result["intencion"] == DESCONOCIDO:
                print()
                answer = input("¿Quieres enseñar al sistema qué significa esta frase? (s/n): ")

                if normalize_text(answer) == "s":
                    print("\nIntenciones disponibles:")
                    for intent in INTENTS:
                        print(f"- {intent}")

                    correct_intent = input("\nIntención correcta: ").strip().upper()

                    if correct_intent in INTENTS:
                        destination = None
                        if correct_intent == "DESTINO":
                            destination = input("¿Qué destino representa? ")

                        nlp.learn_from_feedback(text, correct_intent, destination)
                        print("✓ Frase aprendida (motor por reglas + red neuronal actualizados).")

            print()

        except KeyboardInterrupt:
            print("\nSistema terminado.")
            break


if __name__ == "__main__":
    main()