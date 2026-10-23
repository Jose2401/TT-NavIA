import math



# Manejo de acentos mediante tabla 
_ACCENT_TABLE = str.maketrans(
    "áéíóúüñÁÉÍÓÚÜÑ",
    "aeiouunAEIOUUN",
)


def strip_accents_manual(text):
    return text.translate(_ACCENT_TABLE)


class SimpleRandom:

    def __init__(self, seed=None):
        if seed is None:
            # semilla a partir del reloj, sin usar el módulo random/time)
            seed = id(object()) ^ id([])
        self._state = seed & ((1 << 48) - 1)

    def _next_raw(self):
        # constantes del LCG de 48 bits 
        self._state = (25214903917 * self._state + 11) & ((1 << 48) - 1)
        return self._state

    def random(self):
        #Flotante uniforme en [0, 1)
        return self._next_raw() / (1 << 48)

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def shuffle(self, sequence):
        # algoritmo de Fisher-Yates
        for i in range(len(sequence) - 1, 0, -1):
            j = int(self.random() * (i + 1))
            if j > i:
                j = i
            sequence[i], sequence[j] = sequence[j], sequence[i]

    def sample(self, population, k):
        #Selecciona k elementos distintos de populationa 
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


#config
LEARNING_FILE = "navigation_nlp_memory.json"

MIN_CONFIDENCE = 0.42
LEARNING_CONFIDENCE = 0.75

MAX_LEARNED_PHRASES = 5000

DESCONOCIDO = "DESCONOCIDO"
# hiperparámetros de la red 
HIDDEN_SIZE = 24
LEARNING_RATE = 0.08
L2_REGULARIZATION = 1e-4

BOOTSTRAP_EPOCHS = 80          # entrenamiento inicial (una sola vez)
FEEDBACK_EPOCHS = 25           # entrenamiento al recibir retroalimentación
REPLAY_SAMPLE_SIZE = 30        
NN_TRUST_SATURATION = 400     

# NORMALIZACIOMN Y TOKENIZACIÓN
def remove_accents(text):
   
    return strip_accents_manual(text)


def normalize_text(text):
   # Normaliza texto: minúsculas, acentos, caracteres no alfanuméricos, espacios múltiples
    if not text:
        return ""
    text = str(text).lower().strip()
    text = remove_accents(text)

    characters = []
    for char in text:
        if ("a" <= char <= "z") or ("0" <= char <= "9") or char.isspace():
            characters.append(char)
        else:
            characters.append(" ")
    text = "".join(characters)

    # colapsa espacios múltiples
    return " ".join(text.split())


def tokenize(text):
    #Convierte texto en tokens 
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


def _count_occurrences(tokens):
    #Cuenta ocurrencias de cada elemento (reemplazo de collections.Counter)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


# DISTANCIA LEVENSHTEIN tolerancia a errores del reconocimiento de voz

def levenshtein(a, b):
    #Distancia de Levenshtein implementada desde cero (sin librerías).
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
    #Similaridad entre 0 y 1 basada en Levenshtein
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    distance = levenshtein(a, b)
    maximum = max(len(a), len(b))
    return 1.0 - (distance / maximum)



#lematización aproximada 

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



# STOPWORDS (palabras sin información relevante que se descartan)


STOPWORDS = {
    "a", "al", "el", "la", "los", "las", "de", "del", "en", "por", "para",
    "con", "que", "me", "te", "se", "mi", "un", "una", "unos", "unas",
    "quiero", "necesito", "puedes", "puede", "favor", "porfavor",
    "hacia", "como", "donde", "pues",
}


# INTENCIONES 
INTENTS = {

    #  conjunto núcleo 

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

    #  intención adicional 

    "EMERGENCIA": [
        "necesito ayuda", "hay una emergencia", "auxilio", "emergencia",
        "necesito salir", "estoy en peligro", "ayuda urgente", "socorro",
    ],

    #  extensiones fuera del mínimo (accesibilidad) 

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

SAFETY_KEYWORDS = {

    "DETENER": {"detente", "detener", "parar", "alto", "parate", "deten", "frenar", "stop"},
    "EMERGENCIA": {"auxilio", "emergencia", "peligro", "socorro"},
    "CANCELAR": {"cancela", "cancelar", "olvidalo"},
}


# ACCIÓN ESTRUCTURADA  (responsabilidad #5)

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
  # Construye la acción estructurada que el sistema debe ejecutar a partir de la intención y el destino extraídos del comando del usuario.
    return {
        "intencion": intent,
        "destino": destination,
        "accion": ACTION_MAP.get(intent, "solicitar_repeticion"),
    }



# SINÓNIMOS


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



# LUGARES / DESTINOS  

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



# CONJUNTO DE ENTRENAMIENTO "SEMILLA" (con aumento de datos)


def build_seed_dataset():

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



# VOCABULARIO DINÁMICO + TF-IDF

class Vocabulary:

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
# Convierte una lista de tokens en un vector TF-IDF (lista de flotantes) según el vocabulario dado.
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


#PERCEPTRÓN MULTICAPA
class NeuralIntentClassifier:

    #1 capa oculta

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
        #Agrega filas de pesos nuevas para palabras nunca vistas
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
        #Un paso de descenso de gradiente estocástico (retropropagación)
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
# Clase interna para deserializar datos de memoria 
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

    return _serialize(value)


def load_data(text):

    return _Deserializer(text).parse()


# MEMORIA (frases aprendidas + destinos + vocabulario + red neuronal)


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
        #Todas las frases (semilla + aprendidas) como pares (texto, intención)
        dataset = build_seed_dataset()
        for intent, phrases in self.learned_phrases.items():
            for phrase in phrases:
                dataset.append((phrase, intent))
        return dataset



# MOTOR PRINCIPAL


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

     
        self._fit_tfidf(self.memory.all_learned_examples())

        if self.memory.classifier.trained_examples == 0:
            self.retrain_full(epochs=BOOTSTRAP_EPOCHS, verbose=False)

    
    # NORMALIZACIÓN Y TOKENS  
    

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

    
    # MOTOR POR REGLAS / SIMILITUD 
    

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

    
    # VERIFICACIÓN DE SEGURIDAD (no depende del aprendizaje)
    

    def _safety_override(self, tokens):
        token_set = set(tokens)
        for intent, keywords in SAFETY_KEYWORDS.items():
            if token_set & keywords:
                return intent
        return None

    
    # FUSIÓN REGLAS + RED NEURONAL (ensemble)
    

    def _blend(self, rule_score, nn_score, coverage):
        # Mezcla las puntuaciones de la heurística basada en reglas y la red según la cobertura de tokens conocidos
        examples = self.memory.classifier.trained_examples
        base_weight = min(0.65, 0.15 + examples / NN_TRUST_SATURATION)
        nn_weight = base_weight * coverage
        rule_weight = 1.0 - nn_weight
        return rule_weight * rule_score + nn_weight * nn_score

    
    # CLASIFICAR INTENCIÓN 

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


    # EXTRACCIÓN DE DESTINO  (responsabilidad #4)


    def _find_destination_after_trigger(self, text):

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
                # ante un empate (por ejemplo "poli" y "metro politecnico" coinciden igual de bien dentro del texto)
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

        # respaldo: si no hubo patrón ni alias conocido, se toma lo q quede del texto tras quitar palabras de navegación y stopwords
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

    
    # CONTEXTO


    def resolve_context(self, intent, text):

        return text


    # RESPUESTA HABLADA DE CONVENIENCIA

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

    
    # APRENDIZAJE — retroalimentación en línea
    

    def _sample_replay(self, k=REPLAY_SAMPLE_SIZE):
        # Devuelve un subconjunto aleatorio de ejemplos ya aprendidos para repasar durante el aprendizaje en línea
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

  
    # PROCESAMIENTO COMPLETO


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



# INTERFAZ DE PRUEBA TEMP


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