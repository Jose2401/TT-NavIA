import threading
import time

from voz_recog import VoiceRecognizer
from PLN_oneimport import NavigationNLP, DESCONOCIDO, INTENTS, normalize_text
from Tts import Speaker
from alertas import AlertManager, WARNING
import dialogo

try:
    import pygame          # lo usa Tts.py; aquí solo para poder callar la voz en una alerta
except ImportError:
    pygame = None

MODEL_PATH = "D:\\Users\\thani\\SLAM\\vosk-model-small-es-0.42"
MIC_DEVICE = None
ENABLE_FEEDBACK = False
EXIT_COMMANDS = {"apagar sistema", "terminar programa"}
DEMO_GUIDANCE = True

# Para probar sin sensores: segundos tras iniciar para simular un carro (None = desactivado)
SIMULATE_OBSTACLE_AFTER_S = None

speaker = Speaker(voice="es-MX-DaliaNeural", rate="-5%")


# ---------------------------------------------------------------
# Voz con prioridad: la alerta puede cortar lo que se esté diciendo
# ---------------------------------------------------------------
speech_lock = threading.Lock()


def interrupt_speech():
    """Corta de inmediato la voz que esté sonando (edge-tts/pygame)."""
    if pygame is None:
        return
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass


def speak(text):
    """Voz normal. Si hay una alerta en curso, se descarta para no estorbarla."""
    if alerts.active.is_set():
        return
    with speech_lock:
        speaker.say(text)


def speak_urgent(text):
    """Voz de alerta: corta lo que suene y habla en cuanto queda libre."""
    for _ in range(30):                      # hasta ~3 s intentando
        interrupt_speech()
        if speech_lock.acquire(timeout=0.1):
            try:
                speaker.say(text)
            finally:
                speech_lock.release()
            return


def on_obstacle_speak(event):
    # nivel INFO: solo pitido, sin hablar
    if event.nivel < WARNING:
        return
    speak_urgent(dialogo.obstacle_warning(
        event.tipo, event.direccion, event.distancia_m, event.nivel))


alerts = AlertManager(on_interrupt=interrupt_speech, on_speak=on_obstacle_speak)


# ---------------------------------------------------------------
# Puntos de entrada para tus módulos de sensores y visión.
# Son seguros desde cualquier hilo: llámalos apenas llegue la señal.
# ---------------------------------------------------------------
def on_sensor_signal(distancia_m, direccion="frente", se_acerca=False):
    """Ej.: ultrasonido/LiDAR -> on_sensor_signal(1.8, "izquierda")"""
    alerts.report(None, distancia_m, direccion, "sensor", se_acerca)


def on_vision_detection(clase, direccion="frente", distancia_m=None, se_acerca=False):
    """Ej.: detector -> on_vision_detection("car", "derecha", se_acerca=True)
    Acepta clases en español o inglés (car, motorcycle, truck, bus, bicycle, person...)."""
    alerts.report(clase, distancia_m, direccion, "vision", se_acerca)


# ---------------------------------------------------------------
#  Ejecución de acciones (conecta aquí tu módulo de navegación)
# ---------------------------------------------------------------
def execute_action(result):
    accion = result["accion"]
    destino = result["destino"]
    print(f"[ACCIÓN] {accion} | destino={destino}")
    if accion == "navegar" and DEMO_GUIDANCE:
        # DEMO: reemplaza con los datos reales de tu módulo de navegación
        speak(dialogo.navigation_instruction(turn_degrees=45, distance_m=8))


def teach(nlp, text):
    if normalize_text(input("¿Enseñar esta frase al sistema? (s/n): ")) != "s":
        return
    print("Intenciones:", ", ".join(INTENTS))
    intent = input("Intención correcta: ").strip().upper()
    if intent not in INTENTS:
        print("Intención no válida.")
        return
    destination = input("Destino: ") if intent == "DESTINO" else None
    nlp.learn_from_feedback(text, intent, destination)
    print(" Frase aprendida.")


def handle_command(nlp, voice, text):
    result = nlp.process(text)

    # Falta el destino: se pregunta y se vuelve a escuchar
    if result.get("requiere_aclaracion"):
        speak(dialogo.ask_destination())
        answer = voice.listen(timeout=8)
        if not answer:
            speak(dialogo.no_audio_message())
            return
        result = nlp.process(f"llevame a {answer}")

    speak(dialogo.friendly_response(result))

    if result["intencion"] == DESCONOCIDO:
        if ENABLE_FEEDBACK:
            teach(nlp, text)
        return

    execute_action(result)


def main():
    nlp = NavigationNLP()
    voice = VoiceRecognizer(MODEL_PATH, device=MIC_DEVICE)

    alerts.start()
    if SIMULATE_OBSTACLE_AFTER_S is not None:
        threading.Timer(
            SIMULATE_OBSTACLE_AFTER_S,
            lambda: (on_sensor_signal(3.0, "izquierda", se_acerca=True),
                     on_vision_detection("car", "izquierda", se_acerca=True)),
        ).start()

    speak(dialogo.introduction())
    try:
        while True:
            text = voice.listen()
            if normalize_text(text) in EXIT_COMMANDS:
                speak(dialogo.goodbye())
                break
            handle_command(nlp, voice, text)
    except KeyboardInterrupt:
        print("\nSistema terminado.")
    finally:
        alerts.stop()


if __name__ == "__main__":
    main()