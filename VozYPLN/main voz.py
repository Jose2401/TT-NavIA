
from voz_recog import VoiceRecognizer
from PLN_oneimport import NavigationNLP, DESCONOCIDO, INTENTS, normalize_text

MODEL_PATH = "D:\\Users\\thani\\SLAM\\vosk-model-small-es-0.42"
MIC_DEVICE = None       
ENABLE_TTS = True          
ENABLE_FEEDBACK = False    
EXIT_COMMANDS = {"apagar sistema", "terminar programa"} 


#  Salida hablada (opcional) 
_tts = None
if ENABLE_TTS:
    try:
        import pyttsx3
        _tts = pyttsx3.init()
        _tts.setProperty("rate", 165)
    except Exception as error:
        print(f"[TTS] No disponible: {error}")


def speak(text):
    print(f"Sistema: {text}")
    if _tts:
        _tts.say(text)
        _tts.runAndWait()  # bloquea: evita que el micrófono capte la voz del sistema


#  Ejecución de acciones (conecta aquí tu módulo de navegación) 
def execute_action(result):
    accion = result["accion"]
    destino = result["destino"]
    print(f"[ACCIÓN] {accion} | destino={destino}")
    # Ejemplo:
    # if accion == "navegar": navegador.ir_a(destino)
    # elif accion == "detener_navegacion": navegador.detener()
    # elif accion == "notificar_emergencia": alertas.enviar()


def handle_command(nlp, voice, text):
    result = nlp.process(text)

    # Falta el destino: se pregunta y se vuelve a escuchar
    if result.get("requiere_aclaracion"):
        speak(result["respuesta_sugerida"])
        answer = voice.listen(timeout=8)
        if not answer:
            speak("No escuché el destino.")
            return
        result = nlp.process(f"llevame a {answer}")

    speak(result["respuesta_sugerida"])

    if result["intencion"] == DESCONOCIDO:
        if ENABLE_FEEDBACK:
            teach(nlp, text)
        return

    execute_action(result)


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


def main():
    nlp = NavigationNLP()
    voice = VoiceRecognizer(MODEL_PATH, device=MIC_DEVICE)

    speak("Sistema de navegación listo.")
    try:
        while True:
            text = voice.listen()
            if normalize_text(text) in EXIT_COMMANDS:
                speak("Apagando el sistema.")
                break
            handle_command(nlp, voice, text)
    except KeyboardInterrupt:
        print("\nSistema terminado.")


if __name__ == "__main__":
    main()