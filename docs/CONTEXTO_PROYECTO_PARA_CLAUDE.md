# CONTEXTO COMPLETO DEL PROYECTO — para un Claude sin memoria previa

> Escrito 2026-11-09 
>
> Reglas de este documento:
> - Todo lo de aquí fue **verificado contra el código o medido** durante el desarrollo. Lo incierto está marcado **[VERIFICAR]**.
> - Los números de desempeño vienen de simulación, de mapas construidos con videos reales y de pruebas unitarias. **Nada de esto se ha probado todavía con el hardware físico** (IMU, ultrasónicos, Pi). Eso es justo tu trabajo.
> - Si el código y este documento difieren, **manda el código** (el equipo edita archivos en paralelo).

---

## 0. TL;DR (actualizado 2026-11-10, ya con el hardware real en la Pi 5)

1. El programa final es **`main_nav_pi.py`** (navegación + alertas + voz, sin GUI). `main_pi.py` es solo mapeo + alertas. Los dos mains de laptop (`main_lap.py`, `main_nav_lap.py`) **también corren en la Pi** con su interfaz en el monitor: abren solos la cámara de las gafas y, si está el RP2040, usan la IMU, los ultrasónicos y el botón.
2. **Ejecuta todo desde la raíz del repo** con `.venv/bin/python` (rutas relativas: `models/`, `maps/`, `imu_montaje.json`, `calibracion_usuario.json`, `tts_cache/`).
3. El hardware real está descrito en **`docs/GAFAS_HARDWARE.md`** (manda sobre el CAD y sobre el TT): Raspberry Pi **5** (16 GB); la IMU BNO055 y los 2 MB1232 **no van al I²C de la Pi**, los lee un **RP2040-Zero** dentro de las gafas y llegan por **USB serie** (`/dev/ttyACM0`); la cámara va en **CAM1 desde el 2026-11-12 (antes CAM0; da igual para el código), girada 180°**; botón en **GPIO17** (pull-up interno) con LED en **GPIO27**; audio por **diadema USB**. No hay contraseña de sudo: todo se instala sin root (`instalar_pi.sh`).
4. Lo que **ya quedó hecho y probado con el hardware** (detalle en §0.1): firmware nuevo del RP2040, lector serie con conteo de pasos a 50 Hz, cámara por picamera2, botón (1 click = escuchar / dejar de escuchar, doble click = alertas sí/no), voz sin internet (Piper), micrófono por PipeWire, YOLO en NCNN, alertas de ultrasonido desde su propio hilo, mains de laptop en la Pi, herramientas de grabación y calibración.
5. Lo que **falta y necesita a una persona con los lentes puestos** (§0.3): grabar recorridos caminando (ahora se puede sin monitor, con el botón al prender la Pi) y con eso validar odometría, mapas y navegación.
6. Pruebas: `.venv/bin/python -m unittest discover -s tests -p "test_*.py"` → **278 pruebas, todas pasan en la Pi** (205 de antes + las de `tests/test_gafas.py`, `tests/test_alertas_ruta.py` y `tests/test_montaje.py`).
7. **Segunda ronda del mismo día (§0.5)**, tras la primera prueba con una persona: alertas más calladas (voz solo a ≤ 25 cm, de una ráfaga solo se dice la última), micrófono abierto mientras el sistema habla, contador de pasos ajustado con una caminata real, distancia por cadencia y **arranque por botón al prender la Pi** (script aparte del repo, en `~/gafas/arranque/`).

### 0.1 Qué cambió en la Pi (2026-11-10)

Archivos nuevos:
| Archivo | Rol |
|---|---|
| `firmware/code.py` | Firmware del RP2040 (CircuitPython). Manda líneas CSV: `I,ms,h,r,p,wx,wy,wz,lx,ly,lz,gx,gy,gz,cal` (IMU a **50 Hz**: euler, giroscopio, aceleración lineal, gravedad) y `U,ms,D|I,cm` (ultrasónicos por turnos, ~4.9 Hz cada uno, **sin bloquear la IMU**). Copia maestra también en `~/Desktop/code.py`; el anterior está en `~/rp2040-sensores/respaldo/code_pruebas_2026-11-09.py`. |
| `sensores.py` | `SensoresGafas` (hilo que lee el puerto, se reconecta solo), `ImuGafas` (misma interfaz que `Bno055` + `tomarMovimiento()`), `UltrasonicosGafas` (mismo `read()` que `Mb1232Array`). `python sensores.py` = monitor en vivo. |
| `camara.py` | `CamaraPi` (picamera2, modo 2304×1296 de sensor completo → 1280×720, girada 180°, foco fijo, hilo con el cuadro más reciente) y `abrirCamara()` (cae a `cv2.VideoCapture` en laptop). |
| `audio_pi.py` | Audio sin sudo: PortAudio local, elegir la diadema USB en PipeWire, tonitos de interfaz. |
| `exportar_modelos.py` | Exporta los dos YOLO a NCNN (`vision/*_ncnn_model/`, en .gitignore). |
| `instalar_pi.sh` | Instalación completa y repetible en la Pi sin sudo. |
| `calibrar_gafas.py` | Calibración guiada: montaje de la IMU, altura de los ojos, largo de paso → `imu_montaje.json` y `calibracion_usuario.json`. |
| `grabar_sesion.py` / `sesion_a_mapa.py` | Graban un recorrido (cuadros + todo lo del RP2040) y lo convierten en mapa con el pipeline real. Hermano de `video_a_mapa.py` pero con sensores reales. |
| `servicio/navia.service` | Servicio de usuario de systemd para arrancar directo (**no está activado**; el que sí está activado es el arranque por botón de `~/gafas/arranque/`, §0.5). |
| `tests/test_gafas.py` | Pruebas sin hardware del lector, pasos, botón, oído y rayos libres. |

Cambios en lo que ya existía:
- `odometry.py`: `MontajeImu` (base común), `StepDetector(confirmar=True)` (un pico suelto no es un paso; hacen falta dos seguidos), `MotionEstimator` usa `tomarMovimiento()` cuando la IMU lo trae (pasos y giro acumulados a 50 Hz; antes se muestreaba la aceleración una vez por cuadro de visión y a 2 Hz no se ve un paso), `abrirImu()`.
- `sensores.py` vigila el rumbo: si el heading del BNO055 brinca y el giroscopio dice que la cabeza no giró (pasa en NDOF cuando el magnetómetro se recalibra), ese brinco se ignora.
- `alertas.py`: los pitidos salen por **pygame** (por sounddevice/ALSA tardaban de 3 a 49 s); `reportarSensor` ahora va **dosificado** con la misma llave que las alertas del mapa y con espera creciente si la cosa sigue quieta (2.5 → 5 → 10 → 20 → 30 s); `BotonPullUp` real con gpiozero; `ModuloAlertas.activar()`.
- `interaccion.py`: `VozPiper` (voz neuronal sin internet, caché en `tts_cache/`, 1 hilo), `armarMotorVoz()`, `OidoVoz` reescrito (audio por `pw-record`, modo `boton` o `continuo`, se apaga solo tras 30 s de silencio), `ColaVoz.cerrar()`, alertas por voz ("desactiva las alertas" / "activa las alertas", RF-06), y la respuesta de EMERGENCIA ya no dice que está avisando a alguien (no existe ese módulo).
- `vision/detector.py` y `vision_bridge.py`: usan el modelo NCNN si existe (`NAVIA_SIN_NCNN=1` fuerza el `.pt`), 3 hilos (`NAVIA_HILOS_VISION`), sin sacar máscaras (nadie las usa).
- `occupancy_grid.py` / `map_manager.py`: un rayo con etiqueta `"libre"` **nunca pinta ocupado en la punta** (antes un rayo libre más corto que `maxRange`, como el recortado por eco de piso, marcaba un obstáculo falso al final).
- `vision_bridge.freeSpaceRays(lecturas, maxLibreM=2.0)`: versión conservadora para sensores reales (los mains de la Pi ya la usan).
- `montaje.py`: `LARGO_PASO_M`, `cargarUsuario()`/`guardarUsuario()`.
- `navigation/navigator.py`: si se pide `models/nav_ppo_policy.npz` usa numpy aunque esté SB3 instalado (default de `main_nav_pi`).
- `main_nav_pi.py`: flags nuevos `--voz {auto,piper,edge}`, `--escucha {auto,boton,continua}`, `--sin-boton`, `--sin-pip`, `--sin-gafas`, `--puerto`, `--cam-height`, `--estadisticas N`; avisa por voz si se desconectan los sensores; con SIGTERM guarda el mapa.

### 0.2 Mediciones en la Raspberry Pi 5 (2026-11-10)

| Qué | Resultado |
|---|---|
| YOLOv8s-seg, cuadro 640×360 | PyTorch 990–1390 ms → **NCNN 350–380 ms** (las mismas 8 detecciones en 6 cuadros de prueba, confianza ±0.005, cajas a ≤1 px salvo un borde a 8 px) ; ONNX Runtime 671 ms |
| YOLOv8s-oiv7 | PyTorch 740 ms → **NCNN 225 ms** (9/9 detecciones iguales) ; ONNX 455 ms |
| SegFormer-B0 (peligros) | 1.9 s por cuadro (PyTorch). Sigue apagado por defecto (`--hazards`) |
| Flujo óptico | 16 ms |
| Ciclo completo `main_nav_pi` | **2.0–2.4 cuadros/s** (visión 410–500 ms con el extra cada 8 cuadros); 1.7 mientras se sintetiza una frase nueva |
| Representación por ciclo | fusión 0.01, EKF 0.23, ray casting 0.79, cuadro de navegación 1.36, alertas 0.04 ms (total 2.4 ms) |
| Navegación | observación 2.6 ms, política numpy 0.05 ms (SB3 2.0 ms), `planear` 64 ms con SB3 |
| Eco de ultrasonido → inicio del pitido | **1 ms** en software (+ hasta 200 ms de la medición alternada del sensor) |
| Memoria / temperatura | ~1 GB de RAM, 60–62 °C con ventilador, sin throttling |
| Piper (voz es_MX-claude-high) | carga 3 s; 0.2 s de síntesis por segundo de audio con 4 hilos, 0.3 con 1 hilo |
| Vosk (small-es) | carga 0.75 s; 0.12 s de proceso por segundo de audio |
| IMU quieta, 60 s | 50.0 Hz exactos; rumbo sin deriva (0.00°); giroscopio σ 0.07–0.11 °/s; aceleración lineal máx 0.30 m/s² (umbral de paso 1.8) |
| MB1232 a un plano a 1.87 m | ±1 cm (279 de 292 lecturas idénticas); 4.87 Hz por sensor; sin interferencia entre los dos |
| Hilos | NCNN 4 + Piper 4 = 1160 ms por cuadro mientras habla; **NCNN 3 + Piper 1 = ~600 ms** (y 350 ms cuando no habla) |

### 0.3 Lo que falta (necesita a una persona con los lentes)

1. **Calibración**: el usuario ya corrió `calibrar_gafas.py` (hay `imu_montaje.json`: cabeceo con cero en −4.62°, rumbo invertido; ojos a 1.58 m). El **largo de paso** salió de sus dos caminatas reales por ultrasonido (0.439 m, §0.6) y se afina solo en cada corrida; ya no hace falta medir 10 m.
2. **Diadema USB** (Jabra EVOLVE 20): ya probada por el usuario (se oye y entiende). Medido: la voz del sistema **no se cuela** a su micrófono (§0.5). Falta que el usuario confirme, hablando, que ahora sí lo entiende mientras el sistema habla.
3. **Contador de pasos**: ajustado con UNA caminata real de 10 pasos (§0.5). Falta validarlo con más recorridos (otras velocidades, escaleras, cabeceos) — `grabar_sesion.py` guarda la señal cruda para eso.
4. **Recorridos reales con imagen**: no hay ninguno caminando todavía (el del día 10 quedó sin cuadros). Grabar con dos clicks al prender la Pi (o `grabar_sesion.py`), convertir con `sesion_a_mapa.py` y evaluar el modelo de navegación sobre esos mapas antes de decidir si se reentrena (ver §0.4).
5. Fuera de esa caminata de 5 m, las pruebas en vivo se hicieron con las gafas **quietas sobre el escritorio**. No se ha medido caminando: ni la deriva del rumbo en movimiento, ni el error de posición, ni la visión con escenas reales del prototipo.
6. `--hazards` (SegFormer) tarda 1.9 s por pasada y detiene el ciclo mientras corre. Si se quiere en la Pi, hay que pasarlo a un hilo aparte o exportarlo.
7. Pendientes del doc de hardware que siguen: resistencia del LED del botón sin verificar.

### 0.4 ¿Reentrenar el modelo de navegación?

**Todavía no.** Razones medidas:
- La política ve rayos sacados del **mapa**, no de los sensores crudos; el ruido real del ultrasonido (±1 cm) es menor que el que ya se simula (2 cm, y 5 % en los rayos).
- En la Pi no están los mapas reales `maps/real/v01…v19` (están en .gitignore): reentrenar aquí sin ellos perdería esa cobertura.
- Lo que sí puede cambiar el resultado es la **forma de los mapas hechos con los sensores reales** (haz angosto al centro, pasos contados por la IMU). Eso se mide grabando 3–5 recorridos con las gafas, pasándolos por `sesion_a_mapa.py` y corriendo `python -m navigation.train --eval models/nav_ppo.zip --real-maps maps/real/pNN`. Si el éxito baja de ~90 %, se reentrena mezclando esos mapas con los v01–v19 (copiarlos de la laptop). En la Pi una campaña de 8M pasos tarda del orden de una hora.

### 0.5 Segunda ronda (2026-11-10, tras la primera prueba con una persona)

Lo que reportó el usuario y qué se hizo:

| Reporte | Causa medida | Cambio |
|---|---|---|
| "Cuando habla el sistema no me entiende" | No era eco: el micrófono se **silenciaba a propósito** mientras la voz hablaba (`muteFn`). Con la Jabra puesta en el escritorio, el micrófono marca −65 dBFS mientras el sistema habla (igual que en silencio, −67) y Vosk no oye nada de esa fuga. | `OidoVoz` sigue oyendo mientras el sistema habla (`conectarOido`). Vosk da el tiempo de cada palabra: las que cayeron encima de la voz del sistema **y** son palabras de lo que decía se quitan (`_sinEco`). Si el usuario le habla encima, el sistema **se calla** (`ColaVoz.callar`, a ~1.3 s de que empieza a hablar en el banco de prueba). Un click del botón también lo calla. Si el micrófono sí oye a la bocina (≥ 6 palabras de eco), pasa solo a medio dúplex; `--mic-mudo-al-hablar` lo fuerza. |
| "En alertas habla muchísimo, las dice todas una por una" | Cada alerta crítica interrumpía a la anterior y todas quedaban en fila; en interiores todo está a < 1 m. | `ColaVoz.alerta()`: las alertas se pisan (de una ráfaga se oye la que ya sonaba y **la última**) y caducan a los 3 s. Un pip crítico ya no corta una alerta a medias; el pip de precaución ya no corta la voz. Lo mismo quieto que cambia de lado ya no vuelve a pitar. |
| "El umbral está alto, bájalo a 25 cm" | Crítica a 1.0 m (ultrasonido) / 0.80 m (mapa), precaución a 1.5 m. | `alertas.CRITICA_M = 0.25` (voz) y `PRECAUCION_M = 0.60` (solo pip), con `--umbral-alerta` / `--umbral-precaucion`. Personas en movimiento y peligros del piso conservan sus distancias. **El navegador sigue recalculando a 0.80 m** (`ModuloAlertas._recalcularNav` → `Interaccion.ciclo(recalcular=)`), suene o no. Ojo: a 0.6 m/s, 25 cm son 0.4 s; la voz a esa distancia ya no frena a nadie, para eso está el pip. |
| "No pude medir 10 m; déjalo con el paso promedio" | El detector contaba **5 de 10** pasos (umbral 1.8 m/s²; sus picos van de 1.3 a 2.9) y con paso fijo de 0.68 m una caminata de ~5 m habría salido de 6.8 m. | `sensores.detectorPasosGafas()`: umbral 1.1 / rearme 0.4 / hueco máx 1.4 s (cuenta los 10; parado y volteando no pasa de 0.9). `montaje.largoPaso(cadencia)`: 0.68 m a 105 pasos/min, proporcional a la cadencia (recortado a 0.45–1.30×). En esa caminata da **4.8 m**; el ultrasonido, que vio la pared acercarse, midió 4.6 m en los tramos con eco (paso real 0.48 m a 75 pasos/min; factor 0.99). |
| "Que el script de recorridos saque solo la distancia" | — | `sensores.resumenCaminata()` / `avancePorEco()`: pasos × largo por cadencia, y de comprobación el avance por ultrasonido donde hubo algo enfrente acercándose. Lo usan `grabar_sesion.py` (queda en `meta.json` y lo dice la voz), `sesion_a_mapa.py` y `calibrar_gafas.py --solo paso` (ya no pide 10 m). `python sensores.py sesiones/x/serie.txt` lo imprime. |
| "Sin monitor no puedo abrir el programa" | — | **`~/gafas/arranque/arranque_boton.py`** (fuera del repo, servicio de usuario `navia-arranque`, **activado**): al prender suena un pip y espera 1 min: 1 click = `main_nav_pi.py`, 2 clicks = `grabar_sesion.py --boton parar --voz` (un click la termina), nada = se acaba. Una vez por encendido. Ver `~/gafas/arranque/LEEME.txt` y `docs/GAFAS_HARDWARE.md` §4.4. |
| "Que no salude ni se presente" | — | `main_nav_pi.py` ya solo dice "Listo." al terminar de cargar. |

Tercera vuelta (misma tarde, tras el primer arranque real sin monitor):
- **El programa lanzado por el arranque no tenía botón** (y por eso el micrófono quedaba siempre abierto): el script abría el GPIO y, aunque lo cerraba, el programa recibía `GPIO busy` mientras el script siguiera vivo. Ahora la espera del botón corre en un proceso aparte que termina antes de lanzar. `main_nav_pi.py` imprime `[BOTON] listo` y, si con las gafas puestas no hay botón, lo dice por voz.
- **"Carga mapa conocido aunque no lo conozca"**: `main_nav_pi.py` siempre cargaba `maps/default` de corridas pasadas. Se agregó **`--fresh`** a `main_nav_pi.py` y `main_pi.py` (el mismo flag de los mains de laptop: borra el mapa guardado y empieza de cero). El arranque por botón lanza `--map-dir maps/arranque --fresh`, así no borra `maps/default`.
- **Cámara sin imagen**: `rpicam-hello` daba "Camera frontend has timed out" con la cámara detectada. Se comprobó con PipeWire apagado y nada usando la cámara: es el **cable plano** (hardware), no el software. De paso se arregló que `abrirCamara` cayera a `cv2.VideoCapture` cuando la cámara CSI fallaba: `/dev/video0` "abre" pero nunca da un cuadro y el main corría a ciegas sin avisar. Ahora `CamaraPi` reintenta 3 veces, exige el primer cuadro en 6 s y, si no llega, el main termina con el motivo; si la imagen se pierde a media corrida lo dice por voz.

Otros hallazgos de esta ronda:
- **Sin micrófono, PipeWire le conecta a `pw-record` el monitor de la bocina** y el sistema se oía a sí mismo. Ahora se graba pidiendo el micrófono por nombre (`audio_pi.microfono()`, `--target`) y sin micrófono no se graba (espera a que conecten la diadema).
- `pw-record` quedaba **huérfano** al salir con `os._exit`; ahora muere con el padre (`PR_SET_PDEATHSIG`) y `OidoVoz.cerrar()`.
- Vosk entrega la frase ~1.5 s después de que el usuario termina de hablar (su detector de fin de frase; esta versión no deja ajustarlo).
- Banco de prueba del oído sin diadema: un micrófono virtual con `pw-loopback` (sink `navia-mic-entrada` → source `navia-mic-prueba`) al que se le reproduce un wav con `pw-play --target`.

### 0.6 Tercera ronda (2026-11-12, cámara ya en CAM1 y primeros datos caminando)

Datos reales nuevos: `sesiones/20261110_163633/serie.txt` (95 s caminando con las gafas, **sin cuadros**: la cámara no daba imagen y el botón estaba tomado por el script viejo) y `sesiones/20261112_101239` (13 s quieto, con cuadros). Con eso:

| Qué se revisó | Resultado medido | Cambio |
|---|---|---|
| "La mala navegación de ese día" (arranque del 10, 16:46) | El programa corrió **sin cámara**: no hubo ni un ciclo (ni visión, ni mapa, ni navegación, ni respuesta a comandos, que se atienden dentro del ciclo). Solo sonaba el hilo del ultrasonido. Además cargó `maps/default` viejo. | Ya corregido en §0.5 (termina y avisa; `--fresh`). |
| Alertas en la caminata real (95 s) | Con los umbrales viejos habrían sido **10 alertas habladas + 7 pips**; con 0.25/0.60 m: **0 habladas, 1 pip**. Los ultrasónicos casi nunca vieron algo a < 60 cm (0 % de las lecturas; < 1 m: 3–6 %). | — |
| Algo quieto pegado a los sensores (68 s, como en la corrida sin cámara) | 5 alertas habladas de lo mismo | **Parado, una alerta se dice una vez**: `ModuloAlertas.caminandoFn` (pasos de la IMU, `SensoresGafas.caminando`); se repite cuando vuelve a caminar o cambia la distancia. Queda en 1. Histéresis de 8 cm en el umbral de la voz (`HISTERESIS_M`), y lo mismo quieto que cambia de lado o de nivel no vuelve a sonar. Las alertas del mapa ahora llevan su distancia (`navigation_feed.alertasDeObstaculo`) y también se espacian. |
| Visión, con todo quieto | **"Alto. Se acerca algo al frente" por una cama**: el rastreador de `vision/motion.py` marca movimiento con 8 px de desplazamiento y basta un ajuste de 1° del rumbo (6 px al compensar el giro). | Solo gente, animales y vehículos van "en movimiento" (`VisionBridge.moverEstaticos=False`). "Alto" solo si viene de **frente**; al lado es precaución (quien acompaña al usuario). Tolerancia del 25 % a los brincos de distancia de la cámara. Corrida en vivo de 130 cuadros quieto: 0 alertas. |
| Eco de piso de los ultrasónicos | Con la cabeza agachada 10–15° la mediana de las lecturas fue 3.45 m y la geometría predice 3.32 m; a > 15°, un tercio de las lecturas son piso (el resto sin eco) y todas caen donde `montaje.esEcoDePiso` las descarta. El piso nunca llega a < 2.5 m: no puede disparar alertas. | Ninguno: el modelo quedó validado con datos reales. |
| Largo de paso | El ultrasonido midió 4.6 m en 10 pasos y 15.6 m en 36 pasos: **0.46 y 0.43 m, sin relación con la cadencia** (68–100 pasos/min). El modelo "promedio × cadencia" daba 38.3 m en la caminata larga; a 0.59 m/s fueron ~28. | Calibrado = **fijo** (`montaje.PASO_CALIBRADO`); el promedio por cadencia queda solo para quien no esté calibrado. Guardado `largoPasoM = 0.439` del usuario (`calibrar_gafas.py --archivo a b`). **Se afina solo** al terminar cada corrida o grabación en que el ultrasonido vio acercarse algo ≥ 5 m (`SensoresGafas.aprenderPaso`, 30 % hacia lo medido). |
| "Zona nueva" repetido estando quieto | El origen es la esquina de 4 chunks y la pose tiembla. | Histéresis de 0.3 m al cambiar de chunk (`ChunkedMapManager.MARGEN_CAMBIO_M`). |
| Arranque por botón, de punta a punta | Con pulsaciones **en el pin** (`pinctrl set 17 pd/pu`): 1 clic lanza la navegación; ya dentro, 1 clic abre/cierra el micrófono (LED) y 2 clics apagan/prenden las alertas; 2 clics al arrancar graban (63 cuadros en 13 s) y 1 clic termina. | — |
| Peligros del piso (`--hazards`, SegFormer) en la escena real | 2.2 s por pasada (bloquea el ciclo). Sin hoyos falsos, pero tomó las puertas del ropero por puertas. | Sigue apagado por defecto. **Ojo: sin `--hazards` el sistema no detecta hoyos ni escalones hacia abajo**; solo obstáculos (YOLO + ultrasonido) y escaleras con el detector extra. |

Sigue sin validarse (no hay recorridos caminando con cuadros): distancias de la visión por el plano del piso, mapas reales, navegación guiada en un espacio real.

---

## 1. Qué es el proyecto

- **Trabajo Terminal TT 2026-B045, ESCOM-IPN** (México). Título aproximado: "Navegación con asistencia auditiva en tiempo real para personas con visión reducida, mediante lentes adaptados e inteligencia artificial". Etapas: TT1 (diseño/investigación, documento `docs/TT_2026_B045.pdf`) y TT2 (implementación; doc LaTeX en `docs/TT_2026_B045/`).
- Equipo: usuario principal git `joc-normal` (José Juan Espinosa Martínez, también aparece como jcespinosm); compañeros que hicieron visión original (`vision/`, antes repo TT-NavIA) y voz/PLN (`VozYPLN/`). Directora mencionada en el doc: Ávila Sánchez.
- Producto: **lentes** que guían a una persona con baja visión, **solo con audio** (no hay pantalla en el dispositivo). Interiores y exteriores, peatonal.
- Arquitectura del TT: percepción (visión YOLO + ultrasonido) → representación (SLAM EKF 2D + mapa de ocupación log-odds con ray casting Bresenham) → navegación (agente DRL PPO) + alertas (capa determinística independiente del RL) → interacción (voz Vosk + PLN + TTS).
- MAPS/GPS de TT1 **se quitó** en TT2 (RF-22 pasó a "resolución de destinos sobre landmarks"). No hay navegación global por GPS en el código.

### Preferencias del usuario (importantes)
- Todo en **español**. Comentarios en el código: **pocos, cortos, tono estudiante universitario mexicano relajado** ("ps", "pq", "ojo", algún "xd"). El equipo refactorizó el código para que se vea "humano". **No escribir comentarios estilo IA/corporativo.**
- Nombres: los módulos raíz usan **camelCase** (`fuseObservations`, `buildNavigationFrame`, `mgr.updateFromScan`, `VisionBridge(modelPath=, cameraProfile=)`, `ChunkedMapManager(mapDir=, chunkSizeM=)`). El paquete **`navigation/` sigue en snake_case** (`handle_command`, `NavCommand.tipo`, `goal_label`). Los atributos de los dataclasses de `interfaces.py` siguen en snake_case (`range_est`, `sensor_bearing`). **Respetar esta mezcla**: no renombrar en masa.
- Quiere decisiones **respaldadas con medición** y honestidad sobre lo que no funciona. No promover modelos ni parámetros sin comparar.
- No romper lo que ya funciona; correr las pruebas.
- El equipo hace merges y refactors en paralelo: **lee el archivo antes de editarlo**.

### Entorno de desarrollo
- macOS, repo en `/Users/jcespinosm/Documents/xdddd/SLAM`, rama actual **`main-navegacion`** (antes `slam-ray-map`).
- venv: **`.venv`** (Python 3.13.9) con torch, ultralytics, opencv, numpy, matplotlib, gymnasium, stable-baselines3, transformers, vosk/sounddevice (vosk en venv: sí se usó). **Usa siempre `.venv/bin/python`**: el shell puede traer anaconda sin gymnasium.
- No hay pytest: usar `unittest`.

---

## 2. Hardware final y montaje físico (lo más importante para la Pi)

> **Ojo (2026-11-10):** esta sección describe el diseño del TT y del CAD. El prototipo construido difiere en varias cosas (RP2040 intermedio, IMU de pie, direcciones 0x70/0x71, cámara girada, botón en GPIO17). **La fuente de verdad del hardware es `docs/GAFAS_HARDWARE.md`** y el resumen de §0. Lo de abajo se deja por historia y porque la geometría (dos ultrasónicos al frente, todo en el mismo frente rígido) sí es la real.

### 2.1 Lo que dice el TT (docs/TT_2026_B045.pdf, cap. 6–7)
- Raspberry Pi **4 Model B 4 GB** ("o equivalente", RNF-05). El usuario ha dicho "Raspberry Pi 5" en conversaciones → **[VERIFICAR con el equipo cuál es]**.
- Cámara: **Raspberry Pi Camera Module 3 Wide** (IMX708, FOV diagonal 120°, ~102° H × 67° V → perfil `picam3wide` en `vision_bridge.CAMERA_PROFILES`).
- **2× MB1232 I2CXL-MaxSonar-EZ3**, I2C, direcciones de diseño **0x20 y 0x21** (de fábrica traen 0x70 en 7 bits; hay que re-direccionar uno con la secuencia del fabricante 0xAA, 0xA5, nueva_dir).
- IMU: en TT1 no existía; **se compró BNO055** (I2C, 0x28 o 0x29 si ADR a 3.3 V). Precio ~33–35 USD.
- Micrófono USB; bocina 8 Ω con amplificador PAM8403 por jack 3.5 mm; botón en GPIO17 (pull-up); batería LiPo 5 V 10000 mAh; microSD 32 GB.
- I2C con pull-ups de 4.7 kΩ. Todos los I2C (MB1232 ×2 + BNO055) comparten el bus 1.

### 2.2 Lo que muestra el CAD (fuente de verdad del montaje)
Imágenes: `docs/Render.jpeg` (explosionado), `docs/VistaPrincipal.jpeg` (transparente), `docs/VistaReal.jpeg`.
- **Cámara**: centro del frente, sobre el puente de la nariz.
- **2 MB1232**: en el **frente**, uno sobre cada lente, **apuntando hacia adelante**, paralelos a la cámara, separados ~10 cm. **NO son laterales**: el código viejo los tenía a ±90° y se corrigió el 2026-11-07.
- **BNO055**: placa azul **plana arriba al centro** del armazón. Todo va en el mismo frente rígido ⇒ **pitch IMU = pitch cámara = pitch ultrasónicos**.
- Placa verde en la patilla derecha: conector (USB/alimentación), no es sensor.

### 2.3 Características verificadas del MB1232 (datasheet MaxBotix PD11848)
- Es de **haz ANGOSTO**: "Narrow beam pattern in the I2CXL-MaxSonar-EZ line", "excellent side object rejection", "detecting only large targets … filtering out smaller targets". El de haz ancho de la familia es el MB1202 (EZ0).
- Medio ángulo útil ~±14° (en el código, `US_MEDIO_CONO_DEG = 15`). **No cubre laterales.**
- Alcance según la gráfica del patrón del haz (cuadros de 30 cm), a 5 V / 3.3 V: varilla de 6 mm 1.5/1.2 m; varilla de 2.5 cm (≈ personas, "entre A y B" según el datasheet) 2.7/1.8 m; varilla de 9 cm 3.6/2.7 m; tabla de 28 cm (≈ pared) 7.5/6 m. En el código: `montaje.alcanceUS(label, volts)`.
- Reporta **7.65 m cuando no hay eco** y **20 cm para todo lo que está a 20 cm o MÁS CERCA**. El driver trata 0.20 como válido y crítico (antes era un bug: lo descartaba).
- Comando de medición 0x51; leer 2 bytes (cm) tras ~80–100 ms (el código espera 100 ms, que es lo que pide el datasheet si no se monitorea el pin 2 de status). Hasta 40 Hz en entornos cortos.
- **Como los dos apuntan al mismo lado, si disparan a la vez uno oye el eco del otro** → el driver dispara **alternados** en un hilo (~5 Hz por sensor).
- Opciones si quieren laterales: `US_APERTURA_DEG ≈ 15` (abre cada sensor; se pierde el traslape central/estéreo) o cambiar a MB1202 (mismo precio/tamaño/protocolo).

### 2.4 `montaje.py` (geometría central)
- `US_SEPARACION_M=0.10`, `US_APERTURA_DEG=0.0`, `US_MEDIO_CONO_DEG=15.0`, `US_MAX_M=7.65`, `US_MIN_M=0.20`, `US_VOLTS=3.3`, `CAM_ALTURA_M=1.55`. **Medir en el prototipo y ajustar.**
- Convención: x adelante, **y + = izquierda, bearing + = izquierda**.
- `usPosicion(lado)` → (offsetY, bearing); `usAlCentro(r, b, offsetY)` → polares desde la cámara; `bearingEstereo(rIzq, rDer)` → lado del objeto según la diferencia de rangos (zona muerta de 5 cm para el ruido); `rangoPiso(pitchDeg)` / `esEcoDePiso(r, pitchDeg)` → con la cabeza agachada los frontales rebotan en el piso: esas lecturas se convierten en "corredor libre", no en pared; `alcanceUS(label)`.

### 2.5 IMU BNO055 (`odometry.py`)
- Modo NDOF (fusión a bordo con magnetómetro ⇒ rumbo absoluto; la deriva del giroscopio no se acumula como en la simulación con sesgo).
- **Giro**: diferencia del heading fusionado entre ciclos. **Avance**: dead-reckoning peatonal por **conteo de pasos** (picos de aceleración lineal >1.8 m/s², re-arma <0.8, intervalo mínimo 0.3 s) × `--step-length` (0.68 m por defecto). **No se integra la aceleración dos veces** (deriva).
- También da el **pitch real** → los mains lo inyectan a `bridge.camPitchDeg` y `detPeligros.camPitchDeg` (geometría de plano de piso) y a la validación con ultrasonido (eco de piso).
- Montaje: placements P0–P7 del datasheet (registros AXIS_MAP 0x41/0x42), `PITCH_DESDE` ("pitch" o "roll"), signos y offset. **Calibración guiada (3 pasos con los lentes puestos)**: `python odometry.py --calibrar` → guarda `imu_montaje.json` (en el cwd) y `Bno055` lo carga solo al iniciar. Sin argumentos, `python odometry.py` imprime heading/pitch en vivo.
- Convención: heading + = giro a la izquierda; pitch + = cabeza hacia abajo.
- Fallback: si la IMU no responde, `MotionEstimator` usa el giro por flujo óptico (`VisualYawEstimator`) y avance 0.

---

## 3. Mapa del repositorio (archivo por archivo)

Raíz (módulos núcleo, camelCase):
| Archivo | Rol |
|---|---|
| `main_nav_pi.py` | **PROGRAMA FINAL de la Pi**: navegación DRL + alertas + voz/PLN, headless |
| `main_pi.py` | Pi: mapeo + alertas sin navegación. Contiene **`Mb1232Array`** (driver real de ultrasonido) y `VoiceOutput` (salida por consola, legado) |
| `main_nav_lap.py` | Laptop con GUI: navegación completa; `--sim` (cuarto simulado) o cámara/video; `--voz --tts` opcionales |
| `main_lap.py` | Laptop con GUI: solo mapeo (cámara/video o `--sim`) |
| `video_a_mapa.py` | Convierte videos POV en mapas guardados para entrenar/evaluar navegación |
| `interfaces.py` | Contratos de datos (dataclasses) |
| `montaje.py` | Geometría física del armazón (ver §2.4) |
| `vision_bridge.py` | Visión → detecciones (bearing, distancia, móvil, hazard, puerta abierta/cerrada), validación cruzada con ultrasonido, fusión con el EKF |
| `odometry.py` | BNO055 + pasos + fallback a flujo óptico + calibración |
| `ekf_slam.py` | EKF de pose 2D + mapa ligero de landmarks |
| `occupancy_grid.py` | Mapa de un bloque + núcleo de ray casting + `BloqueosTemporales` |
| `map_manager.py` | **Mapa global por chunks** con persistencia en disco, eventos de chunk/cuarto, re-localización |
| `navigation_feed.py` | Arma el `NavigationFrame` (lo que ve el navegador) + `safetyAlerts` (umbrales del doc) |
| `alertas.py` | **Módulo de alertas** (guardián siempre activo): evaluador + pips del equipo + refuerzo por ultrasonido + `BotonPullUp` |
| `interaccion.py` | Une VozYPLN con el navegador: `ColaVoz`, `OidoVoz`, `Interaccion` |
| `nombres.py` | Etiqueta interna (inglés) → nombre en español para todo lo que se habla |
| `sim_world.py` | Cuarto simulado, ultrasonido simulado (montaje real), visión sintética, `KeyboardUser` |
| `DISENO_SLAM_EKF.md` | Documento de diseño técnico (largo, con historia de decisiones) |
| `REPORTE_ENTRENAMIENTO_NAV.md` | Reporte de todas las campañas de entrenamiento de navegación |
| `requirements.txt` | Dependencias (las de hardware/voz están comentadas como opcionales) |

Paquetes:
- `vision/`: `detector.py` (YOLOv8-seg), `obstacle_logic.py` (alias de clases, por ejemplo tv→monitor, couch→sofa, refrigerator→fridge), `motion.py` (tracker de movimiento por IoU), `scene_segmentation.py` (SegFormer-B0 ADE20K de HuggingFace `nvidia/segformer-b0-finetuned-ade-512-512`; requiere descargarlo la primera vez), `finetune.py` (fine-tune con un dataset YOLO propio), pesos `yolov8s-seg.pt` y `yolov8s-oiv7.pt`.
- `navigation/`: `nav_env.py` (Gymnasium), `train.py` (PPO, crea el modelo), `navigator.py` (runtime), `route_planner.py` (ruta completa para visualización y ruta alternativa), `real_maps.py` (mapas guardados → mundos de entrenamiento).
- `VozYPLN/` (del equipo de voz): `voz_recog.py` (Vosk + sounddevice), `PLN_oneimport.py` (PLN puro Python: TF-IDF + red propia + reglas), `dialogo.py` (personalidad "Guía"), `Tts.py` (`Speaker`: edge-tts con caché → fallback pyttsx3), `alertas.py` (pips estéreo con sounddevice; **mismo nombre que el `alertas.py` de la raíz**, ver §5.9), `main voz.py` (demo standalone del equipo).
- `models/`: `nav_ppo.zip` (SB3) + `nav_ppo_policy.npz` (pesos puros; **la Pi infiere con numpy, sin SB3**) + `nav_ppo_meta.json`; respaldos `nav_ppo_r2*` (modelo anterior) y `nav_ppo_simonly*`; `*_cmp*` (comparación doc vs extended); `vosk-model-small-es-0.42/` (modelo de voz español).
- `maps/`: `maps/real/v01…v19` (mapas de videos reales) y `maps/default/`. **`maps/` está en .gitignore** (una vez se borraron todos; se regeneran con `video_a_mapa.py`).
- `videosxd/`: 19 videos POV reales (gitignore).
- `tests/`: suite + bancos: `simulation.py` (banco SLAM), `metrics.py`, `frases_pln.py`, `figuras_doc*.py` y `eval_interaccion.py` (scripts de figuras/evaluaciones del documento LaTeX).
- `docs/`: PDF del TT1, CAD (jpeg), doc LaTeX TT2 (`docs/TT_2026_B045/`, capítulo "Desarrollo del sistema" con secciones por módulo, figuras en `Capturas_TT2/`), presentaciones.
- **Basura conocida**: `test_map_manager.py` y `test_navigation.py` en la raíz son **copias viejas** (la suite real está en `tests/`). Se pueden borrar.

---

## 4. El ciclo de `main_nav_pi.py` (paso a paso, verificado en el código)

Inicialización:
1. `Navigator(args.model_nav)` (carga `models/nav_ppo`: SB3 si está instalado y existe el .zip; si no, `NumpyPolicy` desde `_policy.npz`).
2. `armarInteraccion(...)` → `inter` (PLN del equipo), `voz` (`ColaVoz` con `Speaker`), `oido` (`OidoVoz` con Vosk; `None` si no hay modelo/mic → entra por teclado/stdin).
3. `ModuloAlertas(hablarFn=voz p0, interrumpirFn)`; `voz.esperarSi = modAlertas.sonando` (la voz no habla encima de un pitido).
4. `VisionBridge(cameraProfile="picam3wide", extraModelPath="auto", extraEvery=15)`; `GroundHazardDetector` **solo con `--hazards`** (en la Pi viene apagado por costo; en `main_nav_lap` está encendido por defecto).
5. `makeMotionEstimator(useImu=not --no-imu, ...)` (anuncia por voz si la IMU quedó activa o si cayó a flujo óptico; pide calibrar la brújula en 8 si `mag<2`).
6. `Mb1232Array()` (si no hay smbus2 o no responde → `available=False`, `read()` regresa `[]`).
7. `BotonPullUp(alCambiar=toggle alertas)` (gpiozero comentado: no hace nada todavía).
8. `ChunkedMapManager(mapDir=--map-dir)`; si hay `landmarks.json` → `slam.importLandmarks` + `mgr.marcarRelocPendiente()`.
9. Hilo de stdin + hilo del micrófono metiendo frases a la misma cola.
10. `cv2.VideoCapture(--camera)` ← **punto a adaptar en la Pi (libcamera/picamera2)**.

Por cuadro:
1. Comandos pendientes → `inter.manejarTexto(texto, navFrame)` → mensajes → `voz.decirMsgs`. Si el usuario dice "apagar sistema" o "terminar programa", sale el `kind="despedida"` y se termina.
2. `motion, odo = motionEst.update(frame, dt)`; si `odo["pitch_deg"]` existe se inyecta al bridge y al detector de peligros.
3. `detections = bridge.process(frame, egoYaw=motion.delta_rot)` (+ peligros/puertas de segmentación cada `--hazard-every` cuadros).
4. `usLecturas = validarConUltrasonido(detections, ultrasonico.read(), pitch)` (modifica detecciones in-place y limpia ecos de piso) → `conectarUltrasonido(modAlertas, usLecturas, pitchDeg)` (pip inmediato si algo está a <1.5 m).
5. `observaciones = fuseObservations(detections, usLecturas)` → `output = slam.step(motion, observaciones)`.
6. `mgr.updateFromScan(pose, slam.observationsForMap() + rayos de piso libre)`, `mgr.clearFootprint(x, y)`, `mgr.updatePosition(...)` (eventos de chunk/cuarto), `mgr.checarReloc(slam)`.
7. `navFrame = buildNavigationFrame(slam, mgr, detections, ultrasonic=usLecturas, ...)`.
8. `alertas, recalc = modAlertas.evaluar(navFrame)` (suena solo) → `inter.ciclo(navFrame, mgr, alertas)` (inferencia del navegador + frases) → voz.
9. Autoguardado del mapa cada `--save-every` s; al salir guarda chunks y landmarks.

Diferencias con la laptop: `main_nav_lap --sim` dibuja una ventana con el mapa global y otra **"NavIA - ruta en vivo"** (zoom de 10 m, ruta planeada en cian o roja si está bloqueada, etiquetas legibles); teclas `b` (caja de demo), `m` (persona de demo), `x` (alertas on/off), `o` (wasd manual), `q` (salir).

**Huecos conocidos de `main_nav_pi`/`main_pi`** (verificados 2026-11-09; **los tres se resolvieron el 2026-11-10**, ver §0.1: rayos libres conservadores, `--cam-height`, y la cámara, los sensores y la voz ya van en hilos aparte):
- **No pasan `freeSpaceRays(usLecturas)` al mapa** (las laptops en `--sim` sí): los rayos ultrasónicos sin eco no pintan "libre". Agregarlo es trivial: `+ freeSpaceRays(usLecturas)` en el `updateFromScan`.
- `main_nav_pi` no tiene `--cam-height`/`--cam-pitch` (usa los defaults 1.55 m / 10°; la IMU sobrescribe el pitch si está activa).
- Bucle síncrono de un solo hilo: el FPS lo dicta la inferencia de visión.

---

## 5. Módulos a fondo

### 5.1 `interfaces.py`
- `MotionEstimate(delta_trans, delta_rot, dt)`.
- `VisionDetection(bearing, label, pixel_width, range_est, moving, confidence, hazard, estado, confirmado)`. `estado`: "abierta"/"cerrada"/"" (puertas). `confirmado`: "ambos"/"contradicho"/"" (validación cruzada).
- `UltrasonicReading(range_m, sensor_bearing, max_range, valid, offset_y, lado)`.
- `Observation(range_m, bearing, label, sigma_r, sigma_phi, can_init_landmark, min_confirm_hits, is_hazard, can_correct)`.
- `SlamOutput(x, y, theta, covariance 3×3, timestamp, confidence)`.
- `NavigationFrame(slam, costmap (2,H,W): canal 0 ocupación, canal 1 peligros, costmap_resolution, costmap_size_m, detections, landmarks, d_front/left/right, risk_front/left/right 0–3, chunk, room, ultrasonic)`.

### 5.2 Visión (`vision_bridge.py` + `vision/`)
- `VisionBridge.process(frame, egoYaw)`:
  - Redimensiona a ancho 640 **conservando el aspecto**. Antes se estiraba a 640×480, lo que inflaba las bbox de 16:9 y hacía salir las distancias ~25 % cortas. Una sola focal, derivada del hfov (píxeles cuadrados).
  - Distancia monocular = focal × altura típica de la clase (`TYPICAL_HEIGHTS_M`) / altura de la bbox, refinada por **plano de piso** con la fila inferior de la bbox (acotada a ±60 %), dentro de 0.2–8 m.
  - Bearing por la columna central (+ = izquierda).
  - `moving` = tracker de IoU (con **compensación del giro propio**: `shiftTrackerForYaw`, para que los objetos fijos no parezcan moverse cuando gira el usuario) o clase dinámica (person, dog, car…).
  - **Detector extra Open Images V7** (`vision/yolov8s-oiv7.pt`, 600 clases) cada `extraEvery` cuadros, solo con las clases que COCO no tiene (`EXTRA_OIV7`): Door→door, Stairs→"escalera" (hazard), Window, Waste container→trash can, Street light, Traffic sign→sign, Ladder, Shelf, Tree, Fountain, Desk, Cabinetry/Filing cabinet/Wardrobe→cabinet, Countertop→counter, Whiteboard. Umbral 0.25. Se descarta si IoU>0.55 con una detección de COCO. En la laptop tarda ~460 ms por cuadro (CPU).
- `GroundHazardDetector.process(frame, yoloDets)` (SegFormer ADE20K, ~0.16–0.3 s por cuadro en la laptop):
  - **Hoyos/coladeras** = región no transitable **rodeada de piso transitable** en la mitad baja de la imagen. Agua y escaleras salen de clases ADE20K. Distancia por plano de piso usando la fila más cercana.
  - **Puertas** (clase 14 de ADE20K) con estado: abierta si ≥7 % de piso transitable dentro de la caja del vano, si no cerrada. Se deduplican contra las puertas de YOLO por bearing.
  - `tomarPisoLibre()`: rayos "piso_libre" (9 columnas; piso contiguo desde abajo) que **restan un impacto** a los hoyos del mapa. Se entregan una sola vez por segmentación.
  - Límites medidos: la segmentación a veces marca la **calle como transitable**, un **tapete puede salir como hoyo**, y una **persona sentada** sale con la distancia sobreestimada. Las **mamparas azules** de oficina no las detecta ningún modelo preentrenado (las cubre el ultrasonido).
- `detectionRisk`: tabla 7.1 del TT. Móvil/escalera/hoyo/agua = 3; puerta abierta = 1; pared = 1; estático = 2.
- `validarConUltrasonido(dets, lecturas, pitchDeg, alfa=0.6)` (algoritmo 2 del TT):
  - Filtra ecos de piso.
  - Detección dentro del cono con eco compatible → `confirmado="ambos"`, toma el rango del ultrasonido, confianza = α·conf + (1−α)·consistencia.
  - Detección **que el haz sí alcanza** (`_alcanzaHaz`: altura típica vs. borde inferior del cono; y `alcanceUS` por tamaño) pero cuyo eco sale más lejos o no hay eco → `"contradicho"`.
  - Puertas: un eco que pasa por el vano → abierta; un eco a la distancia de la puerta → cerrada. Esto corrige las **puertas de vidrio**.
  - Móviles y peligros no se validan.
- `fuseObservations`: polares desde el centro, tolerancia de bearing = medio cono + 5°.
  - Visión + ultrasonido compatible → σ_r 0.05.
  - Solo visión → σ_r 30 % del rango y `min_confirm_hits=4`.
  - Peligros → σ_r 20 % y `is_hazard`.
  - Eco solo → "eco_us" con **`can_correct=False` y `can_init_landmark=False`** (solo va al mapa).
  - Móviles → se descartan.
- `freeSpaceRays(lecturas)`: lecturas inválidas → corredor libre (solo mapa).

### 5.3 SLAM (`ekf_slam.py`) — decisiones validadas (no revertir sin volver a medir)
- Estado = solo la pose `[x, y, θ]` (3×3). Los landmarks van en una lista aparte (no aumentan el estado). Modelo de movimiento de arco; forma de Joseph; **piso de varianza en P** (1e-4 m², 0.5°) para que nunca se vuelva "sordo".
- Parámetros: `motionNoise=(0.02, 0.05, 2°)`, `measurementNoise=(0.05, 3°)`, `associationGate=3.0`, `confirmHits=2`, `candidateMergeRadius=0.6`, `candidateTtl=40`, `landmarkPosSigma=0.4`.
- Candidatos con confirmación; la observación que confirma un candidato **no** corrige la pose en ese ciclo (contaría doble).
- `refine()` con α = max(0.05, 1/n). **Se probó un filtro de Kalman 1D ponderado por varianza y salió PEOR**.
- Fusión de candidatos por **elipse de incertidumbre** (no por radio fijo), compatibilidad de etiquetas (las genéricas son comodín), `_dedupConfirmados`, `observationsForMap()` ancla al landmark consolidado (solo etiquetas no genéricas).
- **Ecos de pared sin respaldo visual no corrigen el EKF** (el punto de reflexión se desliza): ATE 0.53→0.27 al excluirlos (medido después de corregir un bug de geometría del simulador; la regla vieja "los ecos crean landmarks" era un artefacto de ese bug).
- `correccionesTotales` (contador) para la re-localización; `exportLandmarks`/`importLandmarks`.

### 5.4 Ray casting / mapa (`occupancy_grid.py`, `map_manager.py`)
- Log-odds: `L_OCC=1.2` (un eco claro marca ocupado; sesgo conservador validado), `L_FREE=-0.4`, `L_CLAMP=5`, `OCC_THRESHOLD=1.0`.
- **Reglas del rayo**:
  - Un rango impreciso (σ>0.15, monocular) solo despeja "libre" hasta r−2σ y **nunca pinta ocupado** (su objeto lo pinta el landmark anclado).
  - Ningún rayo despeja a través de una celda con log-odds > `L_BLOCK=2.0`.
  - **Excepción** (2026-11-09): un rayo **preciso** cuyo extremo queda a más de `MARGEN_LIBERAR_M=0.30` detrás de esa celda sí le resta (caja retirada). La ablación no mostró erosión.
- Capa de peligros separada (`hazardHits`, activa con ≥2 impactos, máx 10). Los rayos `piso_libre` restan. Una celda de peligro bloquea la navegación aunque no bloquee rayos.
- `ChunkedMapManager`:
  - Chunks de 8×8 m (`chunkSizeM`) anclados al mundo con índices enteros; resolución 0.05 m; LRU en RAM (`keepLoaded=25`).
  - Persistencia: `chunk_i_j.npz` (log-odds float16 + peligros + meta) + `meta.json` + `landmarks.json`. `openExisting(dir)` adopta la geometría guardada.
  - Eventos `chunk_change(known)`, `door_crossed`, `room_recognized`.
  - **Re-localización honesta**: al importar landmarks, `marcarRelocPendiente()` → los chunks de disco salen "NUEVO" hasta que `checarReloc(slam)` ve ≥3 correcciones del EKF. Ojo: `visits` se persiste en disco, por eso hay un `_visitadosSesion` aparte.
  - `clearFootprint(x, y, 0.45)`: por donde caminaste es libre (sin esto, un pasillo sin objetos COCO quedaba sin piso libre).
  - `localCostmap`, `castDistance`, `isBlockedCell` (incluye `BloqueosTemporales`).
- `BloqueosTemporales` (mixin en los dos mapas): `agregarBloqueo(x, y, radio, duracionS)`, `quitarBloqueos()`; los consultan `isBlocked`/`isBlockedCell`; usado por la ruta alternativa.

### 5.5 `navigation_feed.py`
- Umbrales del TT: frente crítico 0.80 / precaución 1.50 m; laterales 0.45 / 0.80. Sectores: frente ±30°, laterales hasta 90°.
- `buildNavigationFrame(slam, grid, detections, ...)` (costmap de 4 m, `directionDistances` por ray casting, `sectorRisks`).
- `safetyAlerts(frame)` → `(prioridad, texto)` con frases del algoritmo 7 **con acentos y nombres en español** ("Atención: silla a 60 centímetros al frente. Deténgase.").

### 5.6 Navegación DRL (`navigation/`)
- **Entorno** (`nav_env.py`):
  - Acciones 0 avanzar 0.30 m, 1/2 girar ±20°, 3 detenerse. Clearance 0.22; meta a 0.45 m; 300 pasos por episodio.
  - Observación **"extended" de 24**: 11 rayos −100°…+100° + distancia a peligro por sector + riesgos + (sin φ, cos φ, dist) + acción previa one-hot. El vector "doc" de 8 del TT se probó y daba 44 % contra 86 % de la extended.
  - Recompensa (del TT con escalas ajustadas): +100 llegar, −100 choque/caída, +4·Δd, −0.1 por paso, −0.15·riesgo frontal, detenerse +0.5 solo con un móvil encima (−0.3 si no).
  - Mundos procedurales: cuarto 40 %, dos cuartos con puerta 35 %, exterior con coladeras 25 %; 0–3 móviles; ruido 5 %; **dropout de 8 % de rayos bloqueados** (simula los huecos del mapa real).
- **Mapas reales** (`real_maps.py`): celdas ocupadas → discos; inicio/meta en el componente libre más grande, con **≥0.30 m de holgura contra los discos**. Antes nacían encima de obstáculos y envenenaban el entrenamiento y la evaluación de mapas densos.
- **Entrenamiento** (`python -m navigation.train`): PPO SB3, 8 entornos en paralelo, ~10k pasos/s en la laptop; `--real-maps ... --real-prob 0.5`; evaluación estratificada por mundo; exporta .zip + `_policy.npz` + meta.
- **Modelo vigente** `models/nav_ppo`: 8M pasos, seed 6, real-prob 0.5, obs extended. Ver §7 para números. Cuatro campañas posteriores **no lo superaron** (meseta ~97–98 %). Mapa difícil: **v09** (pasillo de oficina muy denso, video de un tercero, no se puede regrabar). **Experimento**: entrenar sin v09 baja su éxito a 65 % contra 82.5 % con v09 → **v09 se queda en la mezcla**.
- **Runtime** (`navigator.py`): `Navigator(model_base, repeat_s=3.5, decide_every_s=1.0)`.
  - `handle_command(NavCommand|texto, frame)` con la máquina de estados del algoritmo 11 (en_espera/activo/pausado; frases literales del documento) y `update(frame, grid, alerts)`.
  - Observación con **abanico de 3 sub-rayos ±6°**.
  - **Capa de seguridad**: si el modelo dice avanzar con el frente a <0.8 m → giro forzado (con racha máxima de 8).
  - **Llegada**: a <0.6 m del objetivo, o **por percepción**: la cámara ve la clase destino a <1 m y <60° durante **3 cuadros seguidos**, y si cae en el cono del ultrasonido y hay lecturas debe venir `confirmado="ambos"`.
  - **Ruta alternativa**: bloquea 180 s el tramo planeado de 0.6–2.5 m, replanifica, y si no llega lo dice y quita los bloqueos.
  - `DEST_ALIASES` (salida/puerta/entrada→door, silla→chair, baño→toilet, archivero→cabinet, ...). `NavStatus.goal_nombre` guarda la palabra que dijo el usuario.
  - Backends: `NumpyPolicy` (Pi) y `SB3Policy`; decisión idéntica verificada por prueba.
- **Planificador** (`route_planner.py`):
  - `planear(policy, grid, inicio, goal)` simula la política sobre el mapa (~10–40 ms) → `{points, estado: llega|bloqueada|larga, pasos}`.
  - `_cabe` exige un disco de **0.12 m** libre en cada paso. 0.20 m cerraba pasillos reales: 23/32 metas contra 29/32.
  - `_goalAlcanzable` proyecta las metas tipo puerta (que viven sobre la pared) al punto más cercano donde cabe el cuerpo.
  - `PlanificadorRuta` = hilo con replan cada 2 s o urgente ante una alerta crítica.

### 5.7 Interacción (`interaccion.py` + `VozYPLN/`)
- `ColaVoz`:
  - Usa el `Speaker` del equipo: edge-tts (voz es-MX-DaliaNeural, caché en `tts_cache/`, **necesita internet**) → si falla, pyttsx3 (en Linux necesita espeak/espeak-ng instalado [VERIFICAR]) → si no, consola.
  - Cola con prioridades 0/1/2; la prioridad 0 interrumpe con `pygame.mixer.music.stop()`.
  - Canal "instruccion": **solo se habla la más reciente**, y caduca a los 6 s (corrigió el desfase de "gira a la izquierda").
  - `cadaS` evita repetir; `esperarSi` espera a que termine el pip.
- `OidoVoz`: Vosk en un hilo; se silencia mientras el TTS habla.
- `Interaccion.manejarTexto`:
  - "apagar sistema"/"terminar programa" → despedida y salida.
  - "punto X Y" → atajo sin pasar por el PLN.
  - Aclaración de destino en dos turnos ("quiero ir" → "¿a dónde?" → "la salida").
  - Mapea las intenciones del PLN a comandos del navegador o resuelve local: describir entorno (desde el NavigationFrame, en español), repetir, emergencia (p0 + pausa; **TODO: aviso real a contactos, no existe**), ayuda, confirmar, negar.
  - `amigable=True` pone frases del `dialogo` del equipo sobre las frases literales del navegador. "Deténgase" y las alertas se quedan literales a propósito.
  - Signos: el `dialogo` usa grados>0 = derecha y el sistema φ>0 = izquierda, así que se niega al convertir.
- PLN del equipo (`PLN_oneimport.py`):
  - 14 intenciones (DESTINO, PAUSA, REANUDAR, DETENER, RUTA_ALTERNATIVA, EMERGENCIA, UBICACION, ENTORNO, REPETIR, CANCELAR, AYUDA, CONFIRMAR, NEGAR, ESTADO) + DESCONOCIDO.
  - `SAFETY_KEYWORDS`: la EMERGENCIA (lesión) gana primero.
  - `_sentido_override` (negaciones; "ayuda" sola = AYUDA).
  - Memoria en `navigation_nlp_memory.json` (cwd) con **`_firma_semilla`**: si cambian las frases semilla, la red se reentrena sola.
- `nombres.py`: `nombreEs(label, "def"|"indef"|None)`, `conArticulo(palabra)`; también se inyecta a `dialogo.OBSTACLE_NAMES`.

### 5.8 `alertas.py` (raíz) — módulo de alertas (algoritmo 7, siempre activo)
- **Evaluador**: `safetyAlerts` + móviles cercanos (persona a <1 m = p0, a <2 m = p1; reemplaza a la alerta genérica de ese lado) + peligros de piso al frente (<1.3 m = p0, si no aviso en pasos) + eco ultrasónico sin explicación visual ("pared"/"obstáculo", con lado por estéreo) + detecciones contradichas → solo p1 "Posible X (solo cámara)".
- **Gating**: si `slam.confidence < CONF_MIN_MAPA=0.35`, las distancias del mapa no disparan alertas (la pose desalineada creaba alertas críticas fantasma); solo cuentan los sensores en vivo.
- **Dosificación** por llave `(prioridad, dirección, grupo)` con grupo ∈ {obstáculo, móvil, peligro}; enfriamiento de 2.5 s para p0 y 8 s para p1.
- **Entrega**: con los pips del equipo (`VozYPLN/alertas.AlertManager`: pitidos estéreo izquierda/derecha/frente/atrás con sounddevice), precaución = solo pip y crítica = pip + frase corta del `dialogo`. Sin sounddevice, las críticas se hablan.
- Import del módulo del equipo vía `importlib.import_module("VozYPLN.alertas")`, porque `import alertas` choca con el propio archivo.
- `reportarSensor(dir, dist)` (pip inmediato desde el hilo del hardware). `conectarUltrasonido(mod, lecturas, umbralM=1.5, pitchDeg)` (estéreo + rechazo del piso).
- `evaluar()` regresa `(alertas, recalcular)`; recalcular hace que el navegador decida de inmediato y que el planificador replanifique.
- `ModuloAlertas(conPip=False)` en las pruebas (si no, la suite pita en la laptop).
- `BotonPullUp(pin=27)`: doble click alterna las alertas; **gpiozero comentado**.

### 5.9 Simulación y herramientas
- `sim_world.py`: cuarto de 6×4 m con muebles; ultrasonido simulado con el **montaje real** (2 frontales ±5 cm); visión sintética (ruido de rango del 25 %). Un bug histórico de signo en `rayDistance` (los rayos rebotaban en la extensión infinita de los muros) se corrigió; si un número viejo no cuadra, esa es la razón.
- `tests/simulation.py`: banco SLAM (`python -m tests.simulation`).
- `video_a_mapa.py`:
  - Video POV → visión + giro por flujo óptico + **WalkDetector** (si el flujo residual tras quitar el giro indica caminata, avanza a `--walk-speed` 1.1 m/s; la escala del mapa depende de esto) + `clearFootprint` → mapa guardado + reporte de calidad.
  - Videos de 60 fps: usar `--stride 6`. Celular gran angular: `--hfov 110`.

---

## 6. Decisiones validadas (lista para no repetir errores)
- No reintroducir los ecos de pared como landmarks/correcciones del EKF.
- No bajar `L_OCC` ni quitar `L_BLOCK` sin volver a medir con `tests/simulation.py` en varias semillas.
- No reemplazar `refine` por un filtro de Kalman 1D (se midió peor).
- Observación extended (24) > doc (8).
- Para mapas densos no entrenar más: mejorar el mapa (regrabar con más pasadas).
- No ajustar el PLN contra el set apartado (`APARTADO` en `tests/frases_pln.py`), para que siga midiendo generalización.
- No juzgar el modelo de navegación con una sola corrida sin semilla de `main_nav_lap --sim` (es ruidosa); usar el lazo cerrado con semillas.
- Los mapas de prueba se construyen con rayos **perpendiculares** a la pared (con la liberación de celdas, los rayos paralelos se borran entre sí).

---

## 7. Números vigentes (citar estos; los viejos están en los reportes por historia)
- **SLAM** (banco, 6 semillas, 250 pasos, montaje frontal): ATE prom 0.209 m, máx 0.298; mapa precisión ~0.60, recall ~0.30. En los datos del doc: la odometría pura da 0.23 y el EKF es peor que ella en 2 de 6 semillas; con giro sesgado diverge el rumbo >100° en 2 de 6 (en hardware, la BNO055 NDOF lo mitiga).
- **Navegación**: meta del modelo 97.2 % en 760 episodios (16 mapas); con muestreo justo y 19 mapas, **97.8 %** (v09 85 %). Lazo cerrado del doc: 20/20 llegadas, 17/20 a <1 m. Lazo cerrado con semillas en la sim: 12/12 con y sin alertas.
- **Ablación del mapa**: celdas falsas 82→62 con las reglas de ray casting.
- **PLN** (código actual, 2026-11-09):

  | Set | Resultado |
  |---|---|
  | 102 frases del doc (33 están en la semilla → no es justo) | 89.2 % |
  | 86 frases nuevas del doc (`FRASES_NUEVAS` en `tests/eval_interaccion.py`) | **51.2 %** (emergencia 1/6) |
  | Set propio de 103 frases (in-sample) | 99.0 % |
  | Set apartado propio de 39 frases | 87.2 % |

  Errores que invierten la orden en el set apartado: 0. **RNF-07 (≥92 %) NO se cumple en frases nuevas.** El PLN es el punto más débil.
- **Voz (ASR Vosk)**: voces naturales 95.0 % en limpio, 62–74 % con ruido a 5–10 dB (RNF-07 no se cumple con ruido).
- **Suite**: 237 pruebas OK en la Pi (2026-11-10).
- **En la Pi**: ver la tabla de §0.2.

---

## 8. Pendientes y limitaciones conocidas (honestas)

Hardware/Pi (los puntos 1, 2, 4, 5, 6, 7 ya están hechos; del 3, 8 y 9 falta la parte que requiere caminar, ver §0.3):
1. Captura con Camera Module 3 (picamera2/libcamera) [VERIFICAR].
2. Probar el driver MB1232 real: direcciones, re-direccionamiento, 3.3 V vs 5 V, crosstalk.
3. Probar la BNO055: dirección, `python odometry.py --calibrar`, longitud de paso (caminar 10 m y ajustar `--step-length`).
4. gpiozero en `BotonPullUp` (doble click) y, si se quiere, el botón de sesión GPIO17 del TT.
5. Audio de salida según el modelo de Pi; edge-tts necesita internet (el TT pide offline → pyttsx3/espeak como base).
6. Rendimiento de visión en la Pi (medir FPS; considerar export NCNN/ONNX, `--extra-every` alto, `--hazards` con `--hazard-every` alto).
7. Agregar `freeSpaceRays` en los mains de la Pi.
8. Medir y ajustar `montaje.py` (separación de los ultrasónicos, altura de la cámara).
9. Pruebas reales: después de llegar los sensores, el plan era recalibrar los ruidos del simulador (`nav_env`) con las mediciones reales y reentrenar (~10 min por campaña).

Software:
- PLN débil en frases nuevas (ver §7). Confirmaciones raras: "te llevo a sillon" (sin artículo/acento en `friendly_response` del `dialogo`) y "No conozco ningún 'salida'" (mensaje del navegador cuando no hay landmark).
- Emergencia: no  es para avisar a contactos, es lo de alertas de sonar si se acerca a un obstaculo fijo o movil
- Una sesión del doc dejó al usuario pegado a un sillón (pose desplazada 0.6 m + sillón bajo el haz + demasiado cerca para la cámara). El aviso de bloqueo del navegador es por tiempo real (20 s).
- `door_crossed`/cuartos: heurística simple (pasar a <0.7 m de una puerta confirmada).
- Sin odometría (modo solo cámara) la traslación es 0 y el (x,y) deriva; con la IMU debería mejorar mucho (no medido).
- La segmentación confunde calle con piso transitable y tapete con hoyo (ver §5.2).
- Raíz con copias viejas `test_map_manager.py` y `test_navigation.py`.

---

## 9. Checklist de puesta en marcha en la Pi

**Vigente (2026-11-10):**
1. `bash instalar_pi.sh` (venv, dependencias, PortAudio local, Vosk, Piper, NCNN, firmware).
2. `.venv/bin/python sensores.py` → deben verse rumbo, cabeceo y las dos distancias cambiando.
3. `.venv/bin/python calibrar_gafas.py` con los lentes puestos (una vez por persona).
4. `.venv/bin/python main_nav_pi.py --estadisticas 10` (con `--no-voz` se escriben los comandos en la terminal: "punto 2 0", "que hay enfrente", "pausa", "desactiva las alertas", "apagar sistema").
5. Para verlo en el monitor: `.venv/bin/python main_nav_lap.py --sin-hazards` o `.venv/bin/python main_lap.py`.
6. Sin monitor: al prender la Pi suena un pip y espera el botón (1 click = navegar, 2 = grabar recorrido). Es `~/gafas/arranque/arranque_boton.py`, fuera del repo, servicio de usuario `navia-arranque` (ya activado). `servicio/navia.service` es la otra opción (arrancar directo, sin preguntar); **no activar los dos**.

### Checklist original (2026-11-09, antes de tener el hardware; se deja por historia)
1. Clonar el repo; Python 3 de 64 bits; `pip install -r requirements.txt` (torch/opencv para ARM) + `smbus2 vosk sounddevice pyttsx3` (+ `gpiozero`, `picamera2` del sistema) [VERIFICAR las versiones de ARM].
2. Habilitar I2C (`raspi-config`); `i2cdetect -y 1` debe mostrar 0x28 (BNO055) y los dos MB1232 (0x20/0x21 tras re-direccionar; de fábrica 0x70).
3. Copiar `models/nav_ppo_policy.npz` + `nav_ppo_meta.json` (con eso basta para navegar sin SB3), `vision/*.pt`, `models/vosk-model-small-es-0.42/`; dejar que SegFormer se descargue la primera vez o copiarlo de la caché de HuggingFace.
4. `python odometry.py` (verificar que el heading suba al girar a la izquierda y el pitch suba al mirar al piso) → `python odometry.py --calibrar`.
5. Probar ultrasonido: `python -c "from main_pi import Mb1232Array; import time; a=Mb1232Array(); time.sleep(1); print(a.available, a.razon, a.read())"`.
6. Adaptar la captura de cámara en `main_nav_pi.py`/`main_pi.py` si `cv2.VideoCapture` falla.
7. Primero `python main_pi.py --map-dir maps/prueba` (mapeo + alertas), luego `python main_nav_pi.py --map-dir maps/prueba` (con `--no-voz` para escribir comandos por stdin: "llevame a la salida", "punto 2 0", "pausa", "que hay enfrente", "apagar sistema").
8. Medir FPS y latencia de alertas (RNF-01: ≤2 s, el doc a veces dice 300 ms) y ajustar `--extra-every`/`--hazards`.
9. Correr la suite completa en la Pi.

---

## 10. Comandos útiles
```bash
# pruebas (205)
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
# banco SLAM
.venv/bin/python -m tests.simulation --steps 250 --seed 0
# laptop: navegación en sim con GUI (escribir comandos en la terminal)
.venv/bin/python main_nav_lap.py --sim
# laptop: cámara con voz
.venv/bin/python main_nav_lap.py --voz --tts
# video -> mapa (60 fps, gran angular)
.venv/bin/python video_a_mapa.py videosxd/X.mp4 --map-dir maps/real/vNN --fresh --stride 6 --hazards --hazard-every 4 --hfov 110
# entrenar navegación con todos los mapas reales
.venv/bin/python -m navigation.train --timesteps 8000000 --envs 8 --seed N --real-prob 0.5 --real-maps $(ls -d maps/real/v*) --out models/nav_cand
# evaluar un modelo
.venv/bin/python -m navigation.train --eval models/nav_ppo.zip --real-maps $(ls -d maps/real/v*)
# fine-tune de YOLO con dataset propio (formato YOLO)
.venv/bin/python -m vision.finetune datasets/X/data.yaml
# IMU
python odometry.py --calibrar
# --- en la Pi (2026-11-10) ---
bash instalar_pi.sh                                  # instalación sin sudo
.venv/bin/python sensores.py                         # imu + ultrasónicos en vivo
.venv/bin/python camara.py foto.jpg                  # probar la cámara
.venv/bin/python calibrar_gafas.py                   # calibración guiada
.venv/bin/python main_nav_pi.py --estadisticas 10    # sistema completo
.venv/bin/python main_nav_lap.py --sin-hazards       # lo mismo con interfaz
.venv/bin/python grabar_sesion.py sesiones/pasillo1  # grabar un recorrido
.venv/bin/python sesion_a_mapa.py sesiones/pasillo1 --map-dir maps/real/p01 --fresh
```

---

## 11. Historia resumida (para entender por qué las cosas son así)
- Inicio (rama `slam-ray-map`): EKF + grid + ray casting + visión-bridge; módulo de visión original del equipo (TT-NavIA) integrado y luego limpiado al paquete `vision/`.
- Mapa por chunks con persistencia; peligros de piso con ADE20K; `NavigationFrame` como contrato hacia la navegación.
- Navegación DRL: entorno, PPO, extended vs doc, brecha de dominio mapa↔simulador (abanico + dropout + capa de seguridad), backend numpy para la Pi.
- Videos reales → mapas (WalkDetector + huella libre), reentrenamientos con mapas reales.
- IMU BNO055 elegida e integrada (pasos + heading NDOF + pitch).
- Voz/PLN del equipo integrados (VozYPLN v1 y v2 con dialogo + Tts); canal de instrucciones "solo la más reciente".
- Visión ampliada (OIV7 + puertas ADE20K + fix de aspecto); salidas navegables.
- Ruta visible en vivo + planificador + módulo de alertas (después con pips del equipo).
- Montaje real del CAD (frontales), validación cruzada visión↔ultrasonido, datasheet del MB1232.
- Últimos fixes: llegada por percepción robusta, gating de alertas por confianza, dosificación por grupo, nombres en español, liberación de celdas, piso libre, ruta alternativa real, PLN (negaciones/lesiones/firma de semilla).
- Documentación LaTeX TT2 escrita en otra sesión (memoria `tt2-documentacion.md`): estilo formal-relajado, macro `\TODO{}` para lo que depende de las pruebas reales; los scripts de figuras están en `tests/figuras_doc*.py`.
