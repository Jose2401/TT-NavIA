# Gafas con sensores: base de conocimiento de la electrónica

> Documento de referencia para asistentes de IA y personas que trabajen en este proyecto.
> Recoge dónde está conectado cada componente, cómo está configurado, para qué sirve y
> qué se aprendió al probarlo. Todo lo que aparece aquí se verificó en el hardware real
> entre el 25 de octubre y el 10 de noviembre de 2026, salvo lo marcado como **PENDIENTE**.
>
> Última actualización: 2026-11-10 (firmware v2 del RP2040 y programas del proyecto, §3.2 y §5).

---

## 1. Resumen del sistema

Prototipo de gafas impresas en 3D con sensores en el frontal. Hay dos procesadores:

- **Raspberry Pi 5**: el ordenador principal, externo a las gafas. Lleva la cámara (por
  cable CSI), el botón con LED (en sus GPIO) y se comunica con el RP2040 por USB.
- **Waveshare RP2040-Zero**: microcontrolador dentro de las gafas. Lee por I²C la IMU
  y los dos ultrasónicos, e imprime las lecturas por su puerto serie USB.

```
                     ┌──────────────────────────── GAFAS ────────────────────────────┐
                     │                                                               │
                     │   MB1232 Der (0x71) ──┐                                       │
                     │   MB1232 Izq (0x70) ──┼── bus I²C ── RP2040-Zero ──┐          │
                     │   BNO055     (0x28) ──┘   GP0/GP1                  │ USB-C    │
                     │                                                    │          │
                     │   Cámara IMX708 Wide ───────────────── cable CSI ──┼──┐       │
                     └────────────────────────────────────────────────────┼──┼───────┘
                                                                          │  │
                                                     USB-A /dev/ttyACM0   │  │ CAM1
                                                                          ▼  ▼
                     Botón (GPIO17) + LED (GPIO27) ─────────────── Raspberry Pi 5
```

Montaje físico (ver `img/render.jpg`, `img/vista_principal.jpg`, `img/vista_real.jpg`):
los dos ultrasónicos van en el frontal, uno a cada lado; la cámara en el centro del
frontal, entre los dos lentes; el RP2040-Zero detrás de la cámara; la IMU en el puente
superior, montada **de pie** (vertical). El botón **no** está en las gafas: va pegado
a la Raspberry Pi.

"Derecha" e "izquierda" siempre se refieren al punto de vista de **quien lleva puestas
las gafas**.

---

## 2. Tabla de conexiones

### 2.1 Bus I²C de las gafas (en el RP2040-Zero)

| Señal | Pin RP2040-Zero | BNO055 | MB1232 Der | MB1232 Izq | Color sugerido |
|---|---|---|---|---|---|
| SDA | **GP0** | SDA | Pin 4 | Pin 4 | Azul/verde |
| SCL | **GP1** | SCL | Pin 5 | Pin 5 | Amarillo/blanco |
| Alimentación | **3V3** | VIN | Pin 6 (+V) | Pin 6 (+V) | Rojo |
| Tierra | **GND** | GND | Pin 7 | Pin 7 | Negro |

- Velocidad del bus: **100 kHz** (ver §4.1, por qué no 400 kHz).
- El condensador de 10 µF entre +V y GND de los MaxBotix que aparecía en el diseño
  original es **opcional y no está instalado**. Hasta ahora no ha hecho falta: 0 errores
  I²C en las pruebas con los tres dispositivos a la vez.

> ⚠️ **Todo a 3V3, nunca a 5V (VBUS).** El diseño original
> (`~/Desktop/componentesycables`) alimentaba los sensores desde 5V. Se cambió porque
> el MB1232 lleva resistencias pull-up internas de ~30 kΩ hacia su +V: alimentado a 5V
> subiría SDA/SCL a 5V, y los GPIO del RP2040 **no toleran 5V**. El MB1232 funciona
> desde 3,0 V. El BNO055 de Adafruit sí aceptaría 5V (lleva regulador y adaptadores de
> nivel), pero no aporta nada y se dejó también a 3V3.

### 2.2 GPIO de la Raspberry Pi 5

| Función | GPIO (BCM) | Pin físico | Conexión | Configuración |
|---|---|---|---|---|
| Botón | **GPIO17** | **11** | Botón entre el pin y GND | Entrada con **pull-up interno** |
| LED del botón | **GPIO27** | **13** | GPIO → LED → GND | Salida |
| Tierra | GND | (cualquiera) | GND común de botón y LED | — |

- El botón no lleva resistencia externa: se usa el pull-up interno (~50 kΩ) de la Pi.
- Lógica **activa en bajo**: en reposo el pin lee 1, pulsado lee 0. `gpiozero` lo
  invierte con `Button(17, pull_up=True)`, y `is_pressed` es `True` al pulsar.
- ⚠️ **El pull-up NO está activo desde el arranque.** Lo tiene que pedir cada programa
  que use el botón (§4.2). Sin pull-up, el pin queda flotando y lee 0/1 al azar.
- ⚠️ **Sin verificar:** si el LED del botón lleva resistencia limitadora integrada. Si
  no la lleva, hay que poner 220–330 Ω en serie (un GPIO da 3,3 V y ~16 mA como máximo).

### 2.3 Cámara

| Dato | Valor |
|---|---|
| Módulo | Raspberry Pi Camera Module 3 **Wide** (sensor Sony IMX708, `imx708_wide`) |
| Conector en la Pi 5 | **CAM1** desde el 2026-11-12 (antes estaba en CAM0; se cambió al arreglar la falla de imagen del día 10) |
| Nodo en el device tree | I²C `i2c@80000` (= `i2c_csi_dsi1`), dirección 0x1a. En CAM0 era `i2c@88000` (= `i2c_csi_dsi0`) |
| ¿Hay que cambiar algo al mover el conector? | **No.** `camera_auto_detect=1` la encuentra en cualquiera de los dos y `camara.py` abre "la cámara 0" (la única). Comprobado: 17.5 cuadros/s, imagen derecha |
| Modos | 1536×864 @120 fps, 2304×1296 @56 fps, 4608×2592 @14 fps |
| Montaje | **Girada 180°** (va al revés dentro del frontal, a propósito). **El sistema no lo corrige: cada programa debe girar la imagen** (§4.2) |

### 2.4 USB

| Dispositivo | Puerto de la Pi | Aparece como |
|---|---|---|
| RP2040-Zero | USB-A (cable USB-C ↔ A **de datos**) | `/dev/ttyACM0` (consola serie) y unidad `/media/th/CIRCUITPY` |

Identificadores USB del RP2040 según su estado:

| VID:PID | Estado |
|---|---|
| `2e8a:101f` | Corriendo CircuitPython (estado normal actual) |
| `2e8a:0003` | Modo BOOTSEL (unidad `RPI-RP2`, para grabar firmware) |
| `2e8a:0005` | MicroPython (firmware de fábrica de Waveshare, ya reemplazado) |

---

## 3. Componentes, propósito y configuración

### 3.1 Raspberry Pi 5 (anfitrión)

- Raspberry Pi 5 Model B Rev 1.1, 16 GB RAM, Debian 13 (trixie), kernel `6.18.50+rpt-rpi-2712`.
- Usuario `th`, en los grupos `dialout`, `gpio`, `i2c`, `spi`, `video`. No hace falta
  sudo para la serie, los GPIO ni la cámara. La contraseña de sudo no se recuerda, así
  que `/boot/firmware/config.txt` no se puede editar desde el sistema (solo sacando la
  microSD, partición `bootfs`). Por eso la configuración va en el código (§4.2).
- Bibliotecas disponibles: `gpiozero` + `lgpio` (el backend de GPIO que funciona en la
  Pi 5; RPi.GPIO **no** funciona en la Pi 5), `picamera2`, `pyserial` 3.5, `rpicam-apps`,
  `pinctrl`, `gpiod` (gpioinfo/gpioget/gpioset).
- En la Pi 5 el chip GPIO de la cabecera de 40 pines es `gpiochip0` (`pinctrl-rp1`).
- IDE: Thonny 4.1.7, configurado con el intérprete **CircuitPython** y puerto
  `auto` (`~/.config/Thonny/configuration.ini`).

### 3.2 Waveshare RP2040-Zero (microcontrolador de las gafas)

- **Propósito:** leer los sensores I²C y enviar las lecturas a la Pi por USB serie.
- **Firmware:** CircuitPython **10.3.1** (2026-09-14), placa `waveshare_rp2040_zero`,
  UID `4250304B38353816`.
- **Programa:** `code.py` en la raíz de CIRCUITPY (copia maestra en `~/Desktop/code.py`).
  CircuitPython lo ejecuta al arrancar y lo recarga solo al guardar.
- **Bibliotecas en `CIRCUITPY/lib/`** (bundle 10.x del 2026-10-23): `adafruit_bno055.mpy`,
  `adafruit_bus_device/`, `adafruit_register/`, `neopixel.mpy`.
- **LED NeoPixel integrado (GP16)** como indicador: verde = IMU y al menos un
  ultrasónico responden; ámbar = solo uno de los dos tipos; rojo = nada.
- Respaldo del `main.py` de fábrica (demo del LED WS2812 de Waveshare):
  `~/rp2040-sensores/respaldo/main.py`.

- **Programa actual: firmware v2** (2026-11-10). La fuente está en el repositorio,
  `TT-NavIA/firmware/code.py`, con copia en `~/Desktop/code.py` y en `CIRCUITPY/code.py`.
  El programa de pruebas anterior (salida para personas, una línea cada 0,5 s) quedó en
  `~/rp2040-sensores/respaldo/code_pruebas_2026-11-09.py`.

Formato de salida del firmware v2 (líneas CSV de enteros, pensadas para que las lea
`TT-NavIA/sensores.py`):

```
H,navia,2                                             al arrancar y cada 5 s
I,ms,h,r,p,wx,wy,wz,lx,ly,lz,gx,gy,gz,cal             IMU, 50 veces por segundo
U,ms,D,188                                            ultrasónico derecho, en cm
U,ms,I,187                                            ultrasónico izquierdo, en cm
E,texto                                               error o aviso
```

- `h r p`: rumbo, alabeo y cabeceo en 1/16 de grado, ya con el remapeo de ejes. Con el
  frontal mirando al suelo, p ≈ +90°·16.
- `wx wy wz`: giroscopio en 1/16 de grado/s. `lx ly lz`: aceleración lineal (sin
  gravedad) en cm/s². `gx gy gz`: vector de gravedad en cm/s².
- `cal`: byte de calibración (`sys<<6 | giro<<4 | acel<<2 | mag`).
- Ultrasónicos: `765` = sin eco; `20` = hay algo a 20 cm o menos; `-1` = no responde.
- Los dos MB1232 se disparan por turnos y **sin bloquear**: mientras uno mide (100 ms)
  se sigue leyendo la IMU. Cada sensor da ~4,9 lecturas por segundo. Medido con los tres
  dispositivos a la vez durante 60 s: IMU a 50,0 Hz exactos, 0 errores I²C, ultrasónicos
  estables a ±1 cm.
- Toda la IMU se lee en **una sola transacción** de 34 bytes desde el registro 0x14.
- Si se le manda una `o` por el puerto serie, imprime los offsets de calibración
  actuales (línea `O,...`), para guardarlos en el propio `code.py`.
- Para ver los datos en vivo desde la Pi: `cd ~/TT-NavIA && .venv/bin/python sensores.py`.
- ⚠️ Los scripts viejos que filtraban las líneas que empiezan con `IMU`
  (`~/gafas/probar_todo.py`) ya no muestran nada del RP2040 con este firmware; la cámara
  y el botón de ese script siguen funcionando igual.

### 3.3 IMU Adafruit BNO055 (9 DOF, STEMMA QT), dirección 0x28

- **Propósito:** orientación absoluta de la cabeza (rumbo, alabeo y cabeceo) por fusión
  interna de acelerómetro, giroscopio y magnetómetro (modo NDOF).
- Identificadores verificados: CHIP_ID 0xA0, ACC 0xFB, MAG 0x32, GYR 0x0F. Autotest 0x0F.
- **Oscilador interno, NO el cristal externo.** Esta placa no tiene poblado el cristal
  de 32,768 kHz. Con `use_external_crystal = True` (o `setExtCrystalUse(true)` en
  Arduino) el chip sigue respondiendo por I²C y dice estar en NDOF con fusión, pero
  `euler` devuelve `None` siempre. Fallo silencioso: no activarlo nunca.
- **Remapeo de ejes:** la IMU va de pie en el puente. Se midió con la gravedad
  (gafas derechas: +Y del chip arriba; frontal al suelo: +Z del chip adelante):

  | Dirección en las gafas | Eje del chip |
  |---|---|
  | Derecha | −X |
  | Adelante | +Z |
  | Arriba | +Y |

  Remapeo a la convención del BNO055 (X derecha, Y adelante, Z arriba):
  `X ← −X, Y ← Z, Z ← Y` → `imu.axis_remap = (0, 2, 1, 1, 0, 0)`. El determinante es
  +1, así que el sistema sigue siendo diestro. El registro se borra al apagar: se
  escribe en cada arranque.
- **Calibración guardada** (hecha con las gafas puestas el 2026-11-09, S3 G3 A3 M3),
  restaurada en cada arranque desde `code.py`:

  | | X | Y | Z | Radio |
  |---|---|---|---|---|
  | Acelerómetro | 19 | −21 | −76 | 1000 |
  | Magnetómetro | −562 | 83 | 19 | 751 |
  | Giroscopio | 3 | −1 | −1 | — |

  Tras un reinicio completo llega a `S3 G3 A3 M3` en ~1 s. El chip sigue
  autocalibrándose con el uso. Los offsets se midieron **con el mismo remapeo
  activo**: si se cambia el remapeo, hay que recalibrar.
- **Cuándo recalibrar:** si se mueve electrónica cerca de la IMU (Pi, batería, cables
  con corriente), porque los offsets del magnetómetro corrigen el hierro cercano.

### 3.4 Ultrasónicos MaxBotix MB1232 (I2CXL-MaxSonar-EZ3) ×2

- **Propósito:** distancia a obstáculos delante de cada lado del frontal.

| Lado | Dirección I²C | Origen |
|---|---|---|
| Derecho | **0x71** | Reprogramado de 0x70 a 0x71 (2026-10-26). Grabado en su EEPROM. |
| Izquierdo | **0x70** | Dirección de fábrica |

- Se verificó tapando el sensor derecho con la mano: solo cambió la columna "Der".
- **Lectura:** escribir `0x51` (medir), esperar ~100 ms, leer 2 bytes (big-endian, cm).
- **Rango útil:** 20 a ~765 cm. **Por debajo de 20 cm satura** y siempre devuelve 20,
  así que no distingue algo a 19 cm de algo a 3 cm. **765** significa que no volvió eco.
- **Ruido:** ±1–2 cm sobre una pared fija (369 ±1 cm medidos).
- **Interferencia entre los dos:** no hay, si se disparan **por turnos** (como hace
  `code.py`). Medido: cada sensor da la misma tasa de lecturas válidas solo que alternado.
- **Cono ancho:** devuelve el eco más cercano y fuerte, no necesariamente lo que se
  apunta. La mejor superficie es una pared plana y perpendicular. Superficies oblicuas,
  cortinas o huecos (puertas, pasillos) pueden no devolver eco a 3–4 m.
- **Alineación:** el derecho llegó a quedar desalineado tras soldar (veía un hueco y
  daba casi siempre `sin eco`). Se corrigió. Si un lado da `sin eco` y el otro no,
  revisar primero la orientación física.

**Cambiar la dirección de un MB1232** (protocolo I2CXL): enviar en una sola transacción
`0xAA, 0xA5, <nueva dirección en 8 bits>` a la dirección actual. La de 8 bits es la de
7 bits desplazada (`0x71 << 1 = 0xE2`) y **debe ser par**. Reservadas (8 bits): 0x00,
0x50, 0xA4, 0xAA. **Con un solo sensor afectado en el bus** (si dos comparten 0x70,
ambos reciben el comando). Script: `~/rp2040-sensores/cambiar_direccion.py`.

### 3.5 Botón con LED

- **Propósito actual:** interfaz de usuario. En `boton.py` cada pulsación alterna el
  LED; en `probar_todo.py` cada pulsación saca una foto.
- Antirrebote de 50 ms por margen de seguridad: medido sin antirrebote, 11 pulsaciones
  dieron exactamente 22 transiciones (ningún rebote).

### 3.6 Cámara IMX708 Wide

- **Propósito:** visión del entorno desde el frontal de las gafas.
- Va montada al revés: sin corrección las imágenes salen **giradas 180°**.
- **Todo programa que la use debe pedir el giro**: `Transform(hflip=1, vflip=1)` en
  Picamera2 o `--rotation 180` en rpicam (código en §4.2). Verificado con fotos: así la
  escena sale derecha.

---

## 4. Configuración del sistema

### 4.1 Decisiones de diseño y su motivo

| Decisión | Motivo |
|---|---|
| I²C a 100 kHz | El BNO055 hace *clock stretching* y a 400 kHz se vuelve inestable. |
| Sensores a 3V3 | Pull-ups internas del MB1232 a +V; los GPIO del RP2040 no toleran 5V. |
| Oscilador interno del BNO055 | La placa no tiene cristal externo. |
| Disparo por turnos de los MB1232 | Evita que uno capte el pulso del otro. |
| CircuitPython (no Arduino/MicroPython) | Biblioteca oficial de Adafruit para el BNO055, sin compilar; cargar = copiar `code.py`. |
| `gpiozero` + `lgpio` en la Pi | RPi.GPIO no es compatible con la Pi 5. |

### 4.2 Configuración que debe incluir TODO programa que use la cámara o el botón

> ⚠️ **El sistema operativo no corrige nada.** Se decidió no tocar
> `/boot/firmware/config.txt`: la cámara y el botón solo se usan mientras corre el código
> del proyecto. Por eso **cada programa que use la cámara tiene que girar la imagen 180°
> por su cuenta, y cada programa que use el botón tiene que activar el pull-up**. Si un
> programa nuevo se olvida de hacerlo, las fotos saldrán al revés y el botón dará
> lecturas aleatorias.

**Cámara: girar 180°.** Elegir la línea según la biblioteca:

```python
# Picamera2 (recomendado). Probado: devuelve Transform 'hvflip' = rotación 180°.
from libcamera import Transform
from picamera2 import Picamera2

camara = Picamera2()
camara.configure(camara.create_still_configuration(       # o create_video_configuration /
    main={"size": (2304, 1296)},                          #    create_preview_configuration
    transform=Transform(hflip=1, vflip=1),                # <- OBLIGATORIO en este proyecto
))
```

```bash
# rpicam-apps (línea de comandos)
rpicam-still --rotation 180 -o foto.jpg
rpicam-vid   --rotation 180 -o video.h264
```

```python
# Si llegan fotogramas de otra fuente (p. ej. OpenCV con un array de Picamera2 sin
# transform), girarlos antes de procesarlos:
import cv2
frame = cv2.rotate(frame, cv2.ROTATE_180)
```

Que sea `Transform(hflip=1, vflip=1)` y no un giro de 90°: se verificó con fotos
(prueba P-10) que el montaje es de 180° exactos.

**Botón: pull-up interno en GPIO17.**

```python
from gpiozero import Button, LED

boton = Button(17, pull_up=True, bounce_time=0.05)   # <- pull_up=True OBLIGATORIO
led = LED(27)
# boton.is_pressed es True al pulsar (gpiozero invierte la lógica activa en bajo)
```

```bash
# Desde la terminal o un script de shell (no necesita sudo):
pinctrl set 17 ip pu
```

Al terminar el programa, gpiozero libera el pin y el pull-up desaparece. Es normal,
porque fuera del programa nadie lee el botón.

**Ya lo cumplen:** `~/gafas/probar_todo.py` (cámara + botón), `~/gafas/prueba_transform.py` (cámara) y `~/Desktop/boton.py` (botón).

**Referencia, por si algún día se quiere hacer a nivel de sistema** (requiere la
contraseña de sudo, que ahora mismo no se recuerda): en `/boot/firmware/config.txt`,
poner `camera_auto_detect=0` y añadir `dtoverlay=imx708,cam1,rotation=0` y
`gpio=17=ip,pu`. El `cam0`/`cam1` tiene que coincidir con el conector donde esté la
cámara (ahora CAM1; con `cam0` se usaba cuando estaba en el otro). Si se hace,
hay que **quitar** el `Transform` de los programas, o la imagen se giraría dos veces.

### 4.3 Thonny

`~/.config/Thonny/configuration.ini`: `backend = CircuitPython`, `port = auto`,
directorio de trabajo `~/Desktop`.

### 4.4 Arranque por botón al prender la Pi (sin monitor)

Vive **fuera del repo**, en `~/gafas/arranque/` (`arranque_boton.py`, `voces/*.wav`,
`navia-arranque.service`, `LEEME.txt`). Es un servicio de **usuario** de systemd
(`~/.config/systemd/user/navia-arranque.service`, **activado** el 2026-11-10): corre
cuando la Pi entra sola a la sesión de `th` (autologin), que es donde vive PipeWire.
Usa el Python del sistema (`/usr/bin/python3`: gpiozero + pygame), no el venv.

| Al prender | Qué pasa |
|---|---|
| Pip de tres notas que suben + voz "Lentes listos…" | Ya espera el botón (1 minuto). El LED late. |
| 1 clic | Pip, "Iniciando navegación", lanza `main_nav_pi.py --map-dir maps/arranque --fresh` (dice "Listo." al cargar). **Empieza con el mapa en blanco** y lo va armando y reconociendo en esa corrida; no toca `maps/default`. |
| 2 clics seguidos (< 0.7 s) | Dos pips, "Grabando recorrido", lanza `grabar_sesion.py --boton parar --voz`. LED fijo = grabando. **Un clic la termina** y la voz dice duración, pasos y metros. Queda en `~/TT-NavIA/sesiones/FECHA_HORA/`. Tope: 20 min. |
| Nada en 1 minuto | Pip de tres notas que bajan; termina sin hacer nada. |
| El programa truena | Tono grave doble y la voz dice qué revisar: "No hay imagen de la cámara…", "No detecto los sensores…", "El botón no responde" o el error genérico (lo saca de `ultimo_programa.log`). |

- Pregunta **una sola vez por encendido** (guarda el `boot_id` del kernel en
  `~/.cache/navia/arranque_boot_id`). Para probar sin reiniciar:
  `python3 ~/gafas/arranque/arranque_boton.py --forzar` (con `--simular 5` o
  `--simular 5,5.4` hace como si picaran el botón).
- Qué pasó: `~/gafas/arranque/arranque.log` (el arranque) y un registro por corrida en
  `~/gafas/arranque/registros/NNNN_programa.log` (`ultimo_programa.log` apunta al más
  nuevo). La navegación además queda grabada en `~/TT-NavIA/sesiones/nav_FECHA/`.
- ⚠️ **La Pi no tiene reloj con pila**: sin internet arranca con la hora del último
  apagado, así que las fechas de carpetas y bitácoras se repiten y no sirven para ordenar.
  Por eso los registros van numerados y las carpetas de grabación nunca se reusan.
- ⚠️ **Apagarla desconectándola pierde lo escrito en los últimos ~30 s** (pasó el
  2026-11-12: grabaciones sin sus últimos 15–20 s). Ahora los programas bajan todo a la
  tarjeta cada 2–5 s y antes de avisar por voz: **esperar a oír "Recorrido guardado" o
  "Sistema detenido" antes de desconectar.** Apagar bien sí se puede sin sudo
  (`systemctl poweroff`), falta ponerlo en un gesto del botón.
- Parar lo que esté corriendo (guarda el mapa o la grabación):
  `systemctl --user stop navia-arranque`. Desactivar: `systemctl --user disable navia-arranque`.
- ⚠️ **La espera del botón corre en un proceso aparte** (`--fase esperar`) que termina
  antes de lanzar el programa. La primera versión abría el botón en el mismo proceso
  que luego lanzaba el programa y, aunque lo cerraba, el programa recibía **`GPIO busy`**:
  se quedaba sin botón y con el micrófono siempre abierto. Con gpiozero/lgpio el pin
  no queda libre para otro proceso hasta que el que lo abrió **termina**.
- Espera hasta 25 s a que PipeWire vea la diadema USB y la deja como salida por defecto.

### 4.5 Audio: diadema USB

- **Jabra EVOLVE 20 MS** (USB-A): salida `alsa_output.usb-GN_Netcom…analog-stereo`,
  micrófono `alsa_input.usb-GN_Netcom…mono-fallback`. `audio_pi.py` la elige sola.
- Medido el 2026-11-10 (diadema sobre el escritorio, volumen 0.40): el micrófono marca
  −67 dBFS en silencio y −65 dBFS mientras el sistema habla por los audífonos; Vosk no
  reconoce nada de esa fuga. Por eso el micrófono se deja abierto mientras el sistema habla.
- ⚠️ **Sin micrófono conectado, PipeWire le conecta a `pw-record` el monitor de la salida**
  (el programa se graba a sí mismo). El proyecto graba pidiendo el micrófono por nombre y,
  si no hay, no graba.

---

## 5. Archivos del proyecto

### Programas principales

| Archivo | Corre en | Qué hace |
|---|---|---|
| `~/Desktop/code.py` | RP2040 (copiar a `CIRCUITPY/code.py`) | Programa de los sensores: remapeo y calibración de la IMU, lecturas de los dos MB1232, LED de estado. |
| `~/Desktop/boton.py` | Pi | Botón alterna el LED. |
| `~/gafas/probar_todo.py` | Pi | Prueba conjunta: muestra las líneas del RP2040 y cada pulsación enciende el LED y saca una foto en `~/gafas/fotos/`. |
| `~/gafas/prueba_transform.py` | Pi | Comprueba que Picamera2 gira la imagen 180° (foto cruda y girada en `/tmp`). |
| `~/gafas/arranque/arranque_boton.py` | Pi (solo, al prender) | Espera el botón 1 minuto: 1 clic = programa principal, 2 clics = grabar recorrido (§4.4). |

### Herramientas (`~/rp2040-sensores/`)

| Archivo | Corre en | Uso |
|---|---|---|
| `instalar.sh` | Pi | Graba CircuitPython (si la placa está en BOOTSEL) y copia `code.py` y `lib/`. |
| `entrar_bootsel.py` | Pi | Manda el RP2040 a BOOTSEL por software (REPL o truco de 1200 bps). |
| `ver_consola.py [s]` | Pi | Muestra la consola serie. **Reinicia `code.py`** (Ctrl-C + Ctrl-D). |
| `escuchar.py [s]` | Pi | Escucha la consola **sin** reiniciar `code.py`. Usar este durante la calibración. |
| `leer_offsets.py` | Pi | Para `code.py` y lee los offsets del BNO055 por registros, sin resetear el chip. |
| `listar_placa.py`, `bajar_archivo.py` | Pi | Listar y descargar archivos de una placa con MicroPython. |
| `diag_bno.py` | RP2040 | Diagnóstico por registros del BNO055 (IDs, estado, errores, modo). |
| `diag_orientacion.py` | RP2040 | Gravedad cruda por eje, para deducir cómo está montada la IMU. |
| `calibrar_imu.py` | RP2040 | Calibración guiada; imprime el estado y los offsets al llegar a 3/3/3. |
| `cambiar_direccion.py` | RP2040 | Cambia la dirección I²C de un MB1232, con comprobaciones previas. |
| `diag_ultrasonicos.py` | RP2040 | Compara cada MB1232 disparado solo y alternado, para detectar interferencia. |

Los scripts que "corren en RP2040" se usan copiándolos como `CIRCUITPY/code.py`.
Después hay que volver a copiar `~/Desktop/code.py` para restaurar el programa normal.

### Programas del proyecto (`~/TT-NavIA/`, desde 2026-11-10)

| Archivo | Qué hace |
|---|---|
| `firmware/code.py` | Firmware v2 del RP2040 (el que está cargado). |
| `sensores.py` | Lee el puerto serie del RP2040 en un hilo (IMU, pasos, ultrasónicos). `python sensores.py` muestra todo en vivo. |
| `camara.py` | Cámara por Picamera2, ya con el giro de 180° y el modo de sensor completo. `python camara.py foto.jpg` saca una foto de prueba. |
| `alertas.py` (`BotonPullUp`) | Botón en GPIO17 con pull-up y LED en GPIO27: un clic = escuchar / dejar de escuchar; doble clic = alertas sí/no. |
| `audio_pi.py` | Audio sin sudo (PortAudio local en `~/.local/lib/navia/`, diadema USB por PipeWire). |
| `calibrar_gafas.py` | Calibración guiada con las gafas puestas. El largo de paso se mide caminando hacia una pared (el ultrasonido mide el avance). |
| `grabar_sesion.py` | Graba un recorrido (cámara + todo lo que manda el RP2040) y al final dice cuánto se caminó. |
| `sesion_a_video.py` | Convierte una grabación en `video.mp4` para verla (`vlc`). Con `sesion_a_mapa.py --video` sale `video_sistema.mp4`, con lo que detecta el sistema. Copias en `~/Videos/NavIA/`. |
| `instalar_pi.sh` | Instala todo sin sudo y copia el firmware a `CIRCUITPY`. |

### Otros

- `~/rp2040-sensores/descargas/`: firmware `.uf2` de CircuitPython 10.3.1 y bundle de bibliotecas.
- `~/boton/prueba_conexion.py`: prueba de cableado del botón y el LED (parpadeo + señal cruda).
- `~/Desktop/codigopruebadesechable.cpp`: código Arduino original del proyecto. No se usa.
  Tiene el error del cristal externo (`setExtCrystalUse(true)`).

---

## 6. Problemas conocidos y lecciones

| Síntoma | Causa | Solución |
|---|---|---|
| LED del RP2040 encendido pero la Pi no lo detecta | Cable USB-C de solo carga (lleva VBUS, no D+/D−) | Usar un cable de datos |
| BNO055 responde en I²C, `euler` siempre `None`, calibración 0 | Cristal externo activado sin estar poblado | No activar `use_external_crystal` |
| `AttributeError: 'str' object has no attribute 'rjust'` | CircuitPython no implementa `str.rjust` | Usar ancho en el formato: `"%*s"` |
| Ultrasónico siempre `20 cm!` | Algo a ≤20 cm en el cono (zona muerta) | Despejar; no es una avería |
| Un ultrasónico `sin eco` y el otro bien | Desalineado, o apunta a un hueco o una superficie oblicua | Reorientar; mano a 50 cm para comprobar |
| La calibración se pierde al leer la consola | El constructor del driver resetea el BNO055 y `ver_consola.py` reinicia `code.py` | Usar `escuchar.py` |
| Giroscopio no pasa de G0 con las gafas puestas | El pulso y la respiración cuentan como movimiento | Dejar las gafas quietas sobre la mesa 5 s |
| Fotos al revés | Programa sin `Transform(hflip=1, vflip=1)` | Añadir el giro (§4.2) |
| La imagen se ve "con zoom" (menos campo de visión) | Picamera2 eligió el modo 1536×864, que es un recorte del sensor | Pedir `raw={"size": (2304, 1296)}` en la configuración (lo hace `camara.py`) |
| Un pitido tarda segundos en sonar | sounddevice/PortAudio sobre ALSA con PipeWire en medio (medido: de 3 a 49 s) | Reproducir con pygame (SDL → PipeWire): 0,30 s para un tono de 0,3 s |
| `PortAudio library not found` | La Pi no trae `libportaudio2` y no hay sudo | `instalar_pi.sh` lo baja con `apt-get download` a `~/.local/lib/navia/` |
| `terminate called without an active exception` al salir | Un hilo seguía dentro de onnxruntime (voz Piper) al cerrar Python | `ColaVoz.cerrar()` antes de salir (ya lo hace `main_nav_pi.py`) |
| La visión baja a menos de 1 cuadro/s cuando habla | Piper y NCNN usando los 4 núcleos a la vez | Piper con 1 hilo y NCNN con 3 (por defecto; `NAVIA_HILOS_VOZ`, `NAVIA_HILOS_VISION`) |
| Botón con lecturas aleatorias | Programa sin `pull_up=True` | Añadir el pull-up (§4.2) |
| Probar el botón sin picarlo (pruebas automáticas) | — | `pinctrl set 17 pd; sleep 0.12; pinctrl set 17 pu` baja el pin igual que una pulsación (con el botón suelto) y gpiozero la ve como real. Así se probó el arranque y los clics dentro del programa el 2026-11-12 |
| `rpicam-hello`: "Camera frontend has timed out! Please check that your camera sensor connector is attached securely" (la cámara **sí aparece** en `--list-cameras`) | El sensor contesta por I²C pero la imagen no llega por el cable plano: conector flojo o chueco, casi siempre tras mover las gafas. Comprobado el 2026-11-10 con PipeWire apagado y ningún programa usando la cámara: no es software | Apagar la Pi y volver a meter el cable plano en la cámara y en CAM0. `python camara.py` lo confirma en 7 s; `main_nav_pi.py` y `grabar_sesion.py` ya no arrancan a ciegas: dicen "no entrega imagen" y terminan |
| El programa lanzado por otro script no tiene botón (`GPIO busy`) | El script que lo lanza abrió el GPIO17/27 y sigue vivo; cerrar el `Button` no basta | Abrir el botón solo en un proceso que termine antes de lanzar (así lo hace `arranque_boton.py`, §4.4) |
| `Unable to set controls: Device or resource busy` al abrir la cámara recién prendida la Pi | Visto una vez a los 37 s del arranque; causa no confirmada (posiblemente PipeWire revisando la cámara al iniciar la sesión) | `camara.py` reintenta 3 veces antes de rendirse |
| Dos fotos con el mismo nombre | Pulsaciones en el mismo segundo | Corregido: nombre con milisegundos |
| `thonny --version` abre la ventana | Thonny no tiene modo CLI | Usar `dpkg -l thonny` |
| `pkill -f thonny` mata la propia shell | El patrón aparece en la propia línea de comandos | Usar `pgrep -af '[t]honny'` |

---

## 7. Pendientes

1. **Comprobar la resistencia del LED del botón** (§2.2).
2. Condensador de 10 µF en los MaxBotix: opcional. Ponerlo si aparecen lecturas
   erráticas con alimentación ruidosa (p. ej. con batería).
3. ~~Formato de salida del RP2040 pensado para personas.~~ **Resuelto el 2026-11-10**
   con el firmware v2 (CSV por línea, §3.2).
4. ~~Diadema USB sin probar.~~ **Conectada y probada el 2026-11-10** (Jabra EVOLVE 20,
   §4.5). La Pi 5 no tiene salida analógica de audio.
5. ~~Calibración con las gafas puestas.~~ **Hecha el 2026-11-10** (`imu_montaje.json`:
   cero del cabeceo −4.62°, rumbo invertido; ojos a 1.58 m). Falta solo el largo de
   paso propio (`calibrar_gafas.py --solo paso`, caminando hacia una pared); mientras
   tanto se usa el de un adulto promedio ajustado a la cadencia.
6. **Contador de pasos** (`sensores.py`: umbral 1.1 m/s², rearme 0.4): ajustado con una
   caminata real de 10 pasos y revisado con otra de 65 (2026-11-10). Validar con más
   recorridos grabados, ya con cuadros.
7. **Recorridos con imagen**: todavía no hay ninguno caminando (el del día 10 quedó sin
   cuadros). Con ellos se valida la visión (distancias por el piso) y los mapas.
