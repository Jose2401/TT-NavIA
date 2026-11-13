# code.py — firmware de las gafas NavIA (Waveshare RP2040-Zero, CircuitPython 10.x)
#
# lee la IMU BNO055 y los dos MB1232 por i2c y lo manda a la pi por usb
# serie en lineas csv faciles de parsear (sensores.py las lee).
# copiar a CIRCUITPY/code.py (se recarga solo al guardar).
#
# cableado (todo a 3V3, NUNCA a 5V): GP0 = SDA, GP1 = SCL, bus a 100 kHz
#   0x28 BNO055 | 0x71 MB1232 derecho | 0x70 MB1232 izquierdo
#
# protocolo (una linea por dato, puros enteros):
#   H,navia,<version>                 al arrancar y cada 5 s
#   I,ms,h,r,p,wx,wy,wz,lx,ly,lz,gx,gy,gz,cal
#       h r p      euler en 1/16 de grado (heading, roll, pitch)
#       wx wy wz   giroscopio en 1/16 de grado/s
#       lx ly lz   aceleracion lineal (sin gravedad) en cm/s2
#       gx gy gz   vector de gravedad en cm/s2
#       cal        byte de calibracion (sys<<6 | gyr<<4 | acc<<2 | mag)
#   U,ms,D|I,cm                       un ultrasonico (765 = sin eco,
#                                     20 = a 20 cm o MENOS, -1 = no responde)
#   E,texto                           error / aviso
#   O,...                             offsets de calibracion (al mandar "o")
#
# los ultrasonicos se disparan por turnos (si no, uno oye al otro) y sin
# bloquear: mientras uno mide (100 ms) se sigue leyendo la imu a 50 Hz,
# que es lo que ocupa el contador de pasos.

import sys
import time

import board
import busio
import supervisor

VERSION = 2

DIR_BNO055 = 0x28
AXIS_REMAP_GAFAS = (0, 2, 1, 1, 0, 0)   # imu de pie en el puente: X<-(-X) Y<-Z Z<-Y

# calibracion hecha con las gafas puestas (2026-11-09). el chip la pierde
# al apagarse; se restaura en cada arranque y sigue autocalibrandose.
# si se mueve la electronica de lugar hay que repetirla (calibrar_imu.py)
CAL_ACC = (19, -21, -76)
CAL_MAG = (-562, 83, 19)
CAL_GYR = (3, -1, -1)
CAL_RADIO_ACC = 1000
CAL_RADIO_MAG = 751

CMD_MEDIR = 0x51
ESPERA_ECO_MS = 100      # el mb1232 pide ~100 ms entre ordenar y leer
PERIODO_IMU_MS = 20      # 50 Hz
ULTRASONICOS = (("D", 0x71), ("I", 0x70))

REG_GYRO = 0x14          # de aqui a 0x35 van giro, euler, quat, lia, grv, temp, calib
N_BLOQUE = 34

i2c = busio.I2C(board.GP1, board.GP0, frequency=100_000)   # (SCL, SDA)


def iniciarLed():
    try:
        import neopixel
        return neopixel.NeoPixel(board.NEOPIXEL, 1, brightness=0.1,
                                 auto_write=True)
    except Exception:
        return None


def pintar(led, imuOk, ecoOk):
    if led is None:
        return
    if imuOk and ecoOk:
        led[0] = (0, 255, 0)        # verde: todo responde
    elif imuOk or ecoOk:
        led[0] = (255, 120, 0)      # ambar: falta la imu o los ultrasonicos
    else:
        led[0] = (255, 0, 0)        # rojo: nada


def iniciarImu():
    try:
        import adafruit_bno055
        imu = adafruit_bno055.BNO055_I2C(i2c, address=DIR_BNO055)
        imu.axis_remap = AXIS_REMAP_GAFAS
        imu.offsets_accelerometer = CAL_ACC
        imu.offsets_magnetometer = CAL_MAG
        imu.offsets_gyroscope = CAL_GYR
        imu.radius_accelerometer = CAL_RADIO_ACC
        imu.radius_magnetometer = CAL_RADIO_MAG
        # ojo: NO activar use_external_crystal, esta placa no trae el
        # cristal y el chip se queda sin reloj pa la fusion
        return imu
    except Exception as e:
        print("E,imu no inicializada: %s" % e)
        return None


def s16(buf, i):
    v = buf[i] | (buf[i + 1] << 8)
    return v - 65536 if v > 32767 else v


_reg = bytes([REG_GYRO])
_buf = bytearray(N_BLOQUE)


def leerImu(imu):
    """una sola transaccion i2c con todo lo que se ocupa."""
    with imu.i2c_device as dev:
        dev.write_then_readinto(_reg, _buf)
    b = _buf
    #        giro (0..5)                 euler (6..11)
    return (s16(b, 6), s16(b, 8), s16(b, 10),
            s16(b, 0), s16(b, 2), s16(b, 4),
            # lia (20..25)               gravedad (26..31)        calib (33)
            s16(b, 20), s16(b, 22), s16(b, 24),
            s16(b, 26), s16(b, 28), s16(b, 30), b[33])


def usOrdenar(direccion):
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(direccion, bytes([CMD_MEDIR]))
        return True
    except OSError:
        return False
    finally:
        i2c.unlock()


_buf2 = bytearray(2)


def usLeer(direccion):
    while not i2c.try_lock():
        pass
    try:
        i2c.readfrom_into(direccion, _buf2)
        return (_buf2[0] << 8) | _buf2[1]
    except OSError:
        return -1
    finally:
        i2c.unlock()


def atenderComando(imu):
    # "o" = imprimir los offsets actuales de calibracion (pa guardarlos
    # arriba). ojo: leerlos pasa el chip a modo config un instante
    n = supervisor.runtime.serial_bytes_available
    if not n:
        return
    c = sys.stdin.read(n)
    if "o" in c and imu is not None:
        try:
            print("O,%s,%s,%s,%d,%d" % (
                imu.offsets_accelerometer, imu.offsets_magnetometer,
                imu.offsets_gyroscope, imu.radius_accelerometer,
                imu.radius_magnetometer))
        except Exception as e:
            print("E,offsets: %s" % e)


led = iniciarLed()
pintar(led, False, False)
print("H,navia,%d" % VERSION)
imu = iniciarImu()

ticks = supervisor.ticks_ms
_MASK = (1 << 29) - 1


def desde(t0):
    return (ticks() - t0) & _MASK


tImu = ticks()
tUs = ticks()
tReintento = ticks()
tHola = ticks()
usIdx = 0
usMidiendo = False
ecoOk = False
fallosImu = 0

while True:
    # --- ultrasonicos: maquina de estados, por turnos ---
    if not usMidiendo:
        if usOrdenar(ULTRASONICOS[usIdx][1]):
            usMidiendo = True
            tUs = ticks()
        else:
            print("U,%d,%s,-1" % (ticks(), ULTRASONICOS[usIdx][0]))
            ecoOk = False
            usIdx ^= 1
            time.sleep(0.02)
    elif desde(tUs) >= ESPERA_ECO_MS:
        cm = usLeer(ULTRASONICOS[usIdx][1])
        print("U,%d,%s,%d" % (ticks(), ULTRASONICOS[usIdx][0], cm))
        ecoOk = cm >= 0
        usMidiendo = False
        usIdx ^= 1

    # --- imu a 50 Hz ---
    if desde(tImu) >= PERIODO_IMU_MS:
        tImu = ticks()
        if imu is not None:
            try:
                print("I,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d"
                      % ((tImu,) + leerImu(imu)))
                fallosImu = 0
            except OSError:
                fallosImu += 1
                if fallosImu > 10:
                    print("E,imu sin respuesta")
                    imu = None
        elif desde(tReintento) > 3000:
            tReintento = ticks()
            imu = iniciarImu()
        pintar(led, imu is not None, ecoOk)
        atenderComando(imu)
        if desde(tHola) > 5000:
            tHola = ticks()
            print("H,navia,%d" % VERSION)
