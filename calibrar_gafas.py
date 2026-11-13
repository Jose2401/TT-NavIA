"""
calibrar_gafas.py — calibracion guiada con los lentes PUESTOS. se hace
una vez por persona (y otra vez si se mueve la imu de lugar):

  1. montaje de la imu: cabeza derecha / agachada / girada a la
     izquierda -> imu_montaje.json (cual campo es el cabeceo, sus
     signos y el cero)
  2. altura de los ojos al piso -> la usa la distancia por plano de piso
  3. largo de paso: caminar derecho hacia una pared (el ultrasonido
     mide el avance; ya no hace falta marcar 10 m). ojo: el sistema
     tambien lo va afinando solo al final de cada corrida en que el
     ultrasonido vio acercarse algo (SensoresGafas.aprenderPaso)

lo de la persona queda en calibracion_usuario.json; los mains lo leen
solos al arrancar (montaje.py).

uso:
    python calibrar_gafas.py              (todo)
    python calibrar_gafas.py --solo paso              (hacia una pared)
    python calibrar_gafas.py --solo paso --metros 10  (distancia medida)
    python calibrar_gafas.py --archivo sesiones/calibracion_pasos.txt
"""
import argparse
import os
import time


def _esperar(texto, boton=None):
    """enter en el teclado o un click del boton de la pi."""
    if boton is None or not boton.disponible:
        input(texto + " [Enter] ")
        return
    import threading
    listo = threading.Event()
    boton.alClick = listo.set
    boton.alCambiar = listo.set

    def teclado():
        try:
            input()
        except EOFError:
            return
        listo.set()
    threading.Thread(target=teclado, daemon=True).start()
    print(texto + " [Enter o el botón] ", end="", flush=True)
    listo.wait()
    print()


def largoDeCaminata(r, metros=None):
    """del resumen de una caminata (sensores.resumenCaminata) saca el
    largo de paso, o (None, motivo). con `metros` medidos se usa eso;
    si no, lo que avanzo segun el ultrasonido entre los pasos que se
    contaron en esos tramos."""
    if r["pasos"] < 5:
        return None, (f"muy pocos pasos ({r['pasos']}): si sí caminaste, "
                      "manda el archivo de la señal pa ajustar el detector")
    if metros:
        largo = metros / r["pasos"]
    else:
        e = r.get("eco")
        if not e or e["metros"] < 2.0 or e["pasos"] < 4:
            return None, ("el ultrasonido no vio una pared acercándose al "
                          "menos 2 m: camina DERECHO hacia una pared, o "
                          "mide la distancia y usa --metros")
        largo = e["largoPasoM"]
    if not 0.25 <= largo <= 1.0:
        return None, f"largo de paso raro ({largo:.2f} m): no se guarda"
    return round(largo, 3), ""


def calibrarPaso(sens, metros=None, boton=None, carpeta="sesiones"):
    """ya no hace falta medir 10 m: se camina derecho hacia una pared y
    el ultrasonido mide el avance. guarda tambien la señal cruda pa
    poder ajustar el detector si cuenta mal."""
    import audio_pi
    from sensores import resumenCaminata, textoResumen
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, "calibracion_pasos.txt")
    if metros:
        print(f"\n== largo de paso: marca {metros:g} m en el piso ==")
        _esperar("párate al inicio, viendo al frente, y avisa", boton)
    else:
        print("\n== largo de paso: párate a unos 4 a 6 m de una pared o "
              "puerta cerrada, viéndola de frente ==")
        _esperar("cuando estés listo avisa", boton)
    audio_pi.tonito("escucha_on")
    t0 = time.time()
    with open(ruta, "w", encoding="utf-8") as fh:
        sens.alLinea = lambda linea, t: fh.write(f"{t - t0:.4f} {linea.strip()}\n")
        _esperar("camina derecho a tu paso normal, detente "
                 + ("al final" if metros else "antes de tocar la pared")
                 + " y avisa", boton)
        sens.alLinea = None
    audio_pi.tonito("escucha_off")
    r = resumenCaminata(ruta)
    print(f"{textoResumen(r)}  (señal cruda en {ruta})")
    return largoDeCaminata(r, metros)


def main():
    p = argparse.ArgumentParser(description="calibración guiada de las gafas")
    p.add_argument("--solo", choices=("imu", "altura", "paso"), default=None)
    p.add_argument("--metros", type=float, default=None,
                   help="distancia medida que se camina (si no se da, la "
                        "mide el ultrasonido caminando hacia una pared)")
    p.add_argument("--archivo", nargs="+", default=None,
                   help="sacar el largo de paso de caminatas ya grabadas "
                        "(sesiones/x/serie.txt, una o varias) en vez de "
                        "caminar")
    args = p.parse_args()

    if args.archivo:
        import montaje
        from sensores import resumenCaminata, textoResumen
        metros = pasos = 0.0
        for ruta in args.archivo:
            r = resumenCaminata(ruta)
            print(f"{ruta}: {textoResumen(r)}")
            if len(args.archivo) == 1:
                largo, motivo = largoDeCaminata(r, args.metros)
                if largo is None:
                    raise SystemExit(motivo)
            elif r.get("eco"):
                metros += r["eco"]["metros"]
                pasos += r["eco"]["pasos"]
        if len(args.archivo) > 1:
            if metros < 2.0 or pasos < 4:
                raise SystemExit("en esas caminatas el ultrasonido no vio "
                                 "suficiente pared acercándose")
            largo = round(metros / pasos, 3)
            print(f"juntas: {metros:.1f} m en {pasos:.0f} pasos")
        montaje.guardarUsuario(largoPasoM=largo, pasoFuente="ultrasonido",
                               pasoFecha=time.strftime("%Y-%m-%d"))
        print(f"guardado: paso de {largo:.2f} m -> "
              f"{montaje.CALIBRACION_USUARIO}")
        return

    import montaje
    from sensores import SensoresGafas, ImuGafas
    from odometry import calibrarMontaje
    from alertas import BotonPullUp

    sens = SensoresGafas()
    if not sens.imuOk:
        raise SystemExit(f"sin imu de las gafas ({sens.razon})")
    boton = BotonPullUp(ventanaS=0.05)
    c = sens.calib or {}
    print(f"imu lista. calibración del chip: sys {c.get('sys')} giro "
          f"{c.get('gyro')} acel {c.get('accel')} mag {c.get('mag')} (0-3)")

    try:
        if args.solo in (None, "imu"):
            print("\n== montaje de la imu (ponte los lentes) ==")
            calibrarMontaje(ImuGafas(sens, montajePath="/no/usar/el/viejo"),
                            entrada=lambda t: _esperar(
                                t.replace(" y Enter...", "")
                                .replace(" Enter...", ""), boton))
        if args.solo in (None, "altura"):
            print(f"\n== altura de tus ojos al piso (ahorita "
                  f"{montaje.CAM_ALTURA_M:.2f} m) ==")
            txt = input("en metros, p.ej. 1.58 (Enter = dejarla igual): ")
            if txt.strip():
                h = float(txt.replace(",", "."))
                if 1.0 <= h <= 2.2:
                    montaje.guardarUsuario(alturaCamaraM=round(h, 2))
                    print(f"guardada: {h:.2f} m")
                else:
                    print("fuera de rango (1.0 a 2.2 m), no se guarda")
        if args.solo in (None, "paso"):
            largo, motivo = calibrarPaso(sens, args.metros, boton)
            if largo is None:
                print(f"no se cambia el largo de paso: {motivo}")
            else:
                montaje.guardarUsuario(
                    largoPasoM=largo,
                    pasoFuente="cinta" if args.metros else "ultrasonido",
                    pasoFecha=time.strftime("%Y-%m-%d"))
                print(f"paso de {largo:.2f} m, guardado en "
                      f"{montaje.CALIBRACION_USUARIO}")
    finally:
        sens.cerrar()
        boton.cerrar()
    print("\nlisto. los mains ya usan esta calibración.")


if __name__ == "__main__":
    main()
