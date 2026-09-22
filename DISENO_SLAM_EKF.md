# Módulo SLAM simplificado con EKF — Diseño y documentación

## 1. Qué problema resuelve este módulo

El resto del sistema (mapa de ocupación, Ray Casting, navegación local)
necesita saber en todo momento **"el usuario está en (x, y) y mira hacia
theta"**, dentro del mapa local del cuarto. Ni la cámara ni los sensores
ultrasónicos entregan esa información directamente: la cámara da
*detecciones de obstáculos con dirección* (no distancia ni posición
absoluta) y el ultrasonido da *distancias puntuales* en direcciones fijas.
Además, cualquier estimación de movimiento (odometría, IMU, flujo óptico)
acumula error con el tiempo (deriva).

El SLAM resuelve esto:

1. Integra el movimiento estimado ciclo a ciclo (predicción).
2. Corrige esa predicción usando las observaciones del entorno,
   reconociendo elementos ya vistos antes (asociación de datos).
3. Mantiene una medida explícita de **cuánto se puede confiar** en la
   pose (matriz de covarianza / nivel de confianza).
4. Entrega la pose ya sincronizada con el mapa de ocupación, para que
   Ray Casting y la navegación local trabajen sobre coordenadas
   consistentes.

No se trata de "encontrar dónde está la cámara" en el sentido del SLAM
visual (ORB-SLAM, RTAB-Map, etc.): no se construye ni se rastrea una nube
de puntos 3D ni descriptores visuales. Es un **filtro de estado (EKF)
sobre una pose 2D**, corregido con observaciones simples de rango y
ángulo. Esto es intencional: es entendible, barato de ejecutar y
justificable académicamente, tal como pide el proyecto.

## 2. Decisión de diseño clave: pose en el EKF, mapa fuera del EKF

Un EKF-SLAM "clásico" aumenta el vector de estado con la posición de cada
landmark (`[x, y, theta, l1x, l1y, l2x, l2y, ...]`), lo que hace crecer la
covarianza de forma cuadrática con el número de landmarks. Para un
dispositivo con recursos limitados (Raspberry Pi) eso es indeseable.

Por eso aquí:

- El **estado del EKF es solo la pose**: `mu = [x, y, theta]`, con `P` de
  3×3. Todas las operaciones matriciales son de tamaño constante,
  independientemente de cuántos obstáculos haya.
- Los landmarks se guardan en una **lista ligera fuera del EKF**
  (`EKFSlam.landmarks`), usada únicamente para la asociación de datos
  (saber si una observación corresponde a "algo ya visto"). Esta lista es
  el puente natural hacia el mapa de ocupación real.
- La corrección del EKF usa el landmark asociado como si fuera conocido
  (su posición no se re-estima), lo cual es la simplificación central del
  diseño: se prioriza la precisión y bajo costo de la pose por encima de
  refinar el mapa dentro del filtro. El refinamiento del mapa en sí lo
  hace el módulo de Ray Casting / mapa de ocupación, con la pose que el
  SLAM le entrega.

Esta variante corresponde académicamente a un **EKF de localización con
mapeo incremental de apoyo** — una simplificación deliberada y
documentada del EKF-SLAM completo, coherente con lo pedido en el proyecto.

## 3. Arquitectura interna del módulo

```
                 ┌─────────────────────────────┐
 MotionEstimate  │                             │
 ───────────────►│                             │
                 │        EKFSlam.step()       │
 Observation[]   │                             │───► SlamOutput (pose,
 (visión+US)     │  1. predict()  (arco+Jx+Q)  │      covarianza,
 ───────────────►│  2. correct()               │      timestamp,
                 │     - _predict_observation  │      confianza)
                 │     - _associate  (NN+Mahal)│
                 │     - actualizar mu, P      │
                 │     - _update_confidence    │
                 └───────────────┬─────────────┘
                                 │ pose sincronizada
                                 ▼
                 ┌─────────────────────────────┐
                 │   OccupancyGrid (mapa +      │
                 │   Ray Casting)               │
                 │   - update_from_scan()       │
                 │   - get_nearby_obstacles()   │
                 │   - render_croquis()         │
                 └───────────────┬─────────────┘
                                 ▼
                        NAVEGACIÓN LOCAL
```

Módulos del paquete:

| Archivo               | Responsabilidad                                              |
|------------------------|--------------------------------------------------------------|
| `interfaces.py`        | Estructuras de datos compartidas (contratos entre módulos)   |
| `ekf_slam.py`          | El SLAM en sí (predicción, corrección, covarianza, salida)   |
| `occupancy_grid.py`    | Mapa de ocupación + Ray Casting (implementación mínima)      |
| `vision/TT-NavIA/`     | Módulo de visión REAL del equipo (YOLOv8-seg, sin modificar) |
| `vision_bridge.py`     | Conecta la visión real al SLAM: bbox → bearing + distancia monocular, filtro de objetos móviles, y fusión con ultrasonido |
| `sim_world.py`         | Cuarto simulado (paredes + obstáculos): ultrasonido por ray casting, visión sintética y movimiento por waypoints |
| `simulation.py`        | Simulación end-to-end sin hardware (valida el comportamiento requerido, reporta métricas) |
| `test_ekf_slam.py`     | Pruebas unitarias/funcionales del EKF, la fusión y la simulación |
| `metrics.py`           | ATE, error de orientación, incertidumbre, exactitud del mapa |
| `demo_realtime.py`     | Interfaz en tiempo real (cámara real + YOLO, o `--sim`), croquis al presionar `q` |
| `demo_mapa_camara.py`  | Mapeo en tiempo real SOLO CON LA CÁMARA (sin ultrasonido/IMU): YOLO real + distancias monoculares + giro estimado por flujo óptico (`VisualYawEstimator`); mapa y croquis en vivo |
| `demo_mapa_interactivo.py` | Mapeo en tiempo real sin ningún hardware: controlas al usuario con WASD (o piloto automático) en el cuarto simulado y ves el mapa, la pose estimada y la elipse de incertidumbre construirse en vivo |

Notas sobre la robustez añadida al EKF (implementadas en `ekf_slam.py`):

- **Candidatos con confirmación**: una observación que no se asocia con
  ningún landmark conocido crea un *candidato*; solo tras verse
  `confirm_hits` veces en posiciones consistentes se confirma y empieza a
  corregir la pose. Un falso positivo aislado no arrastra al filtro, y
  los candidatos que no se re-observan se podan (`candidate_ttl`). La
  observación que confirma a un candidato NO corrige la pose en ese mismo
  ciclo (evita contar la misma evidencia dos veces).
- **Piso de varianza en P**: la pose nunca se declara "perfecta"; sin
  esto, tras miles de correcciones el filtro se volvería sordo a
  observaciones nuevas (grave si al usuario lo mueven o tropieza).
- **Landmarks etiquetados** (`get_landmarks()`): el mapa ligero conserva
  la etiqueta de visión (chair, door, ...). Los landmarks NO son
  requisito para operar (sin ellos el SLAM sigue en dead-reckoning y la
  confianza baja), pero quedan listos para el módulo de rutas: p. ej.,
  enrutar al usuario hacia un landmark 'door'/'salida'.
- **Forma de Joseph** en la actualización de covarianza
  (`P = (I-KH) P (I-KH)^T + K R K^T`): numéricamente estable, P se
  mantiene simétrica y semidefinida positiva.
- **Ruido por observación**: cada `Observation` puede traer su propio
  `sigma_r`/`sigma_phi`; así un rango ultrasónico (±5 cm) pesa más que un
  rango monocular (±30 %), sin ramas especiales en el filtro.
- **Incertidumbre del landmark en S**: como el landmark no está en el
  estado, su error de posición (que decrece ~1/sqrt(n) con las
  re-observaciones) se suma a S en cada corrección; el filtro no trata a
  un landmark recién visto como verdad absoluta.
- **Ecos de pared**: un eco ultrasónico sin detección visual asociada
  puede corregir y crear landmarks, pero exige más avistamientos para
  confirmarse (`min_confirm_hits=3`), porque el punto de reflexión de una
  pared se desliza al caminar. Las detecciones monoculares sin respaldo
  ultrasónico exigen 4 (un falso positivo de YOLO de 1-2 frames no debe
  aparecer como obstáculo "inventado").
- **Sin landmarks duplicados**: la fusión de candidatos usa la ELIPSE de
  incertidumbre de la observación (estrecha en dirección, larga en
  profundidad) en lugar de un radio fijo: la misma silla vista a 2.1 m y
  luego a 2.9 m (rango monocular ±30 %) es un solo landmark. Además la
  asociación respeta las etiquetas (chair != table; los ecos genéricos
  son comodín y heredan la etiqueta si visión los identifica), y una
  pasada de deduplicación fusiona confirmados del mismo tipo que la
  deriva haya separado.
- **Mapa sin manchas** (`observations_for_map()`): al mapa de ocupación
  van las observaciones ANCLADAS a la posición consolidada del landmark
  asociado, no el rango crudo: un objeto = una marca firme en la misma
  celda, en vez de una mancha dispersa por el ruido monocular.

## 4. Entradas y salidas

**Entradas por ciclo** (`interfaces.py`):

- `MotionEstimate(delta_trans, delta_rot, dt)` — desplazamiento y giro
  desde la última iteración. *Interfaz simulada por ahora
  (`sim_world.SimulatedMotion`); debe conectarse al módulo real de
  odometría/IMU cuando exista.*
- `VisionDetection(bearing, label, range_est, moving, ...)` — obstáculo
  detectado por la cámara (de `vision_bridge.py` sobre el módulo real
  YOLOv8-seg): dirección precisa, distancia monocular gruesa, y si el
  objeto se está moviendo (los objetos móviles no se usan como landmarks).
- `UltrasonicReading(range_m, sensor_bearing, valid)` — distancia medida
  por un sensor ultrasónico. *Interfaz simulada por ahora
  (`sim_world.SimulatedUltrasonicArray`, por ray casting contra el cuarto
  simulado); debe conectarse al sensor físico.*
- `Observation(range_m, bearing, label, sigma_r, sigma_phi)` — observación
  fusionada (visión + ultrasonido, `vision_bridge.fuse_observations`) que
  efectivamente entra al EKF, con su propia incertidumbre.

**Salida por ciclo** (`SlamOutput.as_dict()`):

```yaml
pose:
  x: 1.83
  y: 0.42
  theta: 0.31
covarianza:
  - [0.0021, 0.0001, 0.0000]
  - [0.0001, 0.0019, 0.0000]
  - [0.0000, 0.0000, 0.0007]
timestamp: 1737657000.231
confianza: 0.94
```

## 5. Representación matemática de la pose

Estado: `mu = [x, y, theta]^T` (m, m, rad). Covarianza: `P` (3×3, PSD).

## 6. Modelo de movimiento (predicción)

Dado `delta_trans = d` y `delta_rot = dθ` desde la última iteración,
modelo de "arco" (traslación aplicada en la orientación promedio del
tramo):

```
theta_mid = theta + dθ/2
x'     = x + d·cos(theta_mid)
y'     = y + d·sin(theta_mid)
theta' = theta + dθ           (normalizado a (-π, π])
```

Jacobiano respecto al estado:

```
Gx = | 1  0  -d·sin(theta_mid) |
     | 0  1   d·cos(theta_mid) |
     | 0  0   1                |
```

Ruido de proceso (crece con la magnitud del movimiento):

```
sigma_trans = a0 + a1·|d|
sigma_rot   = a2·max(|dθ|, 0.01) + a3
Q = diag(sigma_trans², sigma_trans², sigma_rot²)
```

Predicción de covarianza: `P' = Gx·P·Gx^T + Q`.

## 7. Modelo de observación

Para un landmark conocido en `(lx, ly)`:

```
dx = lx - x ;  dy = ly - y ;  q = dx² + dy²
r_hat   = sqrt(q)
phi_hat = atan2(dy, dx) - theta        (normalizado)

H = | -dx/r_hat   -dy/r_hat    0 |
    |  dy/q        -dx/q      -1 |
```

`z = [range_m, bearing]` es la observación fusionada visión+ultrasonido.

## 8. Asociación de datos

Nearest-neighbor con "gating" por distancia de Mahalanobis:

```
innov = z - z_hat  (normalizando el bearing)
S = H·P·H^T + R                  R = diag(sigma_r², sigma_phi²)
d² = innov^T · S^-1 · innov
si d² < umbral² -> asociar con ese landmark
si no -> registrar como landmark candidato nuevo (mapa ligero)
```

## 9. Corrección (EKF update)

```
K = P·H^T·S^-1
mu = mu + K·innov      (normalizar theta)
P  = (I - K·H)·P
```

## 10. Manejo de la matriz de covarianza

- Crece en la predicción (incertidumbre acumulada por el movimiento).
- Se reduce en la corrección (cuando hay observaciones asociadas).
- Se reporta siempre como parte de `SlamOutput` (3×3), nunca se descarta.
- La **confianza** reportada combina `1/(1+traza(P))` con una
  penalización exponencial por ciclos consecutivos sin corrección.

## 11. Manejo de pérdida temporal de observaciones

Si en un ciclo no se asocia ninguna observación con un landmark conocido
(oclusión, entorno sin features, sensor sin eco válido), **no se ejecuta
la corrección**: el SLAM continúa solo con la predicción (dead-reckoning),
y la confianza reportada baja progresivamente para que el resto del
sistema (navegación) sepa que la pose es menos fiable en ese momento.

## 12. Integración con el mapa de ocupación y con Ray Casting

- El SLAM **nunca** escribe directamente en el mapa de ocupación.
- En cada ciclo, la pose (`x, y, theta`) y las observaciones se entregan a
  `OccupancyGrid.update_from_scan()`, que hace el Ray Casting (proyecta
  rayos, marca celdas libres a lo largo del rayo y la celda ocupada al
  final).
- El SLAM puede consultar `OccupancyGrid.get_nearby_obstacles()` si se
  desea usar el mapa (en vez de la lista ligera interna) como fuente de
  landmarks para la asociación — la interfaz ya lo permite, sin mezclar
  responsabilidades.

## 13. Funcionamiento en tiempo real

- Estado de 3 variables, covarianza 3×3: coste por ciclo ~O(1) en el EKF
  en sí (independiente del tamaño del mapa).
- La asociación de datos es O(número de landmarks conocidos); en un
  cuarto típico esto es un número pequeño (decenas), no miles.
- El Ray Casting usa Bresenham (enteros, sin trigonometría por celda).
- La inferencia de visión (YOLOv8-seg en `vision/TT-NavIA`) corre a
  640×480 y es, por mucho, el paso más caro del ciclo; el EKF consume su
  salida de forma asíncrona-amigable (si visión se salta ciclos, el SLAM
  sigue en dead-reckoning y la confianza lo refleja).

## 14. Pseudocódigo completo del ciclo incremental

```
función SLAM_ciclo(motion, detecciones_vision, lecturas_ultrasonido):
    # 1. Obtener movimiento
    d, dtheta = motion.delta_trans, motion.delta_rot

    # 2. Predicción
    theta_mid = theta + dtheta/2
    x  = x + d*cos(theta_mid)
    y  = y + d*sin(theta_mid)
    theta = normalizar(theta + dtheta)
    Gx = jacobiano_movimiento(d, theta_mid)
    Q  = ruido_proceso(d, dtheta)
    P  = Gx * P * Gx^T + Q

    # 3. Fusionar visión + ultrasonido -> observaciones (rango, bearing)
    observaciones = fusionar(detecciones_vision, lecturas_ultrasonido)

    # 4. Corrección
    hubo_correccion = falso
    para cada obs en observaciones:
        landmark, H, S, innov = asociar(obs, landmarks_conocidos, P)
        si landmark encontrado:
            K = P * H^T * S^-1
            (x, y, theta) += K * innov   ; normalizar theta
            P = (I - K*H) * P
            hubo_correccion = verdadero
        si_no:
            registrar_landmark_candidato(obs, x, y, theta)

    si no hubo_correccion:
        incrementar contador_sin_correccion
    si_no:
        contador_sin_correccion = 0

    confianza = calcular_confianza(P, contador_sin_correccion)

    # 5. Entregar pose al resto del sistema
    salida = { pose: (x,y,theta), covarianza: P, timestamp: ahora(),
               confianza: confianza }

    # 6. Sincronizar con el mapa (Ray Casting, módulo separado)
    mapa_ocupacion.actualizar_con_rayos((x,y,theta), observaciones)

    devolver salida
```

## 15. Qué información recibe el módulo de navegación y por qué le alcanza

El SLAM entrega, en cada ciclo, exactamente:

- **pose (x, y, theta)**: dónde está el usuario y hacia dónde mira,
  dentro del mismo sistema de coordenadas que usa el mapa de ocupación
  (por eso la sincronización con Ray Casting es explícita).
- **covarianza (3×3)**: qué tan confiable es esa pose en cada eje
  (permite a la navegación, por ejemplo, ser más conservadora si `P` es
  grande).
- **confianza (escalar 0–1)**: resumen simple de lo anterior, útil para
  decisiones rápidas (p. ej., "si confianza < 0.5, reducir velocidad
  sugerida" o "pedir al usuario detenerse un momento").
- **timestamp**: para que la navegación pueda descartar poses demasiado
  antiguas si el ciclo se retrasó.

Con `pose + mapa de ocupación` (celdas libres/ocupadas alrededor del
usuario), la navegación local ya tiene todo lo necesario para decidir
movimientos seguros: sabe "dónde estoy", "hacia dónde miro" y "qué hay
libre/ocupado a mi alrededor en ese mismo sistema de coordenadas" — sin
necesitar en ningún momento una posición GPS ni ubicación externa.

## 16. Pruebas y métricas

- `test_ekf_slam.py`: valida predicción (avance/rotación), crecimiento de
  incertidumbre en la predicción, reducción de incertidumbre al corregir,
  registro de nuevos landmarks, comportamiento de dead-reckoning sin
  observaciones, y la forma de la salida.
- `simulation.py`: escenario simulado end-to-end (posición inicial
  conocida → avanza → gira → encuentra landmarks → pose se actualiza →
  incertidumbre cambia), genera un croquis final y reporta:
  - **ATE** (error medio de trayectoria, m)
  - **error medio de orientación** (°)
  - evolución de la incertidumbre (traza de `P`)
- `metrics.py`: funciones reutilizables para estas métricas y para
  comparar el mapa estimado contra un mapa de referencia si se dispone de
  uno (útil en pruebas con datos reales más adelante).

## 17. Cómo pasar de simulado a real

Todo lo simulado está aislado en `sim_world.py` y marcado con `TODO`:

- **Visión: YA ES REAL.** `vision_bridge.VisionBridge` ejecuta el módulo
  del equipo (`vision/TT-NavIA`: YOLOv8-seg + clasificador + tracker de
  movimiento) y `demo_realtime.py` (modo cámara) lo usa por defecto.
- `sim_world.SimulatedMotion` → reemplazar por el módulo real de
  odometría/IMU, produciendo `MotionEstimate`. Mientras tanto, el modo
  cámara asume usuario cuasi-estático (`MotionEstimate(0, 0)`).
- `sim_world.SimulatedUltrasonicArray` → reemplazar por lectura real de
  los sensores (HC-SR04), produciendo `UltrasonicReading`. En cuanto
  existan, basta pasarlas a `fuse_observations(detecciones, lecturas)` en
  `demo_realtime.py`; sin ellas, el rango proviene de la estimación
  monocular de visión (con incertidumbre mayor, que el EKF ya pondera).
- `OccupancyGrid` puede sustituirse por el módulo real de mapa de
  ocupación/Ray Casting del equipo, siempre que exponga
  `update_from_scan`, `get_nearby_obstacles` y `render_croquis` (o una
  función equivalente para obtener el croquis final).

## 18. Resultados de referencia (simulación)

Con el cuarto simulado de 6 x 4 m (`sim_world.default_room`), 250 ciclos,
odometría con ruido, 3 sensores ultrasónicos y visión sintética con 25 %
de error de rango (promedio sobre 6 semillas):

- ATE: ~0.20 m (máx 0.24 m)
- Error medio de orientación: ~3°
- Con 60 % de detecciones visuales perdidas (oclusión), el sistema se
  mantiene por debajo de 0.45 m de ATE gracias al dead-reckoning + eco
  ultrasónico, y la confianza reportada baja como se espera.

Reproducir con: `python simulation.py` y `python -m unittest test_ekf_slam`.
