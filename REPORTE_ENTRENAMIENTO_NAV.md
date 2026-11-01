# Reporte de entrenamiento — Modelo de navegación (DRL)

**Modelo final:** `models/nav_ppo` (`.zip` SB3 + `_policy.npz` numpy puro)
**Fecha:** 2026-10-15 · **Algoritmo:** PPO actor-crítico (Stable-Baselines3 + PyTorch) · **Pasos:** 6,000,000 (seed 2) · **Duración del entrenamiento:** ~10 min en laptop (CPU, 8 entornos en paralelo)

---

## 1. Resultado del modelo final

Evaluación sobre **520 episodios** (120 en mundos procedurales nuevos + 400 sobre los 10 mapas construidos de videos reales):

| Métrica global | Valor |
|---|---|
| Éxito (llega al destino) | **98.1 %** |
| Colisión | 1.9 % |
| Caída en hoyo/coladera | **0.0 %** |
| Timeout (no llega en 300 pasos) | 0.0 % |
| Pasos promedio por episodio exitoso | 29 |

### Desglose por tipo de mundo

| Mundo | Éxito | Choque/caída |
|---|---|---|
| Cuarto procedural | 100 % | 0 % |
| Dos cuartos con puerta (procedural) | 95 % | 5 % |
| Exterior con coladeras (procedural) | 100 % | 0 % |
| **Real** v01 calle 1 | 100 % | 0 % |
| **Real** v02 calle 2 | 100 % | 0 % |
| **Real** v03 casa | 100 % | 0 % |
| **Real** v04 pasillo largo | 90 % | 10 % |
| **Real** v05 plaza comercial | 100 % | 0 % |
| **Real** v06 oficina 1 | 100 % | 0 % |
| **Real** v07 oficina 2 | 100 % | 0 % |
| **Real** v08 vertical (piso) | 95 % | 5 % |
| **Real** v09 oficina 3 | 95 % | 5 % |
| **Real** v10 oficina 4 | 100 % | 0 % |

---

## 2. Cómo se entrenó

### Entorno (Gymnasium, `navigation/nav_env.py`)
- **Acciones (4, discretas, según TT §7.3.1):** avanzar 0.30 m · girar ±20° · detenerse.
- **Observación "extended" (24 valores):** 11 rayos de distancia (−100° a +100°), distancia a peligros de piso por sector (los hoyos **no** aparecen en los rayos porque no bloquean nada), riesgos por sector (0–3, tabla 7.1 del TT), meta como (sin φ, cos φ, distancia) y acción previa.
- **Recompensa (TT §7.3.1 con escalas ajustadas):** +100 llegar · −100 colisión o caída · progreso ×4 por Δd a la meta · −0.1 por paso · penalización por riesgo frontal · "detenerse" solo premia con un móvil encima.
- **Mundos procedurales:** cuartos (40 %), dos cuartos conectados por puerta (35 %), exteriores con postes y coladeras (25 %); 0–3 obstáculos móviles (personas) que rebotan.
- **Randomización de dominio:** ruido de 5 % en sensores, ±2° en el ángulo a la meta, y 8 % de rayos bloqueados "soltados" a rango máximo (simula los huecos entre celdas del mapa real).

### Mapas reales (videos → `maps/real/`)
Los 10 videos de `videosxd/` (calles, casa, oficinas, plaza; distintos celulares, gran angular ~110°) se procesaron con `video_a_mapa.py`: visión YOLOv8-seg + giro por flujo óptico + detector de caminata (avance a 1.1 m/s cuando el flujo indica que se camina) + huella libre del recorrido. En entrenamiento, el 45 % de los episodios se jugó sobre estos mapas.

> Nota honesta: sin odometría física, la **escala** de estos mapas depende de la velocidad de caminata asumida. Sirven perfectamente como *geometrías de entrenamiento* (pasillos, vueltas, anchos reales), no como planos a escala exacta.

---

## 3. Historia de iteraciones (todas evaluadas con el mismo protocolo)

| # | Config | Éxito procedural | Éxito mapas reales | Nota |
|---|---|---|---|---|
| 0a | Obs mínima del documento (8 valores), 300k pasos | 44.2 % | — | Fracasa en puertas (10 %): 3 sectores no "ven" el hueco |
| 0b | Obs extendida (24), 300k pasos | 85.8 % | — | **Se elige la extendida** |
| 1 | Extendida, 4M pasos, solo procedural (`nav_ppo_simonly`) | 99.2 % | ~95–100 % (sin entrenar en ellos) | Primer modelo "bueno" |
| 2 | 4M pasos, 35 % mapas reales, seed 1 | 96.5 % | 77.5–100 % | v04_pasillo débil (77.5 %) |
| **3** | **6M pasos, 45 % mapas reales, seed 2 → `nav_ppo` FINAL** | **98.1 %** | **90–100 %** | v04 sube a 90 %; 7/10 mapas al 100 % |

Decisiones técnicas que salieron de estas iteraciones:
- La observación extendida domina a la mínima del documento (85.8 % vs 44.2 % a igual presupuesto), sobre todo en mundos con puertas.
- La brecha mapa-real↔simulador (rayos que se "cuelan" entre celdas de eco) se cerró por tres lados: abanico de sub-rayos en inferencia, dropout de rayos en entrenamiento, y la capa de seguridad determinística (RNF-09) que anula "avanzar" con el frente a <0.6 m.

---

## 4. Artefactos entregados

| Archivo | Qué es |
|---|---|
| `models/nav_ppo.zip` | Modelo SB3 completo (laptop, seguir entrenando) |
| `models/nav_ppo_policy.npz` | Solo pesos del actor — **la Raspberry navega con numpy puro**, sin SB3 (test verifica decisión idéntica) |
| `models/nav_ppo_meta.json` | Config + evaluación completa |
| `models/nav_ppo_simonly*` | Respaldo del modelo solo-simulación |

**Reproducir / reentrenar:**
```bash
python -m navigation.train --timesteps 6000000 --envs 8 --seed 2 \
    --real-maps maps/real/v01_calle1 ... maps/real/v10_oficina4 \
    --real-prob 0.45 --out models/nav_ppo

# solo evaluar:
python -m navigation.train --eval models/nav_ppo.zip --real-maps maps/real/v04_pasillo
```

---

## 5. Limitaciones y cuándo reentrenar

1. **El reentrenamiento que más va a importar es cuando llegue la IMU BNO055 + los MB1232:** se miden los ruidos reales, se recalibra el simulador y se reentrena (~10 min). Hasta entonces este modelo es el definitivo de esta etapa.
2. Los mapas de video tienen escala aproximada (velocidad de caminata asumida); con la IMU los nuevos videos/recorridos tendrán traslación real por conteo de pasos.
3. `phi_goal`/`d_goal` de rutas globales (módulo MAPS/waypoints) aún no existen; el agente navega a metas locales (landmark o punto). La interfaz `set_route()` ya está lista.
4. El peor caso actual es el pasillo largo real (90 %): más videos de pasillos (2–3 pasadas por zona) lo subirían.

---

## 6. Reentrenamiento con mapas enriquecidos (2026-10-26)

Tras ampliar visión (puertas ADE20K + detector Open Images + fix de
aspecto), los 10 mapas reales se regeneraron mucho más ricos (75
landmarks de puerta, escaleras como peligro, más obstáculos). Se
intentó reentrenar con ellos dos veces (6M pasos seed 3 y 8M pasos
seed 4, 45-50 % de episodios reales) y **ninguna superó al modelo
vigente evaluado sobre los mismos mapas nuevos**:

| Modelo | Global | Peor mapa (v09 oficina densa) |
|---|---|---|
| **nav_ppo vigente** | **95.4 %** | **77.5 %** |
| intento v3 (6M, s3) | 94.2 % | 60.0 % |
| intento v4 (8M, s4) | 95.0 % | 60.0 % |

Conclusión: se mantiene `models/nav_ppo`. El caso débil (v09: pasillo
estrecho con 12 puertas + muebles muy denso) se arregla mejor con
mapas más limpios (recorrer esa zona más veces/lento para ensanchar el
corredor libre) que con más entrenamiento.

## 7. Nuevo en esta fase (consumidores del modelo)

- `navigation/route_planner.py`: simula la política sobre el mapa
  actual y entrega la RUTA completa inicio→fin (~15 ms por plan); la
  GUI de `main_nav_lap.py` la dibuja (cian = llega, rojo = bloqueada)
  y se replanifica sola cuando un obstáculo nuevo bloquea el camino
  (validado: 15 pasos directos → 22 pasos con 0.9 m de desvío al caer
  una caja). Demo en `--sim`: tecla `b` (caja adelante), `m` (persona
  cruzando).
- `alertas.py`: módulo de alertas del Alg. 7, SIEMPRE activo (haya
  ruta o no), con avisos específicos (persona en movimiento cerca,
  hoyo/escalera/agua adelante en pasos), dosificación por tipo y
  apagado/encendido: tecla `x` en lap, doble click de botón GPIO en la
  Pi (gancho listo).

---

## 9. Sala de maestros + campaña de 19 mapas (2026-10-31)

Caso reportado: en la sala de maestros de la uni el sistema casi no
detectaba (sobre todo la entrada). Causas y arreglos (medidos sobre los
videos del lugar):
1. `main_nav_lap` en modo cámara no corría el segmentador (puertas
   ADE20K): ya va encendido por defecto (`--sin-hazards` lo apaga). La
   entrada de vidrio esmerilado SÍ la detecta ADE20K ("door cerrada").
2. Muebles de oficina fuera del filtro del detector extra: se agregaron
   Desk/Cabinetry/Filing cabinet/Countertop/Whiteboard/Wardrobe →
   `desk`, `cabinet`, `counter`, `whiteboard` (umbral extra 0.30→0.25).
3. Las mamparas azules no existen en ningún modelo preentrenado:
   limitación conocida; las cubren los MB1232 o un fine-tune propio.

Además: **llegada por percepción** — si la cámara ve el objeto destino
a <1 m al frente, la navegación se detiene aunque la pose derive (clave
en modo solo-cámara).

Campaña (los 16 mapas previos fueron borrados externamente y se
regeneraron desde los videos con la visión mejorada + v17-v19 de la
sala; 19 mapas en total):

| Corrida | Global | Prom. reales | Peor mapa |
|---|---|---|---|
| **nav_ppo vigente** | **96.9 %** | **96.6 %** | **82.5 %** |
| s8 (8M, prob .50) | 96.2 % | 96.4 % | 67.5 % |
| s9 (8M, prob .45) | 96.5 % | 96.2 % | 70.0 % |

**Se mantiene el modelo vigente** (tercera vez que los reintentos
colapsan en v09 mientras el vigente aguanta — el patrón es consistente:
más entrenamiento no arregla ese pasillo denso, regrabarlo sí). La sala
de maestros: 100 % en sus 3 mapas.
