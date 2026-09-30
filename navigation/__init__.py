"""
Módulo de NAVEGACIÓN (aprendizaje profundo por refuerzo).

Dos programas principales (según el diseño del TT, §6.4.3 / §7.3.1):

1. CREAR el modelo:   navigation/train.py
   Entrena un agente PPO (Stable-Baselines3) en un simulador Gymnasium
   (navigation/nav_env.py) con mundos generados proceduralmente
   (cuartos, multi-cuarto con puertas, exteriores con hoyos/coladeras,
   obstáculos dinámicos). Exporta el modelo en dos formatos: .zip de
   SB3 y .npz de pesos puros (inferencia con solo numpy, para la
   Raspberry sin depender de SB3/torch).

2. USAR el modelo:    navigation/navigator.py  (el runtime)
   `Navigator` consume el NavigationFrame que ya emiten los módulos de
   SLAM + mapeo + visión, recibe comandos de alto nivel (ir a un
   landmark/punto, pausa, reanudar, detener, ruta alternativa) y emite
   indicaciones una a una en tiempo real (el formato que luego
   consumirán los módulos de voz/PLN). Integrado en main_nav_lap.py
   (laptop, GUI) y main_nav_pi.py (Raspberry, headless).

El vector de estado, las acciones discretas (avanzar / girar izquierda /
girar derecha / detenerse) y la estructura de recompensa siguen la
documentación del proyecto (TT_2026_B045 §7.3.1).
"""
