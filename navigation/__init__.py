"""
módulo de navegación (DRL).

dos partes: train.py crea el modelo (PPO en un sim gymnasium con mundos
procedurales) y navigator.py lo usa en tiempo real con lo que ya sale
del SLAM + visión. el train exporta el .zip de SB3 y un .npz de puros
pesos, pa que la raspberry corra inferencia solo con numpy (sin
SB3/torch).
"""
