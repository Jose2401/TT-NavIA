"""
Paquete de VISIÓN del sistema (antes vision/TT-NavIA como repo anidado;
ahora solo los componentes que el pipeline usa, como paquete normal):

- detector.py            YOLOv8-seg: bbox + máscara + clase + confianza
- obstacle_logic.py      Clasificación semántica, alias de etiquetas
- motion.py              Movimiento global y por objeto (tracker IoU)
- scene_segmentation.py  SegFormer ADE20K: barreras/piso + máscaras de
                         superficie transitable/agua/escaleras (peligros)
- yolov8s-seg.pt         Pesos YOLOv8-seg

El consumidor de este paquete es vision_bridge.py (raíz del proyecto),
que lo conecta con el SLAM y el mapa. El main original del equipo y los
componentes no usados por el pipeline (renderer, voice, finetune) se
retiraron de la rama; el respaldo del módulo completo vive en el repo
original del equipo de visión.
"""
