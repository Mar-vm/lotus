import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Añadir yolov7 al path
sys.path.insert(0, str(Path(__file__).parent / 'yolov7'))

from models.experimental import attempt_load
from utils.general import non_max_suppression

class PlantDiseaseDetector:
    def __init__(self, model_path, model_version='plantdoc_300_epochs3'):
        """
        Inicializa el detector YOLOv7
        
        Args:
            model_path (str): Ruta al directorio base que contiene modelo/
            model_version (str): Versión del modelo a usar
        """
        self.model_version = model_version
        model_weights = Path(model_path) / 'modelo' / model_version / 'weights' / 'best.pt'
        
        # Cargar modelo
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = attempt_load(model_weights, map_location=self.device)
        self.model.eval()
        
        # Clases (deben coincidir exactamente con el entrenamiento)
        self.class_names = ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot', 'Bell_pepper leaf', 'Blueberry leaf',
            'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf late blight', 'Potato leaf',
            'Raspberry leaf', 'Soyabean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot',
            'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus','Tomato leaf',
            'Tomato mold leaf', 'grape leaf black rot'
        ]
    
    def detect(self, image_path, conf_threshold=0.15):
        """
        Realiza detección en una imagen
        
        Args:
            image_path (str): Ruta a la imagen
            conf_threshold (float): Umbral de confianza (0-1)
            
        Returns:
            dict: Resultados de la detección
        """
        try:
            # Leer y preprocesar imagen
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen en {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)/255.0
            img_tensor = img_tensor.to(self.device)
            
            # Inferencia
            with torch.no_grad():
                pred = self.model(img_tensor)[0]
                pred = non_max_suppression(pred, conf_threshold, 0.45)  # conf, iou
            
            # Procesar resultados
            detections = []
            if pred[0] is not None:
                # Calcular factores de escala
                gain = min(img_tensor.shape[2]/img.shape[0], img_tensor.shape[3]/img.shape[1])
                pad = (img_tensor.shape[3] - img.shape[1] * gain) / 2, (img_tensor.shape[2] - img.shape[0] * gain) / 2
                
                for det in pred[0]:
                    *xyxy, conf, cls = det
                    # Convertir coordenadas a la imagen original
                    xyxy = [
                        int((xyxy[0] - pad[0]) / gain),
                        int((xyxy[1] - pad[1]) / gain),
                        int((xyxy[2] - pad[0]) / gain),
                        int((xyxy[3] - pad[1]) / gain)
                    ]
                    
                    detections.append({
                        'class': self.class_names[int(cls)],
                        'confidence': float(conf),
                        'bbox': xyxy,
                        'model_version': self.model_version
                    })
            
            return {
                'status': 'success',
                'detections': detections,
                'image_size': img.shape[:2],
                'model_info': {
                    'name': 'YOLOv7',
                    'version': self.model_version,
                    'classes': self.class_names
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }