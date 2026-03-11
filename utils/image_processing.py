from PIL import Image
import numpy as np

def resize_and_pad(img, target_size=(256, 256), fill_color=(255, 255, 255)):
    """Redimensiona la imagen manteniendo relación de aspecto y rellenando si es necesario"""
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    # Calcular nueva dimensión manteniendo relación de aspecto
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Redimensionar
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Crear nueva imagen con fondo blanco
    new_img = Image.new('RGB', target_size, fill_color)
    
    # Pegar la imagen redimensionada centrada
    new_img.paste(img, ((target_width - new_width) // 2, 
                        (target_height - new_height) // 2))
    
    return new_img