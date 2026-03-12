from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from detector import PlantDiseaseDetector
import os
import uuid
from pathlib import Path
import logging
import datetime
from db import get_db_connection

# Configuración básica
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

# Directorios
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Configuración
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Crear directorio de uploads si no existe
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Inicializar el modelo (¡SOLO UNA VEZ!)
detector = PlantDiseaseDetector(
    model_path=BASE_DIR,
    model_version='plantdoc_300_epochs3'
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No se proporcionó imagen'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Nombre de archivo vacío'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Tipo de archivo no permitido',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Guardar archivo temporal
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        logger.info(f"Imagen guardada temporalmente en: {filepath}")
        
        # Procesar imagen
        results = detector.detect(str(filepath))
        
        # Eliminar archivo temporal
        filepath.unlink()
        
        # Formatear respuesta
        response = {
            'status': results.get('status', 'success'),
            'data': {
                'detections': results.get('detections', []),
                'image_info': {
                    'original_size': results.get('image_size'),
                    'processed_size': (640, 640)
                },
                'model_info': results.get('model_info', {})
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en /predict: {str(e)}", exc_info=True)
        # Limpieza en caso de error
        if 'filepath' in locals() and filepath.exists():
            filepath.unlink()
        return jsonify({
            'status': 'error',
            'message': 'Error procesando imagen',
            'details': str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint para obtener información del modelo"""
    return jsonify({
        'status': 'success',
        'model': {
            'name': 'YOLOv7',
            'version': 'pplantdoc_300_epochs3',
            'classes': detector.class_names,
            'device': detector.device
        }
    })


@app.route('/disease-info', methods=['GET'])
def disease_info():
    class_name = request.args.get('class')
    
    if not class_name:
        return jsonify({'status': 'error', 'message': 'Parámetro class requerido'}), 400
    
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 'error', 'message': 'Error de conexión a la base de datos'}), 500
        
        cursor = connection.cursor()
        query = "SELECT * FROM plantas WHERE clases = %s"
        cursor.execute(query, (class_name,))
        result = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if result:
            # Extrae solo el nombre del archivo de la ruta completa
            filename = os.path.basename(result['foto'])
            
            return jsonify({
                'status': 'success',
                'data': {
                    'clases': result['clases'],
                    'clases_e': result['clases_e'],
                    'descripcion': result['descripcion'],
                    'solucion': result['solucion'],
                    'foto': filename
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Enfermedad no encontrada'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error de base de datos: {str(e)}'}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
