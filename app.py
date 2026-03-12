from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from detector import PlantDiseaseDetector
import os
import uuid
from pathlib import Path
import logging
import datetime
from db import get_db_connection

import mysql.connector
from mysql.connector import Error

# ─────────────────────────────────────────────
#  Configuración básica
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Directorios
BASE_DIR        = Path(__file__).parent.absolute()
UPLOAD_FOLDER   = BASE_DIR / "uploads"
CLASES_FOLDER   = BASE_DIR / "clases"          # FIX #1 – carpeta de imágenes de plagas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Configuración
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

# Crear directorio de uploads si no existe
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Modelo  (inicializar SOLO UNA VEZ)
# ─────────────────────────────────────────────
detector = PlantDiseaseDetector(
    model_path=BASE_DIR,
    model_version='plantdoc_300_epochs3'
)

# ─────────────────────────────────────────────
#  Base de datos
# ─────────────────────────────────────────────
DB_CONFIG = {
    'host':     '127.0.0.1',
    'port':     3306,
    'database': 'LOTUS',
    'user':     'root',
    'password': ''
}

def get_db_connection():
    """Devuelve una conexión MySQL o None si falla."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        logger.error(f"Error conectando a MySQL: {e}")
        return None

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template("index.html")


# FIX #1 ── Servir imágenes de la carpeta clases/
@app.route('/clases/<path:filename>')
def serve_clases_image(filename):
    """
    Sirve las imágenes de plagas almacenadas en /clases.
    Uso desde el frontend:  <img src="/clases/nombre_imagen.jpg">
    """
    return send_from_directory(CLASES_FOLDER, filename)


@app.route('/predict', methods=['POST'])
def predict():
    # FIX #2 ── Validar que el campo 'image' exista en el form-data
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No se proporcionó imagen'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Nombre de archivo vacío'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status':        'error',
            'message':       'Tipo de archivo no permitido',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400

    filepath = None   # FIX #3 ── definir antes del try para que el except pueda limpiar
    try:
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename  = f"{uuid.uuid4()}.{file_ext}"
        filepath  = UPLOAD_FOLDER / filename
        file.save(filepath)
        logger.info(f"Imagen guardada temporalmente: {filepath}")

        results = detector.detect(str(filepath))

        # FIX #4 ── eliminar archivo en bloque finally (garantiza limpieza siempre)
        response = {
            'status': results.get('status', 'success'),
            'data': {
                'detections':  results.get('detections', []),
                'image_info': {
                    'original_size':  results.get('image_size'),
                    'processed_size': (640, 640)
                },
                'model_info': results.get('model_info', {})
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        return jsonify({
            'status':  'error',
            'message': 'Error procesando imagen',
            'details': str(e)
        }), 500

    finally:
        # FIX #4 ── limpieza siempre, incluso si hay excepción
        if filepath and filepath.exists():
            filepath.unlink()


@app.route('/model-info', methods=['GET'])
def model_info():
    """Información del modelo cargado."""
    return jsonify({
        'status': 'success',
        'model': {
            'name':    'YOLOv7',
            'version': 'plantdoc_300_epochs3',   # FIX #5 ── typo 'pplantdoc' corregido
            'classes': detector.class_names,
            'device':  detector.device
        }
    })


@app.route('/disease-info', methods=['GET'])
def disease_info():
    """
    Devuelve info de una enfermedad/plaga desde la BD.
    La URL de la imagen apunta al endpoint /clases/<filename>
    para que el frontend pueda mostrarla directamente.
    """
    class_name = request.args.get('class', '').strip()

    # FIX #6 ── validar parámetro vacío o ausente
    if not class_name:
        return jsonify({'status': 'error', 'message': 'Parámetro "class" requerido'}), 400

    connection = get_db_connection()
    if not connection:
        return jsonify({'status': 'error', 'message': 'Error de conexión a la base de datos'}), 500

    try:
        cursor = connection.cursor(dictionary=True)
        # FIX #7 ── SELECT explícito en lugar de SELECT * (más robusto ante cambios de esquema)
        query = """
            SELECT clases, clases_e, descripcion, solucion, foto
            FROM plantas
            WHERE clases = %s
            LIMIT 1
        """
        cursor.execute(query, (class_name,))
        result = cursor.fetchone()

        if not result:
            return jsonify({'status': 'error', 'message': 'Enfermedad no encontrada'}), 404

        # FIX #8 ── construir URL relativa usando el endpoint /clases/
        foto_filename = os.path.basename(result['foto'])
        foto_url      = f"/clases/{foto_filename}"

        return jsonify({
            'status': 'success',
            'data': {
                'clases':      result['clases'],
                'clases_e':    result['clases_e'],
                'descripcion': result['descripcion'],
                'solucion':    result['solucion'],
                'foto':        foto_url           # URL lista para usar en <img src="...">
            }
        })

    except Error as e:
        logger.error(f"Error en /disease-info: {e}")
        return jsonify({'status': 'error', 'message': f'Error de base de datos: {e}'}), 500

    finally:
        # FIX #9 ── cerrar cursor y conexión en finally (antes podía quedar abierto si había error)
        try:
            cursor.close()
        except Exception:
            pass
        connection.close()


# ─────────────────────────────────────────────
#  Manejo de errores globales
# ─────────────────────────────────────────────
# FIX #10 ── handlers para errores HTTP comunes
@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Ruta no encontrada'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'status': 'error', 'message': 'Método no permitido'}), 405

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({'status': 'error', 'message': 'Imagen demasiado grande (máx. 16 MB)'}), 413


# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
