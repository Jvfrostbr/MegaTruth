import os
import sys
import time
import base64
import json
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

# --- SSL CERTIFICATE FIX ---
if 'SSL_CERT_FILE' in os.environ:
    if not os.path.exists(os.environ['SSL_CERT_FILE']):
        del os.environ['SSL_CERT_FILE']

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.vision_model_clip import CLIPAIModel
from models.multimodal_model_llava import LLaVAModel
from models.multimodal_model_nemotron import NemotronVL

load_dotenv()

app = Flask(__name__)
CORS(app)

# Create directories
os.makedirs("images/uploaded", exist_ok=True)
os.makedirs("outputs/defect_maps", exist_ok=True)

# Global model instances
clip_model = None
llava_model = None
nemotron_model = None

def get_clip():
    global clip_model
    if clip_model is None:
        print("🔄 Initializing CLIP...")
        clip_model = CLIPAIModel()
    return clip_model

def get_llava():
    global llava_model
    if llava_model is None:
        print("🔄 Initializing LLaVA via Ollama...")
        llava_model = LLaVAModel()
    return llava_model

def get_nemotron():
    global nemotron_model
    if nemotron_model is None:
        print("🔄 Initializing Nemotron Client...")
        nemotron_model = NemotronVL()
    return nemotron_model

def save_uploaded_image(img_data):
    """Save uploaded image to disk and return path"""
    ts = int(time.time() * 1000)
    out_path = os.path.join("images", "uploaded", f"uploaded_{ts}.png")
    
    if isinstance(img_data, str) and img_data.startswith('data:image'):
        # Handle base64 image
        img_data = img_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(img_data)))
        image.save(out_path)
    else:
        # Handle PIL Image
        if hasattr(img_data, 'save'):
            img_data.save(out_path)
        else:
            Image.fromarray(img_data).save(out_path)
    
    return out_path

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'MegaTruth API is running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze image with CLIP and generate defect maps"""
    try:
        data = request.json
        image_data = data.get('image')
        overlay_color = data.get('overlay_color', 'red')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Save image
        img_path = save_uploaded_image(image_data)
        print(f"✅ Image saved at: {img_path}")
        
        # Map friendly color name to internal code
        color_map = {
            "Vermelho (Padrão)": "red",
            "Verde (Para fundos avermelhados)": "green",
            "Azul (Para fundos quentes)": "blue"
        }
        selected_code = color_map.get(overlay_color, "red")
        
        # Get CLIP model and analyze
        clip = get_clip()
        result = clip.predict_with_defect_map(img_path, overlay_color=selected_code)
        
        # Convert defect maps to base64
        defect_maps_base64 = []
        defect_maps = result.get("defect_maps", [])
        for defect_map in defect_maps:
            map_path = defect_map.get("defect_map_path")
            if map_path and os.path.exists(map_path):
                with open(map_path, "rb") as f:
                    map_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                defect_maps_base64.append({
                    'conceito': defect_map.get('conceito'),
                    'probabilidade': defect_map.get('probabilidade'),
                    'prompt': defect_map.get('prompt'),
                    'image_base64': map_base64
                })
        
        # Prepare response
        response = {
            'image_path': img_path,
            'label': result.get("label", "N/A"),
            'probability': result.get("probability", 0.0),
            'conceitos': result.get("conceitos", {}),
            'defect_maps': defect_maps_base64,
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/explain', methods=['POST'])
def generate_explanation():
    """Generate multimodal explanation using Nemotron or LLaVA"""
    try:
        data = request.json
        image_path = data.get('image_path')
        defect_maps = data.get('defect_maps', [])  # Lista de defect maps
        clip_label = data.get('clip_label')
        clip_probability = data.get('clip_probability', 0.0)
        conceitos = data.get('conceitos', {})
        overlay_color = data.get('overlay_color', 'Vermelha')
        prefer_nemotron = data.get('prefer_nemotron', True)
        resize_images = data.get('resize_images', True)
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 400
        
        # Use the first (highest probability) defect map for explanation
        overlay_path = None
        if defect_maps and len(defect_maps) > 0:
            first_defect = defect_maps[0]
            overlay_base64 = first_defect.get('image_base64')
            if overlay_base64:
                overlay_path = os.path.join("outputs", "defect_maps", f"temp_overlay_{int(time.time())}.png")
                overlay_data = base64.b64decode(overlay_base64)
                with open(overlay_path, "wb") as f:
                    f.write(overlay_data)
        
        # Map color for display
        cor_real = "Vermelha"
        if "Verde" in overlay_color:
            cor_real = "Verde"
        elif "Azul" in overlay_color:
            cor_real = "Azul"
        elif "Vermelho" in overlay_color:
            cor_real = "Vermelha"
        
        response_text = None
        model_used = ""
        
        # Try Nemotron first if preferred
        if prefer_nemotron:
            try:
                print("🚀 Trying Nemotron-12B...")
                nemotron = get_nemotron()
                
                response_text = nemotron.analisar_imagens(
                    imagem_original=image_path,
                    defect_map=overlay_path,
                    classificacao_clip=clip_label,
                    probabilidade_clip=clip_probability,
                    conceitos_detectados=conceitos if conceitos else None,
                    color_overlay=cor_real,
                    resize_images=resize_images,
                    max_side=2000,
                    quality=85
                )
                
                if response_text:
                    model_used = "NVIDIA Nemotron-12B (Via API)"
            except Exception as e:
                print(f"⚠️ Nemotron failed: {e}")
        
        # Fallback to LLaVA
        if not response_text:
            try:
                print("🦙 Trying LLaVA-7B (Local)...")
                llava = get_llava()
                
                response_text = llava.analisar_imagens(
                    imagem_original=image_path,
                    defect_map=overlay_path,
                    classificacao_clip=clip_label,
                    probabilidade_clip=clip_probability,
                    conceitos_detectados=conceitos if conceitos else None,
                    color_overlay=cor_real
                )
                
                if response_text:
                    model_used = "LLaVA-7B (Local Ollama)"
            except Exception as e:
                return jsonify({'error': f'Both models failed. LLaVA error: {str(e)}'}), 500
        
        # Clean up temporary overlay
        if overlay_path and os.path.exists(overlay_path):
            os.remove(overlay_path)
        
        if response_text:
            return jsonify({
                'explanation': response_text,
                'model_used': model_used,
                'defect_maps_count': len(defect_maps),
                'status': 'success'
            })
        else:
            return jsonify({'error': 'No explanation generated', 'status': 'error'}), 500
    
    except Exception as e:
        print(f"Error in explanation: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MegaTruth API Server Starting...")
    print("="*60)
    app.run(host='127.0.0.1', port=5000, debug=True)