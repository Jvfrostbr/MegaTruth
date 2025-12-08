import os
import sys
import time
from PIL import Image
from dotenv import load_dotenv

# --- CORREÇÃO DO ERRO DE SSL (WINDOWS) ---
if 'SSL_CERT_FILE' in os.environ:
    if not os.path.exists(os.environ['SSL_CERT_FILE']):
        del os.environ['SSL_CERT_FILE']
# -----------------------------------------

import gradio as gr

# Garantir que o diretório `src` esteja no path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vision_model_clip import CLIPAIModel
from models.multimodal_model_llava import LLaVAModel
from models.multimodal_model_nemotron import NemotronVL

# Carrega .env
load_dotenv()

# Diretórios
os.makedirs("images/uploaded", exist_ok=True)
os.makedirs("outputs/defect_maps", exist_ok=True)

# Instâncias globais
clip_model = None
llava_model = None
nemotron_model = None

def get_clip():
    global clip_model
    if clip_model is None:
        print("🔄 Inicializando CLIP...")
        clip_model = CLIPAIModel()
    return clip_model

def get_llava():
    global llava_model
    if llava_model is None:
        print("🔄 Inicializando LLaVA via Ollama...")
        llava_model = LLaVAModel()
    return llava_model

def get_nemotron():
    global nemotron_model
    if nemotron_model is None:
        print("🔄 Inicializando Cliente Nemotron...")
        nemotron_model = NemotronVL()
    return nemotron_model

def save_uploaded_image(img):
    """Salva imagem enviada no disco dentro da pasta 'images/uploaded'."""
    ts = int(time.time() * 1000)
    out_path = os.path.join("images", "uploaded", f"uploaded_{ts}.png")
    
    if isinstance(img, Image.Image):
        img.save(out_path)
    else:
        Image.fromarray(img).save(out_path)
        
    return out_path

# --- ATUALIZAÇÃO 1: Recebe a cor como parâmetro ---
def analyze_image(image, overlay_color):
    """Analisa imagem com CLIP e gera defect_map na cor escolhida."""
    if image is None:
        return None, "Erro", "Nenhuma imagem enviada", "", None, None
    
    try:
        img_path = save_uploaded_image(image)
        print(f"✅ Imagem salva em: {img_path}")
        
        # Mapeia nome amigável para código interno ('red', 'green', 'blue')
        color_map = {
            "🔴 Vermelho (Padrão)": "red",
            "🟢 Verde (Para fundos avermelhados)": "green",
            "🔵 Azul (Para fundos quentes)": "blue"
        }
        selected_code = color_map.get(overlay_color, "red")

        clip = get_clip()
        print(f" Analisando com CLIP (Overlay: {selected_code})...")
        
        # Passa a cor para o modelo
        result = clip.predict_with_defect_map(img_path, overlay_color=selected_code)
        
        label = result.get("label", "N/A")
        prob = result.get("probability", 0.0)
        conceitos = result.get("conceitos", {}) 
        overlay_path = result.get("overlay_path", None)
        
        conceitos_text = ""
        if conceitos:
            conceitos_text = "\n".join([f"• {k}: {v:.1%}" for k, v in list(conceitos.items())[:5]])
        else:
            conceitos_text = "Nenhum defeito específico detectado"
        
        status_msg = f"Análise CLIP concluída\n {label}\n Confiança: {prob:.2%}"
        
        return img_path, label, f"{prob:.2%}", conceitos_text, overlay_path, status_msg

    except Exception as e:
        print(f"Erro na análise: {e}")
        return None, "Erro", str(e), "", None, f"Erro: {str(e)}"


def explain_with_multimodal(image_path, overlay_path, clip_label, clip_prob_str, conceitos_text, overlay_color):
    """
        Gera explicação usando estratégia Híbrida:
            1. Tenta Nemotron (Melhor qualidade, API).
            2. Se falhar, usa LLaVA (Local, Fallback).
    """

    try:
        if not image_path or not overlay_path:
            return "Erro: Imagem ou overlay não disponível. Execute a análise CLIP primeiro."

        if not os.path.exists(image_path) or not os.path.exists(overlay_path):
            return "rro: Arquivos de imagem ou overlay não encontrados."

        # --- 1. Preparar Dados (Parsing) ---
        # Limpar a probabilidade (remover %)
        prob_clean = clip_prob_str.replace("%", "").strip()
        try:
            prob_float = float(prob_clean) / 100.0
        except:
            prob_float = 0.0

        # Parsing de conceitos (Texto -> Dict)
        conceitos_dict = {}
        if conceitos_text and "•" in conceitos_text:
            for line in conceitos_text.split("\n"):
                if "•" in line:
                    try:
                        parts = line.split("•")[1].split(":")
                        if len(parts) == 2:
                            k = parts[0].strip()
                            v = float(parts[1].strip().replace("%", "")) / 100.0
                            conceitos_dict[k] = v
                    except: pass
                    
        response_text = None
        model_used = ""
        
        
        # Mapear a cor selecionada para o formato esperado pelo Nemotron
        cor_real = "Vermelha" # Default
        if "Verde" in overlay_color:
            cor_real = "Verde"
        elif "Azul" in overlay_color:
            cor_real = "Azul"
        elif "Vermelho" in overlay_color:
            cor_real = "Vermelha"
    
        # --- 2. TENTATIVA A: NEMOTRON (API) ---
        try:

            print("🚀 Tentando Nemotron-12B...")
            nemotron = get_nemotron()
            
            response_text = nemotron.analisar_imagens(
                imagem_original=image_path,
                defect_map=overlay_path,
                classificacao_clip=clip_label,
                probabilidade_clip=prob_float,
                conceitos_detectados=conceitos_dict if conceitos_dict else None,
                color_overlay=cor_real
            )
            
            if response_text:
                model_used = "NVIDIA Nemotron-12B (Via API)"
        except Exception as e:
            print(f"⚠️ Nemotron falhou: {e}. Alternando para LLaVA...")

        # --- 3. TENTATIVA B: LLAVA (Local) ---
        if not response_text:
            try:
                print("🦙 Tentando LLaVA-7B (Local)...")
                llava = get_llava()

                response_text = llava.analisar_imagens(
                    imagem_original=image_path,
                    defect_map=overlay_path,
                    classificacao_clip=clip_label,
                    probabilidade_clip=prob_float,
                    conceitos_detectados=conceitos_dict if conceitos_dict else None, 
                    color_overlay=cor_real
                )
                if response_text:
                    model_used = "LLaVA-7B (Local Ollama)"

            except Exception as e:
                return f"rro Crítico: Ambos os modelos falharam.\nNemotron: (Vide logs)\nLLaVA: {str(e)}"

        # --- 4. Resultado Final ---
        if response_text:
            header = f"🤖 **Modelo Utilizado:** {model_used}\n" + "="*40 + "\n\n"
            return header + response_text

        else:
            return "Erro desconhecido: O modelo retornou uma resposta vazia."

    except Exception as e:
        print(f"Erro geral: {e}")
        return f"Erro ao gerar explicação: {str(e)}"


def build_ui():
    with gr.Blocks(title="MegaTruth — Detecção de IA em Imagens") as demo:

        custom_css = """
        :root {
            --primary-500: #00baff;
            --primary-600: #009ecf;
            --primary-700: #007fa8;
        }

        /* Botões primários */
        button.primary {
            background-color: #00baff !important;
            border-color: #00baff !important;
        }

        /* Botão hover */
        button.primary:hover {
            background-color: #009ecf !important;
            border-color: #009ecf !important;
        }

        /* Radio / checkbox selecionados */
        input[type="radio"]:checked {
            accent-color: #00baff;
        }

        /* Barra de progresso */
        progress::-webkit-progress-value {
            background-color: #00baff;
        }
        """

        gr.HTML(f"""
        <style>
        {custom_css}
        </style>
        """)

        # Cabeçalho com logo
        gr.Markdown("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px;">
            <img src="https://raw.githubusercontent.com/Jvfrostbr/MegaTruth/main/images/logo/logo_mega_truth.png" alt="MegaTruth Logo" style="width: 150px; height: auto;">
            <div>
                <h1 style="margin: 0; padding: 0;">MegaTruth</h1>
                <h3 style="margin: 5px 0; padding: 0; color: #666;">Sistema Forense de Detecção de IA</h3>
                <p style="margin: 10px 0; padding: 0; font-size: 0.95em;">
                    <strong>1. Análise Visual:</strong> O modelo <strong>CLIP (Fine-Tuned)</strong> detecta anomalias e gera um <em>defect_map</em>.<br>
                    <strong>2. Análise Semântica:</strong> O sistema verifica <strong>Conceitos Específicos</strong> (anatomia, física).<br>
                    <strong>3. Parecer Pericial:</strong> Uma <strong>IA Multimodal</strong> gera o laudo final.
                </p>
            </div>
        </div>
        """)
        
        gr.Markdown("---")

        # ========== SEÇÃO DE ENTRADA ==========
        gr.Markdown("### 1. Upload de Imagem")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Envie sua imagem",
                    sources=["upload", "clipboard"],
                    width=600,
                    height=400
                )
            
            with gr.Column(scale=1):
                # --- ATUALIZAÇÃO 2: Componente de Cor ---
                color_selector = gr.Radio(
                    choices=["🔴 Vermelho (Padrão)", "🟢 Verde (Para fundos avermelhados)", "🔵 Azul (Para fundos quentes)"],
                    value="🔴 Vermelho (Padrão)",
                    label="Cor do Overlay (defect_map)",
                    info="Escolha uma cor que contraste com a imagem original."
                )
                
                analyze_btn = gr.Button(
                    "Analisar Imagem",
                    size="lg",
                    variant="primary"
                )
                status_display = gr.Markdown("Aguardando envio...")
        
        # ========== SEÇÃO DE RESULTADOS CLIP ==========
        gr.Markdown("### 2. Resultados Técnicos (Detector)")
        
        with gr.Row():
            col1, col2 = gr.Column(), gr.Column()
            
            with col1:                
                defect_map_display = gr.Image(
                    label="Pontos de Interesse (defect_map)",
                    interactive=False,
                    width=600,
                    height=400
                )
            
            with col2:
                label_display = gr.Textbox(label="Classificação", interactive=False)
                prob_display = gr.Textbox(label="Grau de Certeza", interactive=False)
                conceitos_display = gr.Textbox(
                    label="Defeitos Específicos Detectados",
                    interactive=False,
                    lines=4
                )
         
        # ========== SEÇÃO DE EXPLICAÇÃO ==========
        gr.Markdown("### 3. Parecer Pericial (IA Generativa)")
        
        explain_btn = gr.Button(
            "Gerar Laudo Explicativo",
            size="lg",
            variant="secondary"
        )
        
        explanation_display = gr.Markdown(
            value="Aguardando análise..."
        )
        
        # ========== LÓGICA DE EVENTOS ==========
        
        state_image_path = gr.State(value=None)
        state_overlay_path = gr.State(value=None)
        state_label = gr.State(value="")
        state_prob = gr.State(value="")
        state_conceitos = gr.State(value="")
        
        # --- ATUALIZAÇÃO 3: Passa o valor da cor para a função ---
        def on_analyze(image, color):
            img_path, label, prob, conceitos, overlay_path, status = analyze_image(image, color)
            
            
            return img_path, overlay_path, label, prob, conceitos, overlay_path, status, label, prob, conceitos, "Clique em 'Gerar Laudo'..."
        
        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input, color_selector], # Adicionado o input de cor
            outputs=[
                state_image_path, state_overlay_path, state_label, state_prob, state_conceitos,
                defect_map_display, status_display, label_display, prob_display, conceitos_display, explanation_display
            ]
        )
        
        def on_explain(img_path, overlay_path, label, prob, conceitos, color):
            if not img_path or not overlay_path:
                return "Erro: Execute a análise visual primeiro."
            return explain_with_multimodal(img_path, overlay_path, label, prob, conceitos, overlay_color=color)
        
        explain_btn.click(
            fn=on_explain,
            inputs=[state_image_path, state_overlay_path, state_label, state_prob, state_conceitos, color_selector], # Adicionado o input de cor
            outputs=[explanation_display]
        )
    
    return demo

if __name__ == "__main__":
    app = build_ui()
    print("\n" + "="*60)
    print("🚀 MegaTruth Interface Iniciada")
    print("="*60)
    
    # Caminho do favicon
    favicon_path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "logo", "logo_mega_truth.png")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        favicon_path=favicon_path if os.path.exists(favicon_path) else None
    )