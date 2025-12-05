import os
import sys
import time
from PIL import Image

import gradio as gr

# Garantir que o diretório `src` esteja no path para imports locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vision_model_clip import CLIPAIModel
from models.multimodal_model_llava import LLaVAModel


# Diretórios
os.makedirs("images", exist_ok=True)
os.makedirs("outputs/heatmaps", exist_ok=True)


# Instâncias de modelo (lazy loading)
clip_model = None
llava_model = None


def get_clip():
    """Carrega o modelo CLIP sob demanda."""
    global clip_model
    if clip_model is None:
        print("🔄 Inicializando CLIP...")
        clip_model = CLIPAIModel()
    return clip_model


def get_llava():
    """Carrega o modelo LLaVA sob demanda."""
    global llava_model
    if llava_model is None:
        print("🔄 Inicializando LLaVA via Ollama...")
        llava_model = LLaVAModel()
    return llava_model


def save_uploaded_image(img):
    """Salva imagem enviada no disco."""
    ts = int(time.time() * 1000)
    out_path = os.path.join("images", f"uploaded_{ts}.png")
    if isinstance(img, Image.Image):
        img.save(out_path)
    else:
        Image.fromarray(img).save(out_path)
    return out_path


def analyze_image(image):
    """Analisa imagem com CLIP e gera heatmap."""
    if image is None:
        return None, "❌ Erro", "Nenhuma imagem enviada", "", None, None
    
    try:
        # Salvar imagem
        img_path = save_uploaded_image(image)
        print(f"✅ Imagem salva em: {img_path}")
        
        # Executar CLIP
        clip = get_clip()
        print(f"📊 Analisando com CLIP...")
        result = clip.predict_with_heatmap(img_path)
        
        # Extrair resultados
        label = result.get("label", "N/A")
        prob = result.get("probability", 0.0)
        conceitos = result.get("conceitos", {})  # Se disponível
        overlay_path = result.get("overlay_path", None)
        
        # Carregar overlay/heatmap para exibição
        overlay_img = None
        if overlay_path and os.path.exists(overlay_path):
            overlay_img = Image.open(overlay_path).convert("RGB")
        
        # Formatar conceitos para exibição
        conceitos_text = ""
        if conceitos:
            conceitos_text = "\n".join([f"• {k}: {v:.1%}" for k, v in list(conceitos.items())[:5]])
        else:
            conceitos_text = "Nenhum defeito específico detectado"
        
        status_msg = f"✅ Análise CLIP concluída\n🏷️ {label}\n📈 Confiança: {prob:.2%}"
        
        return img_path, label, f"{prob:.2%}", conceitos_text, overlay_img, status_msg

    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return None, "❌ Erro", str(e), "", None, f"❌ Erro: {str(e)}"


def explain_with_llava(image_path, overlay_path, clip_label, clip_prob_str, conceitos_text):
    """Gera explicação com LLaVA baseada na análise CLIP."""
    try:
        if not image_path or not overlay_path:
            return "❌ Erro: Imagem ou overlay não disponível. Execute a análise CLIP primeiro."
        
        if not os.path.exists(image_path) or not os.path.exists(overlay_path):
            return "❌ Erro: Arquivos de imagem ou overlay não encontrados."
        
        # Limpar a probabilidade (remover %)
        prob_clean = clip_prob_str.replace("%", "").strip()
        try:
            prob_float = float(prob_clean) / 100.0
        except:
            prob_float = 0.0
        
        # Parsing de conceitos
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
                    except:
                        pass
        
        print("🧠 Chamando LLaVA para explicação...")
        llava = get_llava()
        response = llava.analisar_imagens(
            imagem_original=image_path,
            heatmap=overlay_path,
            classificacao_clip=clip_label,
            probabilidade_clip=prob_float,
            conceitos_detectados=conceitos_dict if conceitos_dict else None
        )
        
        if response is None:
            return "⚠️ Nenhuma resposta do LLaVA. Verifique se o Ollama está rodando."
        
        return response

    except Exception as e:
        print(f"❌ Erro ao chamar LLaVA: {e}")
        return f"❌ Erro ao gerar explicação: {str(e)}"


def build_ui():
    """Constrói a interface Gradio."""
    with gr.Blocks(title="MegaTruth — Detecção de IA em Imagens") as demo:
        
        # ========== CABEÇALHO ==========
        gr.Markdown("""
        # MegaTruth — Detecção de Imagens Geradas por IA
        
        **Upload uma imagem** → **Análise visual (CLIP)** → **Explicação em português (LLaVA)**
        
        Este sistema detecta se uma imagem é uma fotografia real ou gerada por IA, com heatmaps explicativos.
        """)
        
        # ========== SEÇÃO DE ENTRADA ==========
        gr.Markdown("### 1. Upload de Imagem")
        
        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="Envie sua imagem",
                sources=["upload", "clipboard"]
            )
            with gr.Column():
                analyze_btn = gr.Button(
                    "Analisar com CLIP",
                    size="lg",
                    variant="primary"
                )
                status_display = gr.Markdown("⏳ Aguardando...")
        
        # ========== SEÇÃO DE RESULTADOS CLIP ==========
        gr.Markdown("### 2. Resultados da Análise Visual (CLIP)")
        
        with gr.Row():
            col1, col2 = gr.Column(), gr.Column()
            
            with col1:                
                heatmap_display = gr.Image(
                    label="Heatmap de Ativação",
                    interactive=False
                )
            
            with col2:
                label_display = gr.Textbox(
                    label="Classificação",
                    interactive=False,
                    lines=1
                )
            
                prob_display = gr.Textbox(
                    label="📈 Confiança",
                    interactive=False,
                    lines=1
                )
                
                conceitos_display = gr.Textbox(
                    label="Padrões Detectados (Concept Analysis)",
                    interactive=False,
                    lines=4
                )
         
        # ========== SEÇÃO DE EXPLICAÇÃO ==========
        gr.Markdown("### 3. Explicação Detalhada (LLaVA)")
        
        explain_btn = gr.Button(
            "Gerar Explicação com LLaVA",
            size="lg",
            variant="secondary"
        )
        
        explanation_display = gr.Textbox(
            label="Análise Forense Detalhada",
            lines=12,
            interactive=False
        )
        
        # ========== LÓGICA DE EVENTOS ==========
        
        # Estado interno para rastrear valores
        state_image_path = gr.State(value=None)
        state_overlay_path = gr.State(value=None)
        state_label = gr.State(value="")
        state_prob = gr.State(value="")
        state_conceitos = gr.State(value="")
        
        def on_analyze(image):
            """Callback do botão 'Analisar'."""
            img_path, label, prob, conceitos, overlay, status = analyze_image(image)
            
            # Armazenar overlay_path: como overlay é PIL.Image, precisamos salvar
            overlay_path = None
            if overlay is not None:
                overlay_path = save_uploaded_image(overlay)
            
            # Retornar na ordem exata dos outputs
            return img_path, overlay_path, label, prob, conceitos, overlay, status, label, prob, conceitos, "⏳ Aguardando explicação..."
        
        # Conectar botão de análise
        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input],
            outputs=[
                state_image_path, state_overlay_path, state_label, state_prob, state_conceitos,
                heatmap_display, status_display, label_display, prob_display, conceitos_display, explanation_display
            ]
        )
        
        def on_explain(img_path, overlay_path, label, prob, conceitos):
            """Callback do botão 'Gerar Explicação'."""
            if not img_path or not overlay_path:
                return "Erro: Execute a análise CLIP primeiro."
            
            explanation = explain_with_llava(img_path, overlay_path, label, prob, conceitos)
            return explanation
        
        # Conectar botão de explicação
        explain_btn.click(
            fn=on_explain,
            inputs=[state_image_path, state_overlay_path, state_label, state_prob, state_conceitos],
            outputs=[explanation_display]
        )
    
    return demo


if __name__ == "__main__":
    app = build_ui()
    print("\n" + "="*60)
    print("Iniciando MegaTruth (Gradio)")
    print("="*60)
    print("Acesse em: http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
