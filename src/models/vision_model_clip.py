import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

class CLIPAIModel:
    def __init__(self, model_path=None, device=None):
        
        #1. Define o caminho relativo padrão onde o modelo deveria estar
        base_dir = os.path.dirname(os.path.abspath(__file__)) # Pega a pasta onde este arquivo .py está (src/models)
        default_path = os.path.join(base_dir, "clip_finetuned")
        
        # 2. Lógica de Seleção do Modelo
        if os.path.exists(default_path) and model_path != "openai/clip-vit-base-patch16":
            self.model_name = default_path
            print(f"Usando modelo Fine-Tuned (Artifact): {self.model_name}")
        else:
            self.model_name = "openai/clip-vit-base-patch16"
            print("AVISO: Modelo Fine-Tuned não encontrado. Usando modelo base da OpenAI.")
            print(f"   (Esperava encontrar em: {default_path})")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Carregando CLIP de: {self.model_name}")
        print(f"Dispositivo: {self.device}")

        if self.device == "cuda":
            torch.cuda.current_device()

        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        except Exception as e:
            print(f"Erro crítico ao carregar modelo: {e}")
            raise e

        self.model.eval()
        self.classes = ["a real photograph", "an AI-generated image"]

    def predict_with_heatmap(self, image_path):
        """
        Classifica a imagem, gera um heatmap intuitivo (tratado) e analisa conceitos.
        """
        # Garante que a pasta de saída existe
        os.makedirs("outputs/heatmaps", exist_ok=True)

        # ---- 1) Carregar imagem ----
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # ---- 2) Processar textos e imagem ----
        text_inputs = self.processor(
            text=self.classes,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # ---- 4) Forward Pass (Classificação) ----
        with torch.no_grad():
            outputs = self.model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values,
                return_loss=False
            )
            logits = outputs.logits_per_image

        # ---- 5) Resultados da Classificação ----
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        prediction_idx = int(np.argmax(probs))
        prediction_label = self.classes[prediction_idx]
        prediction_prob = float(probs[prediction_idx])

        # ---- 6) Gerar GradCAM (Raw Heatmap) ----
        # Habilita gradientes para a imagem
        image_inputs.pixel_values.requires_grad_(True)
        
        # Recalcula embeddings com gradiente ativado
        image_embeds = self.model.get_image_features(pixel_values=image_inputs.pixel_values)
        text_embeds = self.model.get_text_features(
            input_ids=text_inputs.input_ids, 
            attention_mask=text_inputs.attention_mask
        )

        # Calcula score para a classe vencedora e faz backprop
        score = torch.matmul(image_embeds, text_embeds[prediction_idx].unsqueeze(1)).squeeze()
        self.model.zero_grad()
        score.backward()

        # Extrai o gradiente e tira a média dos canais
        grads = image_inputs.pixel_values.grad[0].detach().cpu().numpy()
        heatmap_raw = np.mean(grads, axis=0)

        # ---- 7) PÓS-PROCESSAMENTO "VISÃO HUMANA" (A Mágica) ----
        
        # A. Redimensionar para o tamanho original da imagem
        heatmap = cv2.resize(heatmap_raw, (w, h))
        
        # B. Normalizar (0 a 1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        
        # C. Limpar ruído de fundo (Threshold)
        # Ignora pixels com menos de 20% de importância
        heatmap[heatmap < 0.2] = 0
        
        # D. "Engordar" as áreas (Dilation)
        # Conecta pontinhos isolados para formar uma região coerente
        kernel_size = int(min(w, h) * 0.03) # 3% do tamanho da imagem
        if kernel_size < 3: kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        heatmap_dilated = cv2.dilate(heatmap, kernel, iterations=2)
        
        # E. Suavizar (Gaussian Blur)
        # Deixa a mancha com bordas macias (estilo nuvem/calor)
        blur_size = int(min(w, h) * 0.08) # 8% da imagem
        if blur_size % 2 == 0: blur_size += 1 # Tem que ser ímpar
        heatmap_final = cv2.GaussianBlur(heatmap_dilated, (blur_size, blur_size), 0)
        
        # Re-normaliza após o blur para garantir brilho máximo no centro
        heatmap_final /= (np.max(heatmap_final) + 1e-8)

        # ---- 8) Gerar Overlay "Red Alert" ----
        img_np = np.array(image)
        
        # Cria uma máscara Vermelha Sólida
        red_mask = np.zeros_like(img_np)
        red_mask[:, :, 0] = 255 # Canal R (Vermelho)
        
        # Converte para float para misturar
        img_float = img_np.astype(np.float32) / 255.0
        mask_float = red_mask.astype(np.float32) / 255.0
        
        # O heatmap tratado vira o canal Alpha (Transparência)
        alpha = heatmap_final[:, :, None]
        
        # Fórmula de Blending: 
        # Onde alpha é alto -> Mostra mais vermelho e escurece um pouco a original
        # Onde alpha é zero -> Mostra a original intacta
        overlay_float = (mask_float * alpha * 0.8) + (img_float * (1.0 - (alpha * 0.5)))
        
        # Converte de volta para imagem (0-255)
        overlay = np.clip(overlay_float * 255, 0, 255).astype(np.uint8)
        
        # Converte para BGR (padrão OpenCV) para salvar
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Também gera o JET clássico suavizado (opcional, bom para debug)
        heatmap_jet = cv2.applyColorMap(np.uint8(255 * heatmap_final), cv2.COLORMAP_JET)

        # ---- 9) Salvar Arquivos ----
        base = os.path.basename(image_path)
        
        # Salvamos o Overlay Limpo como padrão
        overlay_path = f"outputs/heatmaps/{base}_overlay.png"
        heatmap_path = f"outputs/heatmaps/{base}_heatmap_raw.png"

        cv2.imwrite(overlay_path, overlay_bgr)
        cv2.imwrite(heatmap_path, heatmap_jet)

        # ---- 10) Analisar Conceitos (Semântica) ----
        # Passa a classificação prevista para ativar o Gating (Filtro Inteligente)
        conceitos_detectados = self.analisar_conceitos(image_path, classificacao_preliminar=prediction_label)

        return {
            "label": prediction_label,
            "probability": prediction_prob,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "conceitos": conceitos_detectados
        }


    def analisar_conceitos(self, image_path, classificacao_preliminar=None):
        """
        Testa a imagem contra uma biblioteca vasta de 'artefatos de IA'.
        Retorna os Top conceitos que mais 'acenderam' na imagem.
        """
        
        # Lista expandida de defeitos comuns em IA (Concept Bottleneck)
        conceitos = [
            # === ANATOMIA & BIOLOGIA  ===
            "deformed hands and fingers", "extra fingers", "missing fingers", "fused fingers",
            "malformed fingernails", "hand blending into object", "impossible finger joints",
            "asymmetric eyes", "misshaped pupils", "strabismus cross-eyed", "heterochromia eyes",
            "teeth blending together", "too many teeth", "gum anomalies",
            "hair blending into clothes", "floating hair strands", "hair defying gravity",
            "unnatural waxy skin", "plastic doll skin", "no skin pores", "oversmoothed facial features",
            "extra limbs", "anatomically impossible pose", "twisted limbs", "neck too long",

            # === FÍSICA DA LUZ E MATERIAIS  ===
            "incorrect light reflection", "missing reflection in mirror", "reflection showing wrong object",
            "inconsistent shadows", "shadows pointing wrong way", "missing cast shadows",
            "light source conflict", "unmotivated lighting",
            "subsurface scattering artifacts", "skin looking like wax", # A luz não entra na pele corretamente
            "glass refraction error", "liquid physics error", "water defying gravity",
            "metallic texture looking plastic", "cloth texture blending into skin",

            # === LÓGICA CONTEXTUAL E OBJETOS ===
            "car with 5 wheels", "bicycle with missing parts", "distorted vehicle wheels",
            "chair with extra legs", "table legs blending", "floating furniture",
            "holding object incorrectly", "object merging with hand", "levitating objects",
            "impossible clothing folds", "zippers leading nowhere", "buttons inconsistent",
            "glasses blending into skin", "asymmetric glasses frames",
            "jewelry melting into skin", "watch face gibberish",

            # === ARQUITETURA E PERSPECTIVA ===
            "impossible architecture", "stairs leading nowhere", "mismatched windows",
            "curved pillars", "asymmetrical building structure", "rooflines not matching",
            "warped straight lines", "non-euclidean geometry", "tilted horizon",
            "floor texture tiling error", "vanishing point mismatch",

            # === NATUREZA E PADRÕES ===
            "animal with extra legs", "animal with missing legs", "morphed animal faces",
            "leaves blending together", "repetitive texture tiling", "identical clouds",
            "flowers merging", "tree branches ending abruptly", "floating rocks",
            
            # === ARTEFATOS DIGITAIS PUROS ===
            "oversaturated hdr colors", "excessive contrast", "unnatural bokeh blur",
            "high frequency noise artifacts", "jpeg compression artifacts", "grid pattern noise",
            "oil painting filter effect", "smudged textures", "chromatic aberration abuse",
            "gibberish text", "alien hieroglyphs", "illegible signboard", "morphed logos"
        ]
        
        
        conceitos_completos = conceitos + ["a high quality natural photograph"]

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Se foi classificado como REAL (0), sobe a régua para 0.25 (só aceita defeito óbvio)
            # Se foi classificado como FAKE (1), mantém régua baixa 0.10 (aceita pistas sutis)
            if classificacao_preliminar == "a real photograph" or classificacao_preliminar == 0:
                threshold = 0.25 
            else:
                threshold = 0.10
                
            inputs = self.processor(
                text=conceitos_completos,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image 
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            resultado = {}
        
            for i in range(len(conceitos)):
                # Usa o threshold dinâmico
                    if probs[i] > threshold: 
                        resultado[conceitos[i]] = float(probs[i])
            
            
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            print(f"Erro na análise de conceitos: {e}")
            return {}