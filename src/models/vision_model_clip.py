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

        # --- Caminho do Modelo clip---
        if os.path.exists("./clip-finetuned-artifact"):
            self.model_name = "./clip-finetuned-artifact"
        else:
            self.model_name = "openai/clip-vit-base-patch16"
            print("AVISO: Usando modelo base da OpenAI (não treinado no Artifact).")

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
        Função para classificar a imagem e gerar o heatmap e o overlay.
        Retorna o label, a probabilidade e os conceitos que mais 'acenderam' na imagem.
        """

        os.makedirs("outputs/heatmaps", exist_ok=True)

        # ---- 1) Carregar imagem ----
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # ---- 2) Processar textos ----
        text_inputs = self.processor(
            text=self.classes,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # ---- 3) Processar imagem ----
        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # ---- 4) Forward normal (sem gradiente) ----
        with torch.no_grad():
            outputs = self.model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values,
                return_loss=False
            )
            logits = outputs.logits_per_image

        # ---- 5) Classificação ----
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        prediction_idx = int(np.argmax(probs))
        prediction_label = self.classes[prediction_idx]
        prediction_prob = float(probs[prediction_idx])

        # ---- 6) Gerar GradCAM ----
        image_inputs.pixel_values.requires_grad_(True)

        image_embeds = self.model.get_image_features(
            pixel_values=image_inputs.pixel_values
        )

        text_embeds = self.model.get_text_features(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )

        score = torch.matmul(
            image_embeds,
            text_embeds[prediction_idx].unsqueeze(1)
        ).squeeze()

        self.model.zero_grad()
        score.backward()

        grads = image_inputs.pixel_values.grad[0].detach().cpu().numpy()
        heatmap = np.mean(grads, axis=0)

        # ---- AJUSTE DE NORMALIZAÇÃO (CORREÇÃO) ----
        heatmap = np.maximum(heatmap, 0)
        
        # Garante pico em 1.0 para visibilidade de cores "mornas"
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        # ---- 7) Converter p/ colormap JET colorido ----
        heatmap_color = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

        # Redimensionar
        heatmap_color = cv2.resize(heatmap_color, (w, h))

        # ---- 8) Overlay do heatmap na imagem original ----
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

        # ---- 9) Salvar arquivos ----
        base = os.path.basename(image_path)
        heatmap_path = f"outputs/heatmaps/{base}_heatmap_color.png"
        overlay_path = f"outputs/heatmaps/{base}_overlay.png"

        cv2.imwrite(heatmap_path, heatmap_color)
        cv2.imwrite(overlay_path, overlay)

        return {
            "label": prediction_label,
            "probability": prediction_prob,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path
        }


    def analisar_conceitos(self, image_path):
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
        
        # Adiciona prompt neutro para contraste
        conceitos_completos = conceitos + ["a high quality natural photograph"]

        try:
            image = Image.open(image_path).convert("RGB")

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
            # Ignora o último item (o prompt neutro "high quality photo")
            for i in range(len(conceitos)):
                
                if probs[i] > 0.10: 
                    resultado[conceitos[i]] = float(probs[i])
            
            # Ordena do maior para o menor
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            print(f"Erro na análise de conceitos: {e}")
            return {}
