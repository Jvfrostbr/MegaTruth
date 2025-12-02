import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

class CLIPAIModel:
    def __init__(self, device=None):
        self.model_name = "openai/clip-vit-base-patch16"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Dispositivo de execução selecionado: {self.device}")

        if self.device == "cuda":
            torch.cuda.current_device()

        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

        self.classes = ["a real photograph", "an AI-generated image"]

    def predict_with_heatmap(self, image_path):
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

        # Remove valores negativos
        heatmap = np.maximum(heatmap, 0)
        
        # Normaliza pelo valor máximo global para garantir que o pico seja 1.0
        # Isso ajuda a tornar visíveis ativações fracas (que ficariam pretas)
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

        # ---- Retorno ----
        return {
            "label": prediction_label,
            "probability": prediction_prob,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path
        }