import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class CLIPAIModel:
    def __init__(self, device=None):
        self.model_name = "openai/clip-vit-base-patch32"
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Dispositivo de execução selecionado: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.classes = ["a real photograph", "an AI-generated image"]

    def predict_with_heatmap(self, image_path):
        # ---- 1) Carregar imagem ----
        image = Image.open(image_path).convert("RGB")

        # ---- 2) Preparar textos ----
        text_inputs = self.processor(
            text=self.classes,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # ---- 3) Preparar imagem ----
        image_inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # ---- 4) Forward padrão (CLIP) ----
        with torch.no_grad():
            outputs = self.model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values,
                return_loss=False
            )
            logits = outputs.logits_per_image

        # ---- 5) Probabilidades ----
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        prediction_idx = int(np.argmax(probs))
        prediction_label = self.classes[prediction_idx]
        prediction_prob = float(probs[prediction_idx])

        # ---- 6) HEATMAP (Gradiente do embedding da imagem) ----
        image_inputs.pixel_values.requires_grad_(True)

        text_embeds = self.model.get_text_features(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )

        image_embeds = self.model.get_image_features(
            pixel_values=image_inputs.pixel_values
        )

        score = torch.matmul(image_embeds, text_embeds[prediction_idx].unsqueeze(1)).squeeze()
        score.backward()

        grads = image_inputs.pixel_values.grad[0].cpu().numpy()
        heatmap = np.mean(grads, axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # ---- 7) Converter heatmap → PIL ----
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size)

        return {
            "label": prediction_label,
            "probability": prediction_prob,
            "probabilities": {
                self.classes[i]: float(probs[i]) for i in range(len(self.classes))
            },
            "heatmap": heatmap_img
        }


if __name__ == "__main__":
    model = CLIPAIModel()
    result = model.predict_with_heatmap("src/exemplo.jpg")
    
    print(result["label"], result["probability"])
    result["heatmap"].show()

    print("CUDA disponível?", torch.cuda.is_available())
    print("Número de GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Nome da GPU:", torch.cuda.get_device_name(0))
