import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ZERO_SHOT_PROMPTS = ["a real photograph", "an AI-generated image"]

# --- Classe do modelo Fine-Tuned (CLIP + Head) ---

class CLIPWithHead:
    def __init__(self, base_path, head_path):
        self.device = DEVICE

        print(f"Carregando modelo base: {base_path}")
        self.processor = CLIPProcessor.from_pretrained(base_path, use_fast = True)
        self.clip = CLIPModel.from_pretrained(base_path).to(self.device)
        self.clip.eval()

        print(f"Carregando head: {head_path}")
        head_file = os.path.join(head_path, "classifier_head.pth")
        state = torch.load(head_file, map_location=self.device)

        # A dimensão do CLIP-base é 512
        self.head = nn.Linear(512, 2).to(self.device)
        self.head.load_state_dict(state)
        self.head.eval()

        # classes já no padrão correto
        self.classes_map = ["an AI-generated image", "a real photograph"]

        print("Modelo fine-tuned carregado com sucesso!\n")

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # extrai embeddings de imagem do CLIP
            image_features = self.clip.get_image_features(inputs.pixel_values)

            # normaliza como o CLIP usa internamente
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # passa pelo head treinado
            logits = self.head(image_features)

            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        return np.argmax(probs)


# --- Classe do modelo Zero-Shot Original ---
class CLIPZeroShot:
    def __init__(self, model_path):
        self.device = DEVICE
        self.processor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
        self.clip = CLIPModel.from_pretrained(model_path).to(self.device)
        self.clip.eval()

        self.classes_map = ZERO_SHOT_PROMPTS

        print(f"Modelo Zero-Shot carregado: {model_path}")

    def predict(self, image):
        text_inputs = self.processor(text=self.classes_map, return_tensors="pt", padding=True).to(self.device)
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.clip(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values,
                return_loss=False
            )
            logits = outputs.logits_per_image
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        return 1 - np.argmax(probs)  # inverter para ficar igual ao dataset (0=AI, 1=Real)


# --- Avaliação ---
def evaluate(model, dataset):
    preds, labels = [], []

    for item in tqdm(dataset, desc="Avaliação"):
        pred = model.predict(item["image"])
        preds.append(pred)
        labels.append(item["label"])

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds)

    return acc, f1, cm


# --- MAIN ---
if __name__ == "__main__":
    # dataset
    full_dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    split_ds = full_dataset.train_test_split(test_size=0.1, seed=42)
    val = split_ds["test"]

    # Zero-Shot
    zs = CLIPZeroShot("openai/clip-vit-base-patch32")

    # Fine-Tuned
    base_path = "src/models/clip_finetunned/clip_finetuned_base"
    head_path = "src/models/clip_finetunned/clip_finetuned_head"

    ft = CLIPWithHead(base_path, head_path)

    print("\n=== Fine-Tuned ===")
    acc_ft, f1_ft, cm_ft = evaluate(ft, val)
    print(acc_ft, f1_ft)
    print(cm_ft)

    print("\n=== Zero-Shot ===")
    acc_zs, f1_zs, cm_zs = evaluate(zs, val)
    print(acc_zs, f1_zs)
    print(cm_zs)
