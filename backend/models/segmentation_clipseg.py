import torch
import numpy as np
from PIL import Image
import cv2
import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegModel:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print("🎨 Carregando CLIPSeg (Segmentação Visual)...")
        try:
            self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
            self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
            self.seg_model.eval()
        except Exception as e:
            print(f"❌ Erro ao baixar CLIPSeg: {e}")
            raise e

    def _generate_segmentation(self, image, prompts):
        """
        Gera as máscaras cruas usando o CLIPSeg.
        """
        inputs = self.seg_processor(
            text=prompts, 
            images=[image] * len(prompts), 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.seg_model(**inputs)
        
        preds = outputs.logits
        
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(0)
            
        masks = torch.sigmoid(preds).cpu().numpy()
        
        w, h = image.size
        final_mask = np.zeros((h, w), dtype=np.float32)
        
        for mask in masks:
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            mask_resized = cv2.resize(mask, (w, h))
            final_mask = np.maximum(final_mask, mask_resized)
            
        return final_mask

    def generate_defect_overlay(self, image_path, prompts, overlay_color="red", prompt_index=0):
        """
        Gera a máscara, aplica pós-processamento e salva o overlay.
        """
        os.makedirs("outputs/defect_maps", exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        
        print(f"   >>> Gerando Segmentação para: {prompts}")
        defect_map = self._generate_segmentation(image, prompts)

        # --- Pós-Processamento Visual ---
        defect_map_min = np.min(defect_map)
        defect_map_max = np.max(defect_map)
        if defect_map_max > defect_map_min:
            defect_map = (defect_map - defect_map_min) / (defect_map_max - defect_map_min)
        else:
            defect_map = np.zeros_like(defect_map)
        
        # Limiarização
        defect_map[defect_map < 0.35] = 0

        # Suavização Adaptativa
        h, w = defect_map.shape
        k_size = int(min(h, w) * 0.03)
        if k_size % 2 == 0:
            k_size += 1
        if k_size < 3:
            k_size = 3
            
        defect_map_smooth = cv2.GaussianBlur(defect_map, (k_size, k_size), 0)

        # --- GERAÇÃO DO OVERLAY ---
        img_np = np.array(image)
        color_mask = np.zeros_like(img_np)
        
        if overlay_color == "green":
            color_mask[:, :, 1] = 255
        elif overlay_color == "blue":
            color_mask[:, :, 2] = 255
        else: 
            color_mask[:, :, 0] = 255
        
        img_float = img_np.astype(np.float32) / 255.0
        mask_float = color_mask.astype(np.float32) / 255.0
        alpha = defect_map_smooth[:, :, None]
        
        # Mistura alpha ponderada
        overlay = (mask_float * alpha * 0.6) + (img_float * (1.0 - (alpha * 0.3)))
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        # Converte para BGR e salva
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        base = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base)[0]
        
        # Adiciona índice para diferenciar múltiplos overlays
        overlay_path = f"outputs/defect_maps/{name_without_ext}_defect_{prompt_index}.png"
        cv2.imwrite(overlay_path, overlay_bgr)
        
        return overlay_path