import torch
import numpy as np
from PIL import Image
import cv2
import os
import warnings
from transformers import CLIPProcessor, CLIPModel, CLIPSegProcessor, CLIPSegForImageSegmentation

warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

class CLIPAIModel:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Dispositivo de Inferência: {self.device}")
        
        if self.device == "cuda":
            torch.cuda.current_device()

        # 1. Carrega Arquivos de Configuração (Conceitos e Âncoras)
        self._load_configurations()

        # 2. Modelo Tuned (O Juiz - Classificação)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(base_dir, "clip_finetuned")
        path_tuned = "openai/clip-vit-base-patch16" # Fallback

        if os.path.exists(default_path) and model_path != "openai/clip-vit-base-patch16":
            path_tuned = default_path
            print(f"🧠 Usando modelo Fine-Tuned (Especialista): {path_tuned}")
        else:
            print("⚠️ Modelo Fine-Tuned não encontrado. Usando Base para tudo.")

        try:
            self.proc_tuned = CLIPProcessor.from_pretrained(path_tuned, use_fast=True)
            self.model_tuned = CLIPModel.from_pretrained(
                path_tuned,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model_tuned.eval()
        except Exception as e:
            print(f"Erro crítico ao carregar modelo Tuned: {e}")
            raise e
        
        # 3. Modelo Base (O Semântico - Conceitos)
        print("👁️ Carregando Modelo Base (Conceitos)...")
        try:
            self.proc_base = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
            self.model_base = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.model_base.eval()
        except Exception as e:
             print(f"Erro ao carregar modelo Base: {e}")
             self.model_base = self.model_tuned
             self.proc_base = self.proc_tuned

        # 4. CLIPSeg (O Desenhista - defect_maps Precisos)
        print("🎨 Carregando CLIPSeg (Segmentação Visual)...")
        try:
            self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
            self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
            self.seg_model.eval()
        except Exception as e:
            print(f"❌ Erro ao baixar CLIPSeg: {e}")
            raise e
        
        # Classes internas em Inglês para o CLIP
        self.classes_eng = ["a real photograph", "an AI-generated image"]
        
        # Mapeamento para Português (Output)
        self.classes_pt_map = {
            "a real photograph": "Fotografia Real",
            "an AI-generated image": "Imagem Gerada por IA"
        }

    def _load_configurations(self):
        """Lê os arquivos txt de conceitos e âncoras para memória."""
        self.concepts_eng = []      # Lista para o CLIP (Inglês)
        self.concepts_map = {}      # Tradução (Inglês -> Português)
        self.visual_anchors = {}    # Mapeamento (Keyword -> Target Visual)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config") 

        # Carregar Conceitos
        try:
            with open(os.path.join(config_dir, "concepts.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if ";" in line:
                        eng, pt = line.strip().split(";")
                        self.concepts_eng.append(eng.strip())
                        self.concepts_map[eng.strip()] = pt.strip()
            print(f"✅ Carregados {len(self.concepts_eng)} conceitos.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar concepts.txt: {e}")
            # Fallback básico se arquivo falhar
            self.concepts_eng = ["artifacts", "blur"]
            self.concepts_map = {"artifacts": "artefatos", "blur": "borrão"}

        # Carregar Âncoras
        try:
            with open(os.path.join(config_dir, "anchors.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if ";" in line:
                        key, target = line.strip().split(";")
                        self.visual_anchors[key.strip()] = target.strip()
            print(f"✅ Carregadas {len(self.visual_anchors)} âncoras visuais.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar anchors.txt: {e}")

    def _generate_segmentation(self, image, prompts):
        """
        Usa CLIPSeg para gerar máscaras precisas.
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

    def predict_with_defect_map(self, image_path, overlay_color="red"):
        """
        Pipeline principal: Classifica -> Analisa Conceitos -> Gera defect_map -> Traduz Saída.
        Args:
            image_path (str): Caminho da imagem.
            overlay_color (str): 'red', 'green', ou 'blue'. Define a cor da mancha.
        """
        os.makedirs("outputs/defect_maps", exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        
        # --- 1. Classificação (Tuned - Inglês) ---
        inputs = self.proc_tuned(
            text=self.classes_eng, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model_tuned(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label_eng = self.classes_eng[pred_idx]
            prob = float(probs[pred_idx])

        # --- 2. Definição dos Prompts para o CLIPSeg ---
        # Prompts padrão (fallback)
        seg_prompts = None
        conceitos_eng = {}
        
        # Se for FAKE ou incerto, buscamos o defeito específico
        if pred_idx == 1 or prob < 0.85:
            # Analisa conceitos (retorna dict em Inglês)
            conceitos_eng = self.analisar_conceitos(image_path, classificacao_preliminar=label_eng)
            
            if conceitos_eng:
                original_concept = list(conceitos_eng.keys())[0] # Ex: "deformed fingers"
                
                # Busca âncora visual
                visual_target = original_concept
                for key, val in self.visual_anchors.items():
                    if key in original_concept.lower():
                        visual_target = val
                        break
                
                print(f"   >>> CLIPSeg Alvo: '{visual_target}' (Origem: {original_concept})")
                seg_prompts = [visual_target]

        # --- 3. Geração da Máscara ---
        if seg_prompts and len(seg_prompts) > 0:
            print(f"   >>> Gerando Segmentação para: {seg_prompts}")
            defect_map = self._generate_segmentation(image, seg_prompts)

            # --- 4. Pós-Processamento Visual ---
            defect_map_min = np.min(defect_map)
            defect_map_max = np.max(defect_map)
            if defect_map_max > defect_map_min:
                defect_map = (defect_map - defect_map_min) / (defect_map_max - defect_map_min)
            else:
                defect_map = np.zeros_like(defect_map)
            
            defect_map[defect_map < 0.35] = 0
            defect_map_smooth = cv2.GaussianBlur(defect_map, (15, 15), 0)

            # --- 5. GERAÇÃO DO OVERLAY COLORIDO ---
            img_np = np.array(image)
            color_mask = np.zeros_like(img_np)
            
            # Define a cor da máscara (RGB aqui, pois o PIL abriu como RGB)
            if overlay_color == "green":
                color_mask[:, :, 1] = 255  # Canal G (Verde)
            elif overlay_color == "blue":
                color_mask[:, :, 2] = 255  # Canal B (Azul)
            else: # Default: Red
                color_mask[:, :, 0] = 255  # Canal R (Vermelho)
            
            img_float = img_np.astype(np.float32) / 255.0
            mask_float = color_mask.astype(np.float32) / 255.0
            alpha = defect_map_smooth[:, :, None]
            
            # Mistura: (Cor * alpha) + (Imagem * (1 - alpha*0.3))
            # O fator 0.3 no alpha negativo mantém a imagem original visível por baixo
            overlay = (mask_float * alpha * 0.6) + (img_float * (1.0 - (alpha * 0.3)))
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            
            # Converte RGB -> BGR para o OpenCV salvar corretamente
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            base = os.path.basename(image_path)
            overlay_path = f"outputs/defect_maps/{base}_clipseg.png"
            cv2.imwrite(overlay_path, overlay_bgr)
            
        else:
            overlay_path = image_path  # Sem overlay gerado

        # --- 6. TRADUÇÃO PARA SAÍDA (PT-BR) ---
        label_pt = self.classes_pt_map.get(label_eng, label_eng)
        
        conceitos_pt = {}
        for k_eng, v_prob in conceitos_eng.items():
            k_pt = self.concepts_map.get(k_eng, k_eng)
            conceitos_pt[k_pt] = v_prob

        probs_pt = {self.classes_pt_map[self.classes_eng[i]]: float(probs[i]) for i in range(len(self.classes_eng))}

        return {
            "label": label_pt,
            "probability": prob, 
            "probabilities": probs_pt,
            "defect_map_path": overlay_path,
            "overlay_path": overlay_path, 
            "conceitos": conceitos_pt,
            "color_used": overlay_color 
        }
        
    def analisar_conceitos(self, image_path, classificacao_preliminar=None):
        """
        Testa a imagem contra a lista de conceitos carregada (Inglês).
        """
        # Adiciona prompt de controle
        conceitos_completos = self.concepts_eng + ["a high quality natural photograph"]

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Gating
            if classificacao_preliminar == "a real photograph" or classificacao_preliminar == 0:
                threshold = 0.25 
            else:
                threshold = 0.10
                
            inputs = self.proc_base(
                text=conceitos_completos,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model_base(**inputs)
                logits_per_image = outputs.logits_per_image 
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            resultado = {}
            
            # Varre apenas os conceitos (ignora o último que é o controle)
            for i in range(len(self.concepts_eng)):
                if probs[i] > threshold: 
                    resultado[self.concepts_eng[i]] = float(probs[i])
            
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            print(f"Erro na análise de conceitos: {e}")
            return {}