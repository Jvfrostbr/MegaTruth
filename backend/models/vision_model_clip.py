import torch
import numpy as np
from PIL import Image
import os
import warnings
from transformers import CLIPProcessor, CLIPModel

# Importando o novo módulo de segmentação
from .segmentation_clipseg import CLIPSegModel

warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

class CLIPAIModel:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Dispositivo de Inferência: {self.device}")
        
        if self.device == "cuda":
            torch.cuda.current_device()

        # 1. Carrega Arquivos de Configuração
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

        # 4. Instancia o novo módulo de Segmentação
        self.segmenter = CLIPSegModel(device=self.device)
        
        # Classes internas
        self.classes_eng = ["a real photograph", "an AI-generated image"]
        self.classes_pt_map = {
            "a real photograph": "Fotografia Real",
            "an AI-generated image": "Imagem Gerada por IA"
        }

    def _load_configurations(self):
        """Lê os arquivos txt de conceitos e âncoras para memória."""
        self.concepts_eng = []      
        self.concepts_map = {}      
        self.visual_anchors = {}    

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config") 

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
            self.concepts_eng = ["artifacts", "blur"]
            self.concepts_map = {"artifacts": "artefatos", "blur": "borrão"}

        try:
            with open(os.path.join(config_dir, "anchors.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if ";" in line:
                        key, target = line.strip().split(";")
                        self.visual_anchors[key.strip()] = target.strip()
            print(f"✅ Carregadas {len(self.visual_anchors)} âncoras visuais.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar anchors.txt: {e}")

    def predict_with_defect_map(self, image_path, overlay_color="red"):
        """
        Pipeline principal: Classifica -> Analisa Conceitos -> Gera defect_map via CLIPSeg.
        """
        image = Image.open(image_path).convert("RGB")
        
        # --- 1. Classificação ---
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

        # --- 2. Definição dos Prompts ---
        seg_prompts = None
        conceitos_eng = {}
        
        if pred_idx == 1 or prob < 0.85:
            conceitos_eng = self.analisar_conceitos(image_path, classificacao_preliminar=label_eng)
            
            if conceitos_eng:
                seg_prompts = []
                for original_concept in conceitos_eng.keys():
                    visual_target = original_concept
                    for key, val in self.visual_anchors.items():
                        if key in original_concept.lower():
                            visual_target = val
                            break
                    
                    print(f"   >>> CLIPSeg Alvo: '{visual_target}' (Origem: {original_concept})")
                    seg_prompts.append(visual_target)

        # --- 3. Delegação da Máscara para o Módulo de Segmentação ---
        defect_maps = []
        if seg_prompts and len(seg_prompts) > 0:
            # Gera defect map para cada prompt/conceito
            for i, prompt in enumerate(seg_prompts):
                overlay_path = self.segmenter.generate_defect_overlay(image_path, [prompt], overlay_color, prompt_index=i)
                concept_eng = list(conceitos_eng.keys())[i] if i < len(conceitos_eng) else prompt
                concept_pt = self.concepts_map.get(concept_eng, concept_eng)
                prob_concept = conceitos_eng.get(concept_eng, 0.0)
                
                defect_maps.append({
                    "conceito": concept_pt,
                    "probabilidade": prob_concept,
                    "defect_map_path": overlay_path,
                    "prompt": prompt
                })

        # --- 4. Tradução para Saída ---
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
            "defect_maps": defect_maps,
            "conceitos": conceitos_pt,
            "color_used": overlay_color 
        }
        
    def analisar_conceitos(self, image_path, classificacao_preliminar=None):
        """
        Testa a imagem contra a lista de conceitos carregada.
        """
        conceitos_completos = self.concepts_eng + ["a high quality natural photograph"]

        try:
            image = Image.open(image_path).convert("RGB")
            
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
            for i in range(len(self.concepts_eng)):
                if probs[i] > threshold: 
                    resultado[self.concepts_eng[i]] = float(probs[i])
            
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            print(f"Erro na análise de conceitos: {e}")
            return {}