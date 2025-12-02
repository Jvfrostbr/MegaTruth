import os
import sys

# Adiciona o diretório src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vision_model_clip import CLIPAIModel
from models.multimodal_model_llava import LLaVAModel

# Exemplo de uso integrado dos modelos CLIP e LLaVA
if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE ANÁLISE DE IMAGENS - CLIP + LLaVA")
    print("=" * 60)
    
    # Caminho da imagem a ser analisada
    imagem_path = "images/AI/monalisa_picture.jpg"
    
    # Verifica se a imagem existe
    if not os.path.exists(imagem_path):
        print(f"Imagem não encontrada: {imagem_path}")
        print("Por favor, coloque uma imagem em 'images/exemplo.jpg' e tente novamente.")
        sys.exit(1)
    
    try:
        # ---- FASE 1: Análise com CLIP ----
        print("\nFASE 1 - Análise com CLIP")
        print("-" * 40)
        
        print("Inicializando modelo CLIP...")
        clip_model = CLIPAIModel()
        
        print(f"Analisando imagem: {imagem_path}")
        resultado_clip = clip_model.predict_with_heatmap(imagem_path)
        
        print("\nRESULTADO DA CLASSIFICAÇÃO (CLIP):")
        print(f"Classificação: {resultado_clip['label']}")
        print(f"Confiança: {resultado_clip['probability']:.2%}")
        print("Probabilidades completas:")
        for classe, prob in resultado_clip['probabilities'].items():
            print(f"  - {classe}: {prob:.2%}")
        print(f"Heatmap salvo em: {resultado_clip['heatmap_path']}")
        
        # ---- FASE 2: Análise explicativa com LLaVA ----
        print("\n\nFASE 2 - Análise Explicativa com LLaVA")
        print("-" * 50)
        
        print("Inicializando modelo LLaVA...")
        llava_model = LLaVAModel()
        
        print("Solicitando análise explicativa do heatmap...")
        analise_llava = llava_model.analisar_imagens(
            imagem_original=imagem_path,
            heatmap=resultado_clip["overlay_path"],
            classificacao_clip=resultado_clip["label"],
            probabilidade_clip=resultado_clip["probability"]
        )
        
        if analise_llava:
            print("\nAnálise completada com sucesso!")
        else:
            print("\nFalha na análise do LLaVA")
            
    except FileNotFoundError as e:
        print(f"\nErro de arquivo: {e}")
    except Exception as e:
        print(f"\nErro durante a execução: {e}")
        import traceback
        traceback.print_exc()