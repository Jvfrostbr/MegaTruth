import os
import sys
import pandas as pd  # Para timestamp no relatório

# Adiciona o diretório src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vision_model_clip import CLIPAIModel
from models.multimodal_model_llava import LLaVAModel
from models.multimodal_model_nemotron import NemotronVL

if __name__ == "__main__":
    print("=" * 60)
    print("MEGATRUTH - SISTEMA FORENSE DE DETECÇÃO DE IA")
    print("=" * 60)
    
    # Caminho da imagem a ser analisada
    imagem_path = "images/AI/monalisa_picture.jpg"
    
    # Verifica se a imagem existe
    if not os.path.exists(imagem_path):
        print(f"❌ Erro: Imagem não encontrada: {imagem_path}")
        sys.exit(1)
    
    try:
        # ==============================================================================
        # FASE 1: ANÁLISE TÉCNICA (CLIP + CONCEPTS)
        # ==============================================================================
        print("\n🔍 FASE 1 - Análise Técnica (Visual e Semântica)")
        print("-" * 50)
        
        # 1.1. Detecção Visual e Heatmap
        print("Inicializando CLIP...")
        clip_model = CLIPAIModel() # Ele tentará carregar o checkpoint local se existir
        
        print(f"Analisando imagem: {imagem_path}")
        resultado_clip = clip_model.predict_with_heatmap(imagem_path)
        
        print(f"   -> Classificação: {resultado_clip['label'].upper()}")
        print(f"   -> Confiança:     {resultado_clip['probability']:.2%}")
        print(f"   -> Heatmap salvo: {resultado_clip['overlay_path']}")

        # 1.2. Análise de Conceitos (Concept Bottleneck)
        print("\nExecutando varredura de defeitos específicos (Concept Bottleneck)...")
        conceitos = clip_model.analisar_conceitos(imagem_path)
        
        if conceitos:
            print("   ⚠️  Defeitos detectados:")
            for k, v in list(conceitos.items())[:3]: # Mostra top 3 no console
                print(f"       - {k}: {v:.1%}")
        else:
            print("   ✅ Nenhum defeito semântico óbvio detectado.")

        # ==============================================================================
        # FASE 2: ANÁLISE EXPLICATIVA (INTELIGÊNCIA HÍBRIDA)
        # ==============================================================================
        print("\n\n🧠 FASE 2 - Análise Explicativa Multimodal")
        print("-" * 50)
        
        analise_final = None
        modelo_utilizado = "Nenhum"

        # --- TENTATIVA A: NEMOTRON (Nuvem/API - Mais Inteligente) ---
        try:
            print("Tentando conexão com Nemotron-12B (OpenRouter)...")
            nemotron = NemotronVL()
            analise_final = nemotron.analisar_imagens(
                imagem_original=imagem_path,
                heatmap=resultado_clip["overlay_path"],
                classificacao_clip=resultado_clip["label"],
                probabilidade_clip=resultado_clip["probability"],
                conceitos_detectados=conceitos 
            )
            
            if analise_final:
                modelo_utilizado = "NVIDIA Nemotron-12B (Via API)"
                print("✅ Sucesso! Análise gerada pelo Nemotron.")
                
        except Exception as e:
            print(f"⚠️  Nemotron indisponível ou erro de API: {e}")
            print("   -> Alternando para modelo local...")

        # --- TENTATIVA B: LLAVA (Local - Fallback) ---
        if not analise_final:
            print("Iniciando LLaVA-7B (Local)...")
            try:
                llava_model = LLaVAModel()
                analise_final = llava_model.analisar_imagens(
                    imagem_original=imagem_path,
                    heatmap=resultado_clip["overlay_path"],
                    classificacao_clip=resultado_clip["label"],
                    probabilidade_clip=resultado_clip["probability"],
                    conceitos_detectados=conceitos
                )
                
                if analise_final:
                    modelo_utilizado = "LLaVA-7B (Ollama Local)"
                    print("✅ Sucesso! Análise gerada pelo LLaVA.")
                    
            except Exception as e:
                print(f"❌ Erro crítico: O LLaVA também falhou. {e}")

        # ==============================================================================
        # FASE 3: GERAÇÃO DE RELATÓRIO
        # ==============================================================================
        if analise_final:
            print("\n\n📝 FASE 3 - Gerando Relatório Final")
            print("-" * 50)
            
            base_name = os.path.basename(imagem_path).split('.')[0]
            report_path = f"outputs/heatmaps/{base_name}_relatorio.txt"
            
            # Formata lista de conceitos para o relatório
            conceitos_txt = "Nenhum defeito específico identificado."
            if conceitos:
                conceitos_txt = "\n".join([f"- {k} ({v:.1%})" for k, v in conceitos.items()])

            relatorio_texto = f"""RELATÓRIO DE ANÁLISE FORENSE - MEGATRUTH
==================================================================
ARQUIVO: {imagem_path}
DATA:    {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
MODELO:  {modelo_utilizado}
==================================================================

1. RESULTADOS TÉCNICOS (Detector)
---------------------------------
Classificação:   {resultado_clip['label'].upper()}
Grau de Certeza: {resultado_clip['probability']:.2%}
Heatmap (Foco):  {os.path.abspath(resultado_clip['overlay_path'])}

2. ANÁLISE SEMÂNTICA (Defeitos Específicos)
-------------------------------------------
O sistema verificou a presença de anomalias físicas e lógicas:
{conceitos_txt}

3. PARECER PERICIAL (IA Multimodal)
-----------------------------------
{analise_final}

==================================================================
Fim do Relatório
"""
            # Salva o arquivo
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(relatorio_texto)
            
            print(f"📄 Relatório completo salvo em: {report_path}")
            
            # Imprime um preview no terminal
            print("\n--- PREVIEW DO PARECER ---")
            print(analise_final)
            print("--------------------------")

    except Exception as e:
        print(f"\n❌ Erro fatal durante a execução: {e}")
        import traceback
        traceback.print_exc()