import os
import sys
import pandas as pd  # Para timestamp no relatório

# Adiciona o diretório src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# Importa apenas os modelos locais
from models.vision_model_clip import CLIPAIModel        # noqa: E402
from models.multimodal_model_llava import LLaVAModel    # noqa: E402

if __name__ == "__main__":
    print("=" * 60)
    print("MEGATRUTH - TESTE DE INTEGRAÇÃO (VERSÃO LOCAL - LLAVA)")
    print("=" * 60)
    
    # Caminho da imagem a ser analisada
    imagem_path = "images/inferences/AI/mao_deformada.jpeg" # <--- Altere aqui para testar outras
  
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
        print("Inicializando CLIP (Detector)...")
        clip_model = CLIPAIModel() 
        
        print(f"Analisando imagem: {imagem_path}")
        resultado_clip = clip_model.predict_with_heatmap(imagem_path)
        
        print(f"   -> Classificação: {resultado_clip['label'].upper()}")
        print(f"   -> Confiança:     {resultado_clip['probability']:.2%}")
        print(f"   -> Heatmap salvo: {resultado_clip['overlay_path']}")

        # 1.2. Análise de Conceitos (Concept Bottleneck - Lista V3)
        print("\nExecutando varredura de defeitos específicos (Concept Bottleneck)...")
        conceitos = clip_model.analisar_conceitos(imagem_path)
        
        # Mostra no terminal os conceitos encontrados
        if conceitos:
            print("   ⚠️  Defeitos detectados (Top 3):")
            for k, v in list(conceitos.items())[:3]: 
                print(f"       - {k}: {v:.1%}")
        else:
            print("   ✅ Nenhum defeito semântico óbvio detectado.")

        # ==============================================================================
        # FASE 2: ANÁLISE EXPLICATIVA (LLAVA LOCAL)
        # ==============================================================================
        print("\n\n🧠 FASE 2 - Análise Explicativa (LLaVA-7B)")
        print("-" * 50)
        
        print("Inicializando LLaVA (Ollama)...")
        llava_model = LLaVAModel()
        
        print("Solicitando análise pericial...")
        
        # Chama o LLaVA passando o Overlay E a lista de Conceitos
        analise_final = llava_model.analisar_imagens(
            imagem_original=imagem_path,
            heatmap=resultado_clip["overlay_path"], 
            classificacao_clip=resultado_clip["label"],
            probabilidade_clip=resultado_clip["probability"],
            conceitos_detectados=conceitos, 
            color_overlay="Vermelho" 
        )

        # ==============================================================================
        # FASE 3: GERAÇÃO DE RELATÓRIO
        # ==============================================================================
        if analise_final:
            print("\n\n📝 FASE 3 - Gerando Relatório Final")
            print("-" * 50)
            
            base_name = os.path.basename(imagem_path).split('.')[0]
            report_path = f"outputs/reports/{base_name}_relatorio_llava.txt"
            
            # Prepara o texto dos conceitos para o arquivo
            conceitos_txt = "Nenhum defeito específico identificado."
            if conceitos:
                conceitos_txt = "\n".join([f"- {k} ({v:.1%})" for k, v in conceitos.items()])

            relatorio_texto = f"""RELATÓRIO DE ANÁLISE FORENSE - MEGATRUTH (LOCAL)
                ==================================================================
                ARQUIVO: {imagem_path}
                DATA:    {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
                MODELO:  LLaVA-7B (Local/Ollama)
                ==================================================================

                1. RESULTADOS TÉCNICOS (Detector)
                ---------------------------------
                Classificação:   {resultado_clip['label'].upper()}
                Grau de Certeza: {resultado_clip['probability']:.2%}
                Heatmap (Foco):  {os.path.abspath(resultado_clip['overlay_path'])}

                2. ANÁLISE SEMÂNTICA (Defeitos Específicos - Concept Bottleneck)
                ----------------------------------------------------------------
                O sistema verificou a presença de anomalias físicas e lógicas.
                Conceitos encontrados (acima de 10% de confiança):

                {conceitos_txt}

                3. PARECER PERICIAL (LLaVA Vision)
                ----------------------------------
                {analise_final}

                ==================================================================
                Fim do Relatório
                """
            # Salva o arquivo
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(relatorio_texto)
            
            print(f"📄 Relatório salvo em: {report_path}")
            print("\n--- ANÁLISE DO LLAVA ---")
            print(analise_final)
            
        else:
            print("\n❌ Falha: O LLaVA não retornou uma análise.")

    except Exception as e:
        print(f"\n❌ Erro fatal durante a execução: {e}")
        import traceback
        traceback.print_exc()