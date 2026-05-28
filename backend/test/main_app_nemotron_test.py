import os
import sys
import pandas as pd  # Para timestamp no relatório
from dotenv import load_dotenv

# Carregar variáveis de ambiente (.env) para a API Key
try:
    
    load_dotenv()
except ImportError:
    print("python-dotenv não instalado. Certifique-se que a chave OPENROUTER_API_KEY está no ambiente.")

# Adiciona o diretório src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.vision_model_clip import CLIPAIModel         # noqa: E402
from models.multimodal_model_nemotron import NemotronVL  # noqa: E402


if __name__ == "__main__":
    print("=" * 60)
    print("MEGATRUTH - TESTE DE INTEGRAÇÃO (NEMOTRON 12B - API)")
    print("=" * 60)
    
    # Caminho da imagem a ser analisada
    imagem_path = "images/inferences/AI/monalisa_picture.jpg" # <--- Altere aqui
    
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
        
        # 1.1. Detecção Visual e defect_map
        print("Inicializando CLIP (Detector)...")
        clip_model = CLIPAIModel() 
        
        print(f"Analisando imagem: {imagem_path}")
        resultado_clip = clip_model.predict_with_defect_map(imagem_path)
        
        print(f"   -> Classificação: {resultado_clip['label'].upper()}")
        print(f"   -> Confiança:     {resultado_clip['probability']:.2%}")
        print(f"   -> defect_map salvo: {resultado_clip['overlay_path']}")

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
        # FASE 2: ANÁLISE EXPLICATIVA (NEMOTRON VIA API)
        # ==============================================================================
        print("\n\n🧠 FASE 2 - Análise Explicativa (NVIDIA Nemotron-12B)")
        print("-" * 50)
        
        print("Conectando à API OpenRouter...")
        nemotron = NemotronVL()
        
        print("Solicitando análise pericial...")
        
        # Chama o Nemotron passando o Overlay E a lista de Conceitos
        analise_final = nemotron.analisar_imagens(
            imagem_original=imagem_path,
            defect_map=resultado_clip["overlay_path"],
            classificacao_clip=resultado_clip["label"],
            probabilidade_clip=resultado_clip["probability"],
            conceitos_detectados=conceitos,
            color_overlay="vermelho" 
        )

        # ==============================================================================
        # FASE 3: GERAÇÃO DE RELATÓRIO
        # ==============================================================================
        if analise_final:
            print("\n\n📝 FASE 3 - Gerando Relatório Final")
            print("-" * 50)
            
            base_name = os.path.basename(imagem_path).split('.')[0]
            report_path = f"outputs/reports/{base_name}_relatorio_nemotron.txt"
            
            # Prepara o texto dos conceitos para o arquivo
            conceitos_txt = "Nenhum defeito específico identificado."
            if conceitos:
                conceitos_txt = "\n".join([f"- {k} ({v:.1%})" for k, v in conceitos.items()])

            relatorio_texto = f"""RELATÓRIO DE ANÁLISE FORENSE - MEGATRUTH (NUVEM)
                ==================================================================
                ARQUIVO: {imagem_path}
                DATA:    {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
                MODELO:  NVIDIA Nemotron-12B (OpenRouter API)
                ==================================================================

                1. RESULTADOS TÉCNICOS (Detector)
                ---------------------------------
                Classificação:   {resultado_clip['label'].upper()}
                Grau de Certeza: {resultado_clip['probability']:.2%}
                defect_map (Foco):  {os.path.abspath(resultado_clip['overlay_path'])}

                2. ANÁLISE SEMÂNTICA (Defeitos Específicos - Concept Bottleneck)
                ----------------------------------------------------------------
                O sistema verificou a presença de anomalias físicas e lógicas.
                Conceitos encontrados (acima de 10% de confiança):

                {conceitos_txt}

                3. PARECER PERICIAL (Nemotron Vision)
                -------------------------------------
                {analise_final}

                ==================================================================
                Fim do Relatório
                """
            # Salva o arquivo
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(relatorio_texto)
            
            print(f"📄 Relatório salvo em: {report_path}")
            print("\n--- ANÁLISE DO NEMOTRON ---")
            print(analise_final)
            
        else:
            print("\n❌ Falha: O Nemotron não retornou uma análise (Verifique API Key/Créditos).")

    except Exception as e:
        print(f"\n❌ Erro fatal durante a execução: {e}")
        import traceback
        traceback.print_exc()

    def test_classification(imagem_path):
        """
        Função de teste para verificar a classificação do CLIP.
        """
        try:
            clip_model = CLIPAIModel()
        except Exception as e:
            print(f"Erro durante o teste de classificação: {e}")

        resultados = clip_model.predict_with_defect_map(imagem_path)

        classificacao = resultados['label']
        probabilidade = resultados['probability']
        defect_map_path = resultados['overlay_path']

        conceitos = clip_model.analisar_conceitos(imagem_path)

        try:
            nemotron = NemotronVL()

        except Exception as e:
            print(f"Erro ao inicializar o Nemotron: {e}")
            return classificacao, probabilidade, defect_map_path, conceitos
        
        analise_final = nemotron.analisar_imagens(
            imagem_original=imagem_path,
            defect_map=defect_map_path,
            classificacao_clip=classificacao,
            probabilidade_clip=probabilidade,
            conceitos_detectados=conceitos,
            color_overlay="vermelho"
        )

        return classificacao, probabilidade, defect_map_path, conceitos, analise_final
