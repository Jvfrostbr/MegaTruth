import os
import pandas as pd # Importei pandas apenas para formatar data se precisar, mas o foco é o texto

# Remove a variável de ambiente problemática se ela existir
if 'SSL_CERT_FILE' in os.environ:
    os.environ.pop('SSL_CERT_FILE')

import ollama  

class LLaVAModel:
    def __init__(self):
        self.model_name = "llava:7b"
        self._verificar_modelo()
    
    def _verificar_modelo(self):
        """Verifica se o modelo LLaVA está disponível e baixa se necessário"""
        try:
            models = ollama.list()
            model_names = [m['model'] for m in models['models']]
            
            if self.model_name not in model_names:
                print("LLaVA-7B não encontrado. Baixando...")
                ollama.pull(self.model_name)
                print("LLaVA-7B baixado com sucesso!")
            else:
                print("LLaVA-7B já está disponível.")
                
        except Exception as e:
            print(f"Erro ao verificar modelo LLaVA: {e}")
            raise

    def analisar_imagens(self, imagem_original, heatmap, classificacao_clip, probabilidade_clip, conceitos_detectados=None):
        """
        Analisa a imagem original e o heatmap usando LLaVA-7B.
        Agora inclui os 'conceitos_detectados' (Concept Bottleneck) como evidência.
        """
        
        # Verifica se as imagens existem
        if not os.path.exists(imagem_original):
            raise FileNotFoundError(f"Imagem original não encontrada: {imagem_original}")
            
        if not os.path.exists(heatmap):
            raise FileNotFoundError(f"Heatmap não encontrado: {heatmap}")
        
        # Lê os arquivos como bytes
        with open(imagem_original, 'rb') as f:
            image_original_bytes = f.read()
        with open(heatmap, 'rb') as f:
            heatmap_bytes = f.read()

        print("Analisando imagens com LLaVA-7B...")
        
        # --- PREPARAR A LISTA DE CONCEITOS PARA O PROMPT ---
        texto_conceitos = "Nenhum defeito específico detectado."
        if conceitos_detectados:
            # Pega os top 5 conceitos para não poluir demais
            top_conceitos = list(conceitos_detectados.items())[:5]
            
            # Formata uma lista : "- 'deformed hands' (85% de sinal)"
            lista_str = "\n".join([f"   - '{k}' ({v:.1%} de intensidade)" for k, v in top_conceitos])
            
            texto_conceitos = f"""
            ALERTA DE ANÁLISE SEMÂNTICA (IMPORTANTE):
            O detector identificou os seguintes padrões visuais específicos nesta imagem:
            {lista_str}
            
            > USE ESTA LISTA COMO GUIA: Verifique se esses defeitos específicos aparecem nas áreas coloridas do heatmap.
            """

        try:
            # Prompt atualizado com a injeção dos conceitos
            prompt = f"""
                VOCÊ É UM ASSISTENTE DE ANÁLISE VISUAL FOCADO EM DETALHES FORENSES.
                
                Sua tarefa é analisar DUAS imagens para explicar uma detecção de IA:
                1. A Imagem Original (Retrato/Cena).
                2. O Overlay (Sobreposição): Mapa de calor onde pontos COLORIDOS/BRILHANTES indicam anomalias.
    
                CONTEXTO GERAL:
                Classificação do Detector: "{classificacao_clip}" ({probabilidade_clip:.1%} de certeza).
                
                {texto_conceitos}
                
                INSTRUÇÃO: Responda em PORTUGUÊS seguindo a lista abaixo.

                1. O que você vê na imagem original? (Descreva brevemente a cena/pessoa).
                2. Onde estão concentrados os pontos coloridos no Overlay? (Ex: Olhos, mãos, fundo, cabelo).
                3. Olhando para a imagem original nessas áreas destacadas, a textura parece natural?
                   - Se a lista acima mencionou defeitos (ex: 'waxy skin', 'deformed hands'), confirme se você os vê nessas áreas.
                4. Conclusão: Como os pontos do heatmap e os defeitos listados confirmam a classificação de "{classificacao_clip}"?
            """
            
            # Envia as duas imagens para o LLaVA
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_original_bytes, heatmap_bytes]
                    }
                ]
            )
            
            print("Análise concluída!\n")
            print("=" * 60)
            print("RESPOSTA DO LLaVA:")
            print("=" * 60)
            print(response['message']['content'])
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Erro ao analisar imagens: {e}")
            return None