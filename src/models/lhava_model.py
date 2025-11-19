import os

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

    def analisar_imagens(self, imagem_original, heatmap, classificacao_clip, probabilidade_clip):
        """
        Analisa a imagem original e o heatmap usando LLaVA-7B, incluindo a classificação do CLIP
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

        print("Analisando imagens com LLaVA-7B.")
        
        try:
            # Prompt atualizado - assume que a classificação do CLIP está correta
            prompt = f"""
            Analise estas duas imagens:
            - A primeira é uma imagem original
            - A segunda é um heatmap que mostra as regiões importantes para uma decisão de classificação

            CONTEXTO:
            Um modelo de visão computacional (CLIP) classificou esta imagem como: "{classificacao_clip}"
            com {probabilidade_clip:.1%} de confiança. O heatmap mostra quais regiões da imagem foram 
            mais importantes para o modelo chegar a essa conclusão.

            Com base nas imagens e assumindo que a classificação do CLIP está correta, explique:

            1. O que você vê na imagem original
            2. Quais regiões estão destacadas no heatmap como mais importantes para a classificação
            3. Como as regiões destacadas no heatmap se relacionam com características que distinguem 
               "{classificacao_clip}" do outro tipo de imagem
            4. Por que o heatmap faz sentido em relação ao conteúdo da imagem original e à classificação
            5. Quais elementos visuais na imagem suportam a classificação como "{classificacao_clip}"
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