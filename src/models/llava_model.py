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
            # Prompt atualizado com instrução de idioma
            prompt = f"""
                VOCÊ É UM ASSISTENTE DE ANÁLISE VISUAL FOCADO EM DETALHES.
                
                Sua tarefa é analisar DUAS imagens:
                1. A Imagem Original (Retrato).
                2. O Overlay (Sobreposição): Mostra pontos coloridos/brilhantes sobre a imagem onde o detector encontrou sinais de IA.

                CONTEXTO:
                Um detector de IA classificou esta imagem como: "{classificacao_clip}" ({probabilidade_clip:.1%} de certeza).
                
                INSTRUÇÃO: Responda em PORTUGUÊS seguindo a lista abaixo.

                1. O que você vê na imagem original? (Descreva a pessoa, o estilo e o fundo).
                2. Onde estão concentrados os pontos coloridos/brilhantes no Overlay? (Ex: Estão nos olhos? Na pele? No cabelo? No fundo?).
                3. Olhando para a imagem original nessas mesmas áreas destacadas, a textura parece natural?
                - Procure por: Pele lisa demais (plástico), olhos assimétricos, cabelo borrado ou fundo estranho.
                4. Conclusão: Por que esses detalhes visuais suportam a classificação de "{classificacao_clip}"?
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