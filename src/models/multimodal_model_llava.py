import os
import pandas as pd # Importei pandas apenas para formatar data se precisar, mas o foco é o texto
import base64

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

    def analisar_imagens(self, imagem_original, defect_map, classificacao_clip, probabilidade_clip, conceitos_detectados=None, color_overlay="vermelho"):
        """
        Analisa a imagem original e o defect_map usando LLaVA-7B.
        Agora inclui os 'conceitos_detectados' (Concept Bottleneck) como evidência.
        """
        
        # Verifica se as imagens existem
        if not os.path.exists(imagem_original):
            raise FileNotFoundError(f"Imagem original não encontrada: {imagem_original}")
            
        if not os.path.exists(defect_map):
            raise FileNotFoundError(f"defect_map não encontrado: {defect_map}")
        
        # Lê os arquivos como bytes
        with open(imagem_original, 'rb') as f:
            image_original_bytes = f.read()
        with open(defect_map, 'rb') as f:
            defect_map_bytes = f.read()

        # Converter para base64 para garantir compatibilidade
        image_original_b64 = base64.b64encode(image_original_bytes).decode('utf-8')
        defect_map_b64 = base64.b64encode(defect_map_bytes).decode('utf-8')
        
        print(f"📸 Imagem original: {len(image_original_bytes)} bytes")
        print(f"🔥 defect_map: {len(defect_map_bytes)} bytes")

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
            
            > USE ESTA LISTA COMO GUIA: Verifique se esses defeitos específicos aparecem nas áreas coloridas do defect_map.
            """

        try:
            prompt = f"""
                VOCÊ É UM PERITO FORENSE DIGITAL SÊNIOR.
                
                Sua tarefa é cruzar dados visuais e semânticos para explicar uma detecção de IA.
                
                DADOS DE ENTRADA:
                1. Imagem Original.
                2.  **Overlay (Capa de Chuva)**: É a imagem original contendo Uma NÉVOA / MANCHA, de cor {color_overlay}
                indicando as regiões que o detector considerou importantes.

                CONTEXTO GERAL:
                Classificação: "{classificacao_clip}" ({probabilidade_clip:.1%} de certeza).

                {texto_conceitos}
                    
                **DIRETRIZ DE SEGURANÇA (IMPORTANTE):**
                - A lista de conceitos acima é uma indicação do que o detector semântico encontrou.
                - Se a imagem for REAL, a tendencia é que a lista possa estar vazia ou conter "falsos positivos" (ruído). **NÃO INVENTE DEFEITOS** só para concordar com a lista.
                - Se a imagem for FAKE, a lista provavelmente indica o erro exato. Use-a como guia.

                INSTRUÇÃO: Responda em PORTUGUÊS, de forma técnica e direta.

                1. Análise da Cena: Descreva brevemente o sujeito e o ambiente da imagem original.
                2. Interpretação do defect_map: Explique o que as áreas coloridas do overlay indicam sobre o foco do modelo.
                3. Foco do defect_map: Onde estão concentrados os pontos coloridos no Overlay? (Olhos, mãos, pele, fundo?).
                4. Verificação de Defeitos: Olhando para a imagem original nessas áreas, você confirma a presença dos defeitos listados em {texto_conceitos}?
                . Veredito: Explique como a combinação do defect_map com os conceitos detectados confirma a classificação de "{classificacao_clip}".
            """
            
            # Envia as duas imagens para o LLaVA usando base64
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_original_b64, defect_map_b64]
                    }
                ]
            )
            
            print("✅ Análise concluída!\n")
            print("=" * 60)
            print("RESPOSTA DO LLaVA:")
            print("=" * 60)
            print(response['message']['content'])
            
            return response['message']['content']
            
        except Exception as e:
            print(f"Erro ao analisar imagens: {e}")
            return None