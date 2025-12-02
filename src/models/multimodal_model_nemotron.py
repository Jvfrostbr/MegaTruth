import os
import base64
import requests


class NemotronVL:
    def __init__(self):
        self.model_name = "nvidia/nemotron-nano-12b-v2-vl:free"
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("A variável de ambiente OPENROUTER_API_KEY não está definida!")

        print(f"Usando modelo: {self.model_name}")

    def _carregar_imagem_base64(self, caminho):
        """Lê imagem como bytes e converte para base64."""
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        with open(caminho, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analisar_imagens(self, imagem_original, heatmap, classificacao_clip, probabilidade_clip):
        """
        Envia imagem original + heatmap para o Nemotron VL Free
        e retorna uma análise multimodal detalhada em português.
        """

        print("Carregando imagens...")

        img1_b64 = self._carregar_imagem_base64(imagem_original)
        img2_b64 = self._carregar_imagem_base64(heatmap)

        print("Imagens carregadas. Enviando para o Nemotron...")

        # -------- PROMPT MULTIMODAL --------
        prompt = f"""
            VOCÊ É UM PERITO FORENSE DIGITAL.
            
            Sua tarefa é analisar DUAS imagens:
            1. A Imagem Original.
            2. O Overlay (Sobreposição): Um mapa de calor + sobreposto com a imagem original, 
            onde contém pontos COLORIDOS (Verde/Amarelo/Ciano) indicam suspeita de IA.

            CONTEXTO:
            O detector (CLIP) classificou como: "{classificacao_clip}" ({probabilidade_clip:.1%} de certeza).

            INSTRUÇÃO:
            Responda em PORTUGUÊS.

            1. O que você vê na imagem original? (Descrição breve).
            2. Onde estão os pontos coloridos/brilhantes no Overlay? (Ex: Olhos, pele, cabelo, fundo).
            3. Comparando com a original: Nessas áreas destacadas, existe alguma textura estranha, pele muito lisa ou falha visual?
            4. Conclusão: Por que esses detalhes visuais suportam a classificação de "{classificacao_clip}"?
            """

        # -------- REQUISIÇÃO - FORMATO CORRETO DO OPENROUTER --------
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Nemotron VL Analyzer"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},

                        # → FORMATO CORRETO PARA IMAGENS
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img1_b64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img2_b64}"
                            }
                        }
                    ]
                }
            ]
        }

        # -------- ENVIO --------
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            result = data["choices"][0]["message"]["content"]

            print("\n=== RESPOSTA DO NEMOTRON VL ===\n")
            print(result)
            print("\n================================\n")

            return result

        except Exception as e:
            print(f"Erro ao enviar para o Nemotron: {e}")
            try:
                print("Resposta bruta da API:", response.text)
            except:
                pass
            return None
