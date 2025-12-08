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

    def analisar_imagens(self, imagem_original, defect_map, classificacao_clip, probabilidade_clip, conceitos_detectados=None, color_overlay="vermelho"):
        """
        Envia imagem original + defect_map + conceitos semânticos para o Nemotron.
        """

        print("Carregando imagens...")
        try:
            img1_b64 = self._carregar_imagem_base64(imagem_original)
            img2_b64 = self._carregar_imagem_base64(defect_map)
        except Exception as e:
            print(f"Erro ao carregar imagens: {e}")
            return None

        print("Imagens carregadas. Preparando prompt com conceitos...")

        # --- 1. PREPARAR A LISTA DE CONCEITOS ---
        texto_conceitos = "Nenhum defeito específico listado pelo detector semântico."
        if conceitos_detectados:
            # Pega os top 5
            top_conceitos = list(conceitos_detectados.items())[:5]
            lista_str = "\n".join([f"   - '{k}' ({v:.1%} de sinal)" for k, v in top_conceitos])
            
            texto_conceitos = f"""
            ALERTA DE ANÁLISE SEMÂNTICA (IMPORTANTE):
            O detector identificou os seguintes padrões de defeito nesta imagem:
            {lista_str}
            
            > USE ESTA LISTA COMO GUIA: Verifique se esses defeitos específicos aparecem nas áreas coloridas do defect_map.
            """

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
        
        # -------- REQUISIÇÃO OPENROUTER --------
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Megatruth Analyzer"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                    ]
                }
            ]
        }

        # -------- ENVIO --------
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=90) # Aumentei timeout para 90s
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["message"]["content"]
                print("\n=== RESPOSTA DO NEMOTRON VL ===\n")
                print(result)
                print("\n================================\n")
                return result
            else:
                print(f"Resposta inesperada da API: {data}")
                return None

        except Exception as e:
            print(f"Erro ao enviar para o Nemotron: {e}")
            return None