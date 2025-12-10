import os
import base64
import requests
from PIL import Image
from io import BytesIO
import time

class NemotronVL:
    def __init__(self):
        self.model_name = "nvidia/nemotron-nano-12b-v2-vl:free"
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("A variável de ambiente OPENROUTER_API_KEY não está definida!")

        print(f"Usando modelo: {self.model_name}")

    def _carregar_imagem_base64(self, caminho, resize_before_send=True, max_side=2000, quality=85, save_resized_path=None):
        """Lê imagem, opcionalmente redimensiona/comprime e converte para base64.

        Args:
            caminho (str): caminho do arquivo original.
            resize_before_send (bool): se True, redimensiona imagem mantendo proporção
                para que o maior lado <= `max_side` e converte para JPEG com `quality`.
            max_side (int): maior dimensão em pixels permitida antes de redimensionar.
            quality (int): qualidade JPEG (0-100) para compressão em memória.
            save_resized_path (str|None): caminho para salvar a versão redimensionada (opcional).
        Returns:
            str: string base64 pronta para inclusão em payload.
        """
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        # Se não for para redimensionar, apenas retorna o base64 do arquivo original
        if not resize_before_send:
            with open(caminho, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # Abrir imagem e preparar para envio
        try:
            img = Image.open(caminho)
        except Exception as e:
            raise RuntimeError(f"Falha ao abrir imagem {caminho}: {e}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_side:
            ratio = float(max_side) / float(max(w, h))
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"⤵️ Redimensionado para envio: {new_w}x{new_h} (max_side={max_side})")

        # Salvar em JPEG na memória
        buffer = BytesIO()
        try:
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
        except OSError:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

        bytes_data = buffer.getvalue()

        # Opcional: salvar versão redimensionada em disco para auditoria
        if save_resized_path:
            try:
                with open(save_resized_path, "wb") as outf:
                    outf.write(bytes_data)
                print(f"🔖 Versão redimensionada salva em: {save_resized_path}")
            except Exception as e:
                print(f"⚠️ Falha ao salvar versão redimensionada: {e}")

        return base64.b64encode(bytes_data).decode("utf-8")

    def analisar_imagens(self, imagem_original, defect_map, classificacao_clip, probabilidade_clip, conceitos_detectados=None, color_overlay="vermelho", resize_images=True, max_side=2000, quality=85):
        """
        Envia imagem original + defect_map + conceitos semânticos para o Nemotron.
        """

        print("Carregando imagens...")
        try:
            # Gerar nomes para salvar versões redimensionadas (opcional)
            resized1_path = None
            resized2_path = None
            # Se quisermos manter rastreio, salvamos em outputs/temp
            tmp_dir = os.path.join(os.getcwd(), "outputs", "temp")
            os.makedirs(tmp_dir, exist_ok=True)

            if resize_images:
                resized1_path = os.path.join(tmp_dir, f"resized_{int(time.time()*1000)}_1.jpg")
                resized2_path = os.path.join(tmp_dir, f"resized_{int(time.time()*1000)}_2.jpg")

            img1_b64 = self._carregar_imagem_base64(imagem_original, resize_before_send=resize_images, max_side=max_side, quality=quality, save_resized_path=resized1_path if resize_images else None)
            img2_b64 = self._carregar_imagem_base64(defect_map, resize_before_send=resize_images, max_side=max_side, quality=quality, save_resized_path=resized2_path if resize_images else None)
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
            3. Caso o overlay não contenha manchas de cor {color_overlay} e que o overlay é idêntico a imagem original,
            considere que o detector não encontrou áreas relevantes, e nesse caso vc pode pular a pergunta  "3. Foco do defect_map".

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