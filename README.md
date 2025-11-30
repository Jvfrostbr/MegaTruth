
# **MegaTruth**

MegaTruth é um sistema integrado de análise de imagens projetado para **detectar se uma imagem é real ou gerada por IA**, oferecendo **explicações visuais e textuais** por meio de heatmaps (CLIP) e modelos multimodais (LLaVA).
O foco do projeto é unir **classificação**, **detecção de inconsistências visuais** e **interpretação auditável**.

---

## **Visão Geral do Sistema**

O MegaTruth combina dois componentes principais:

### **1. CLIP + GradCAM → Classificação e Heatmap**

* Detecta se a imagem tende a ser *real* ou *IA*.
* Gera um **heatmap explicável**, destacando regiões relevantes para a decisão.
* Normaliza e salva o mapa em um arquivo `.png`.

### **2. LLaVA → Explicação Multimodal**

* Recebe:

  * imagem original
  * heatmap
  * rótulo previsto
  * confiança
* Produz uma **explicação textual coerente**, descrevendo:

  * evidências visuais
  * padrões suspeitos
  * fatores que influenciaram a classificação
  * incertezas e limitações

O MegaTruth não apenas diz *o que* a imagem parece ser — mas *por quê*.

---

## **Funcionalidades Principais**

### ✔ **Classificação Real vs IA (CLIP)**

Modelos CLIP pré-treinados, com possibilidade de finetuning.

### ✔ **Heatmap Explicável (GradCAM)**

Localiza regiões que motivaram a decisão do modelo.

### ✔ **Explicação Textual (LLaVA)**

Relatórios claros e coerentes sobre as evidências visuais.

### ✔ **Pipeline Integrado**

CLIP → Heatmap → LLaVA → Resposta estruturada.

### ✔ **Integração Flexível**

Implementado em Python, com classes separadas por modelo.

---

# **Estrutura do Projeto**

```
MegaTruth/
│
├── src/
│   ├── models/
│   │   ├── vision_model.py     # CLIP + GradCAM
│   │   ├── llava_model.py      # LLaVA multimodal
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── heatmap_utils.py
│       └── image_processing.py
│
├── images/
│   └── exemplo.jpg
│
├── main.py                      # Pipeline CLIP → LLaVA
└── README.md
```

---

# **Como Usar**

1. Coloque uma imagem em `images/exemplo.jpg`.
2. Execute:

```bash
python main.py
```

3. O sistema irá:

* classificar a imagem
* gerar o heatmap
* criar uma explicação
* exibir tudo no terminal

---

# **Requisitos**

* Python 3.10+
* PyTorch
* Transformers
* Pillow
* NumPy
* Matplotlib

Instalação:

```bash
pip install -r requirements.txt
```

---

# **Roadmap do Projeto**

Focado nas prioridades estratégicas para tornar o MegaTruth mais preciso, explicável e acessível.

## [ ] **GUI (Gradio/Streamlit)**

**Objetivo:** interface simples e funcional para demonstrações.

**Inclui:** upload, heatmap, explicação, comparação e histórico.
<!-- **Dificuldade:** ⭐⭐
**Impacto:** ⭐⭐⭐. -->

## [ ] **Finetuning do CLIP**

**Objetivo:** melhorar a precisão no domínio *real vs IA*.

**Inclui:** dataset especializado, LoRA, comparação com baseline, heatmap adaptado.
 <!--**Dificuldade:** ⭐⭐⭐ -->
<!-- **Impacto:** 🚀 altíssimo. -->

## [ ] **Concept Bottleneck (Explicabilidade Profunda)**

**Objetivo:** criar explicações intermediárias baseadas em conceitos visuais.

**Inclui:** definição de conceitos, modelo preditor, integração ao LLaVA.
<!-- **Dificuldade:** ⭐⭐⭐⭐ -->
<!-- **Impacto:** 🔥 muito alto. -->


## [ ] **Chatbot Explicativo**

**Objetivo:** conversar sobre a análise e suas evidências.

**Inclui:** JSON estruturado, prompts multimodais, histórico de conversa.
<!--**Dificuldade:** ⭐⭐
**Impacto:** ⭐⭐–⭐⭐⭐. -->
