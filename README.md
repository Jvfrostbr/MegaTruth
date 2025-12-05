
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

## 💡 Roadmap do MegaTruth (Checklist)

Aqui está o *roadmap* do MegaTruth formatado como uma lista de verificação (checklist), detalhando os subtópicos e entregáveis para cada módulo planejado.

---

### 1. GUI (Gradio/Streamlit)

Criação da interface de usuário **simples e funcional** para demonstrações e usabilidade.

* [x] **Design e Estrutura Inicial (MVP):**
    * [x] Definir o *framework* de UI (Gradio/Streamlit).
    * [x] Implementar o componente de **Upload de Imagem** (`PNG`, `JPG`).
* [x] **Módulo de Saída Principal:**
    * [x] Exibir **Rótulo de Classificação** (`Real` vs `IA`) e **Confiança**.
    * [x] Área dedicada à visualização do **Heatmap** (Grad-CAM).
    * [x] Caixa de texto para a **Explicação Textual** (saída do LLaVA).
* [ ] **Funcionalidades Adicionais:**
    * [ ] Criar um **Histórico Simples** de análises da sessão.

---

### 2. Finetuning do CLIP

Melhoria da precisão e **robustez** do classificador CLIP para o domínio *real vs IA*.

* [ ] **Preparação do Dataset Especializado:**
    * [ ] Curadoria de um **dataset balanceado** (Real vs. IA de múltiplos modelos generativos).
    * [ ] Implementar **Estratégia de Aumento de Dados** (*Data Augmentation*) simulando compressão (JPEG) e ruído.
* [ ] **Implementação do Finetuning (LoRA):**
    * [ ] Selecionar o *backbone* CLIP e definir a **arquitetura LoRA**.
    * [ ] Treinar o modelo utilizando LoRA e definir hiperparâmetros (taxa de aprendizado, épocas).
* [ ] **Avaliação e Comparação:**
    * [ ] Estabelecer a **linha de base (*baseline*)** do CLIP sem *finetuning*.
    * [ ] Avaliar o modelo *finetunado* em métricas como **Acurácia, AUC e F1-Score**.
* [ ] **Adaptação do Heatmap:**
    * [ ] Verificar a coerência do **Grad-CAM** após o *finetuning*.

---

### 3. Concept Bottleneck (Explicabilidade Profunda)

Fornecer explicações intermediárias baseadas em **conceitos semânticos e visuais** de artefatos. 

[Image of a Concept Bottleneck Model diagram showing input, concept layer, and output]


* [ ] **Definição de Conceitos:**
    * [ ] Definir uma ontologia de **artefatos de IA** e **inconsistências visuais** (ex: "Dedos Deformados", "Textura Irregular").
    * [ ] Rotular um subconjunto do *dataset* com a **presença/ausência** desses conceitos.
* [ ] **Desenvolvimento do CBM:**
    * [ ] Treinar um **modelo auxiliar leve** para **prever a probabilidade de cada conceito** (Gargalo Conceitual).
* [ ] **Integração ao LLaVA:**
    * [ ] Modificar o *prompt* do LLaVA para incluir a **Lista de Conceitos Preditos**.
    * [ ] Instruir o LLaVA a **incorporar esses conceitos** na explicação textual.

---

### 4. Chatbot Explicativo

Transformar a explicação estática em uma **interação dinâmica** sobre a análise e as evidências.

* [ ] **Estrutura de Diálogo:**
    * [ ] Implementar o rastreamento do **histórico de conversas** (*history buffer*).
    * [ ] Definir a **memória curta** focada na imagem atual e análise.
* [ ] **JSON Estruturado de Saída:**
    * [ ] Garantir que a saída inicial do LLaVA esteja em formato **JSON** com dados chave (`rótulo`, `evidências`, `regiões`).
* [ ] **Prompts Multimodais para Conversa:**
    * [ ] Criar *templates* de *prompt* para o LLaVA que respondam a perguntas comuns, utilizando o **JSON e a Imagem/Heatmap** como contexto.
* [ ] **Testes de Coerência:**
    * [ ] Realizar testes para garantir que o Chatbot **não alucine informações** sobre o Heatmap ou a classificação.
