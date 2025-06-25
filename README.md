# Deception-detection


# 🤖 Deception Detection in Diplomacy Dialogues  
**Truthfulness Classification in Multi-Agent Language via Hierarchical BiLSTM-BERT**


---

## 🧠 Project Summary
This project focuses on detecting **deceptive language** in the strategic game **Diplomacy**, where players often use persuasion and lying as core tactics.

By modeling these interactions, this work explores how AI systems can be trained to **detect untruthful intent in complex, multi-agent conversations** — a crucial component of **AI Safety**, especially for developing **aligned, trustworthy language models**.

📍 Based on the [QANTA Diplomacy Dataset](https://sites.google.com/view/qanta/projects/diplomacy)

---

## 🚀 Why It Matters for AI Safety

Modern language models can convincingly generate false information. This project contributes to:
- 🔍 **Detecting strategic deception** in agent communication  
- 📉 Improving **truthfulness calibration** in multi-agent systems  
- 🛡️ Building models capable of **resisting misuse or manipulation**

It aligns with goals of OpenAI, Anthropic, and Open Philanthropy around developing **truthful, interpretable, and robust AI**.


---

## 📊 Final Test Results

| Metric                | Value     |
|----------------------|-----------|
| **Loss**             | 0.2831    |
| **Accuracy**         | 76.1%     |
| **Macro F1**         | 0.5950    |
| **Truthful F1**      | 0.8546    |
| **Deceptive F1**     | 0.3354    |
| **False Negatives**  | 579       |
| **False Positives**  | 75        |

> The model detects truthful messages well but struggles with rare, subtle deceptive ones — highlighting the importance of **robust minority class handling** in high-stakes AI deployment.

---

## Architecture Highlights

- **Hierarchical BiLSTM with BERT embeddings**
- **Dialogue-aware attention**
- Rich **strategic metadata**: seasons, speaker deception history, game phase, etc.
- **Focal Loss & Weighted Sampling** to handle severe class imbalance
- Optional **CRF head for sequence modeling**
- The dataset is highly imbalanced, with only 4.5% of messages labeled deceptive and 95.5% labeled truthful, reflecting the natural sparsity of strategic lies in gameplay communication.


---

## 🔧 Setup

Here's your environment:

```bash
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas matplotlib seaborn tqdm
pip install torchcrf

