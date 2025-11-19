# Fine-Tuning Portfolio

This repository showcases practical, end-to-end notebooks for fine-tuning large language models **free Colab / Kaggle hardware**.  
The collection covers a wide range of modern techniques:

- LoRA adapters on all linear layers (**PEFT**)
- **QLoRA** with 4-bit quantization (**BitsAndBytes** NF4 + double quant)
- Supervised Fine-Tuning (**SFT**) and Reinforcement Learning (**GRPO**)
- Modern distributed training with **FSDP2** + Hugging Face **Accelerate** 
- Structured output, custom rewards, chain-of-thought, memory monitoring

All notebooks are designed to run with minimal VRAM while still delivering strong performance.

## Notebooks

### 1. Fine-Tuning Gemma-7B-It for Structured JSON Output & Financial Sentiment
- **Filename:** `SFT_Gemma_7B_it.ipynb`
- **Colab Link:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amanzoni1/llm_fine_tuning/blob/main/SFT_Gemma_7B_it.ipynb)
- **Description:**  
  Turns Gemma-7B-Instruct into a domain-specialized financial sentiment classifier with guaranteed clean JSON output. Deep exploration of Financial PhraseBank dataset and tokenizer behavior (lengths, vocab, tokenization), QLoRA + gradient checkpointing on one T4, plus detailed metrics (accuracy, F1, etc.) and temperature analysis for structured generation.
  
### 2. Reinforcement Learning – Llama3.1-8B with GRPO & Chain-of-Thought Reasoning
- **Filename:** `RL_LLama3_1_8B_GRPO.ipynb`
- **Colab Link:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amanzoni1/llm_fine_tuning/blob/main/RL_LLama3_1_8B_GRPO.ipynb)
- **Description:**  
  Two-phase training that turns LLaMA-3.1-8B into a powerful reasoning + summarization model. Phase 1: supervised fine-tuning on GSM8K (chain-of-thought). Phase 2: Group Relative Policy Optimization (GRPO) on XSum with custom rewards (ROUGE + BERTScore + format compliance). Uses Unsloth + 4-bit QLoRA for speed and low memory.

### 3. Distributed Training of Qwen2.5-7B-Instruct with FSDP2 and QLoRA
- **Filename:** `DT_Qwen2_5_7B_It_FSDP2.ipynb`
- **Run with dual T4 GPUs:** [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/amanzoni1/llm_fine_tuning/blob/main/DT_Qwen2_5_7B_It_FSDP2.ipynb)
- **Description:**  
  Production-ready template for multi-GPU QLoRA fine-tuning of Qwen2.5-7B-Instruct using the new FSDP2 (via Hugging Face Accelerate) on Kaggle’s free dual T4 setup. Combines 4-bit NF4 quantization + double quant, LoRA on all linear layers, full sharding, CPU offload, activation checkpointing, and a separate .py training script with detailed accelerate config.

---

> _Feel free to run, fork, or contribute · Pull requests very welcome!_
