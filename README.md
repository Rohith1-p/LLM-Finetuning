## 🔧 Fine-Tuning Large Language Models (LLMs)
Finetune a Opensource LLM and optimizing the training 

The Google Colab files in this repository contain various open-source LLMs, fine-tuned using datasets from Hugging Face. Most models are under 7 billion parameters, as larger models require substantial GPU power, which can be costly. Since free GPU access is available on Google Colab and Kaggle Notebooks, I used Google Colab for training models under the 7 billion parameter limit.

These are some of the Opensource LLMs used in this respository. 
1. Llama
2. PHi-3
3. Mistral
4. There are other LLMs as well in my drive. I will add them when i get some free time. 


Fine-tuning large language models (LLMs) allows us to adapt powerful pre-trained models (like GPT, T5, LLaMA, etc.) to specific downstream tasks or domains. However, due to their large size, full fine-tuning of LLMs can be computationally expensive and inefficient. This is where **parameter-efficient fine-tuning (PEFT)** methods like **LoRA (Low-Rank Adaptation)** come in.

### 🧠 What is LoRA?

**LoRA** (Low-Rank Adaptation) is a fine-tuning method that freezes the original weights of a large model and instead adds a small number of trainable parameters through low-rank matrices. This drastically reduces the number of parameters updated during training, making fine-tuning:

* **Faster**
* **Cheaper**
* **More memory-efficient**

LoRA has become one of the go-to methods for adapting LLMs to custom tasks with limited resources.

### 🔍 Why Use LoRA?

* 🚀 **Reduces GPU memory usage**
* 💸 **Cuts down training costs**
* 🔁 **Enables frequent updates and retraining**
* 🧩 **Modular and easy to integrate with Hugging Face Transformers**

### 🔬 How It Works

Instead of updating the full weight matrix `W` in a layer, LoRA decomposes the update into two smaller matrices:

```
W' = W + ΔW  where ΔW = A × B, with A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), and r << d,k
```

This approximation allows us to learn meaningful task-specific changes while keeping most of the model parameters frozen.

### 🛠️ Tools and Frameworks Used

* **Hugging Face Transformers**
* **PEFT Library by Hugging Face**
* **BitsAndBytes** (for quantized training like QLoRA)
* **PyTorch**
* **Accelerate & Transformers Trainer**
* Optional: **Weights & Biases**, **TensorBoard** for experiment tracking

### 📂 Use Cases Covered

* Fine-tuning **LLaMA**, **T5**, and **Phi-3** on domain-specific datasets
* Applying **LoRA** for topic classification, chatbot intent understanding, and cognitive health assessments
* Using **QLoRA** for quantized training of large models with limited hardware
* Integrating **LoRA-adapted models** into RAG pipelines

