# Why Do We Need AutoTokenizer?

Before learning about `AutoTokenizer`, we must first understand an important fact about Transformer models.

Many beginners think that a Transformer model can directly understand human language.

This is **not true**.

The Transformer architecture **does not contain a tokenizer**.

Its computation begins only after the input has been converted into numerical token IDs.

---

# Where Does the Transformer Actually Start?

The original Transformer architecture, introduced in the paper **"Attention Is All You Need" (2017)**, starts with the **Embedding Layer**.

```
Input Token IDs
       │
       ▼
Embedding Layer
       │
       ▼
Positional Encoding
       │
       ▼
Transformer Layers
       │
       ▼
Output
```

Notice something important.

The architecture expects **Token IDs** as input.

It does **not** accept raw text.

For example, the Transformer cannot process:

```text
I love Transformers
```

Instead, it expects something like:

```text
[1045, 2293, 19081]
```

This means that **tokenization happens before the Transformer begins its work**.

---

# Why Isn't Tokenization Part of the Transformer?

The Transformer is a neural network.

A neural network performs mathematical operations such as:

- Matrix multiplication
- Dot products
- Addition
- Softmax
- Linear transformations

These operations can only be performed on numbers.

Human language is text, not numbers.

Therefore, converting text into numbers is considered a **preprocessing step**, not a part of the Transformer architecture.

This preprocessing step is called **Tokenization**.

---

# A Challenge

If every Transformer model starts with an Embedding Layer, who converts text into Token IDs?

The answer is:

**The tokenizer.**

The tokenizer sits **outside** the Transformer.

```
Sentence

      │

      ▼

Tokenizer

      │

      ▼

Token IDs

      │

      ▼

Embedding Layer

      │

      ▼

Transformer
```

The Transformer never sees the original sentence.

It only receives numerical token IDs.

---

# Does Every Model Use the Same Tokenizer?

No.

This is one of the most important concepts to understand.

Every pretrained model is trained using its own tokenizer.

Different models use different tokenization algorithms, vocabularies, and token IDs.

For example:

| Model | Tokenizer Algorithm |
|---------|--------------------|
| BERT | WordPiece |
| GPT-2 | Byte Pair Encoding (BPE) |
| T5 | SentencePiece |
| LLaMA | SentencePiece |

Because these models were trained differently, they cannot share the same tokenizer.

---

# Example 1: BERT

Suppose we load the BERT model.

```python
model_name = "bert-base-uncased"
```

During training, BERT learned language using the **WordPiece tokenizer**.

Therefore, when we later write:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Hugging Face automatically downloads the **WordPiece tokenizer** that belongs to BERT.

It also downloads:

- Vocabulary
- Special tokens
- Tokenization rules
- Configuration files

This ensures that the input is converted into the exact Token IDs that BERT expects.

---

# Example 2: GPT-2

Now consider GPT-2.

```python
model_name = "gpt2"
```

GPT-2 was trained using **Byte Pair Encoding (BPE)**.

If we write:

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Hugging Face does **not** download the BERT tokenizer.

Instead, it automatically downloads GPT-2's BPE tokenizer.

Although the code is identical, the tokenizer loaded is completely different.

---

# Why Is This Important?

Imagine we accidentally use the BERT tokenizer with GPT-2.

The sentence would be converted into Token IDs that GPT-2 has never seen during training.

As a result:

- Words may be split differently.
- Token IDs may not match GPT-2's vocabulary.
- The embedding layer will receive incorrect IDs.
- The model's predictions will be poor or incorrect.

Therefore, **every pretrained model must always use the same tokenizer that was used during its training.**

---

# How Does AutoTokenizer Know Which Tokenizer to Load?

This is where Hugging Face simplifies our work.

When we execute:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

the following steps happen internally:

```
Model Name
      │
      ▼
"bert-base-uncased"
      │
      ▼
Read Model Configuration
      │
      ▼
Identify Tokenizer Type
      │
      ▼
Download the Correct Tokenizer
      │
      ▼
Create Tokenizer Object
```

If we change only the model name:

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

the process is the same, but Hugging Face downloads GPT-2's tokenizer instead.

The developer does not need to know whether the model uses WordPiece, BPE, or SentencePiece.

`AutoTokenizer` automatically detects and loads the correct tokenizer.

---

# Key Takeaways

- The Transformer architecture does **not** include tokenization.
- The Transformer begins with the **Embedding Layer**.
- Tokenization is a preprocessing step performed before the Transformer starts.
- Every pretrained model is trained with its own tokenizer.
- Different models use different tokenization algorithms and vocabularies.
- `AutoTokenizer.from_pretrained()` automatically loads the correct tokenizer for the selected model.
