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
- # AutoTokenizer vs TensorFlow/Keras Tokenizer

If you have previously worked with TensorFlow or Keras, you may wonder:

```python
from tensorflow.keras.preprocessing.text import Tokenizer
```

and

```python
from transformers import AutoTokenizer
```

Both are called **Tokenizer**, but they work very differently.

---

# TensorFlow/Keras Tokenizer

Suppose we have two sentences.

```python
sentences = [
    "I love this phone",
    "This movie is terrible"
]
```

We create a tokenizer.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
```

The tokenizer reads our dataset and **creates its own vocabulary**.

Example:

```
I         → 5
love      → 3
this      → 1
phone     → 7
movie     → 2
terrible  → 6
is        → 4
```

Now,

```python
tokenizer.texts_to_sequences(sentences)
```

returns

```
[[5,3,1,7],
 [1,2,4,6]]
```

Notice something important.

The vocabulary is created **from our own dataset**.

If we train on another dataset, the IDs may change.

Example

Dataset A

```
cat → 1
dog → 2
```

Dataset B

```
dog → 1
cat → 2
```

The tokenizer builds a new vocabulary every time.

---

# Hugging Face AutoTokenizer

Now consider:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

This tokenizer does **not create a vocabulary**.

Instead, it downloads the vocabulary that was used when BERT was pretrained.

Example:

```
I      → 1045
love   → 2293
this   → 2023
phone  → 3042
```

These IDs are **fixed**.

Anyone who downloads BERT will get exactly the same vocabulary.

---

# Why Can't We Use the Keras Tokenizer with BERT?

Suppose Keras creates:

```
love → 3
phone → 7
```

But BERT expects:

```
love → 2293
phone → 3042
```

If we send

```
[3,7]
```

to BERT, it will not work correctly because BERT's embedding layer was trained using IDs:

```
2293
3042
```

not

```
3
7
```

The tokenizer and the pretrained model must use the **same vocabulary**.

---

# Why AutoTokenizer is Required

During BERT pretraining:

```
Tokenizer
      ↓
Vocabulary
      ↓
Embedding Layer
      ↓
Transformer
```

The tokenizer and the Transformer learned together.

Therefore, whenever we download a pretrained model, we must also download its matching tokenizer.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

This guarantees that the token IDs perfectly match the embedding layer inside BERT.

---

# Visual Comparison

## Traditional NLP

```
Your Dataset
      ↓
Keras Tokenizer
(Creates New Vocabulary)
      ↓
Token IDs
      ↓
Your Neural Network
```

---

## Transformer NLP

```
Pretrained Model
      ↓
Pretrained Vocabulary
      ↓
AutoTokenizer
      ↓
Correct Token IDs
      ↓
Embedding Layer
      ↓
Transformer
```

---

# Key Difference

| TensorFlow/Keras Tokenizer | Hugging Face AutoTokenizer |
|----------------------------|----------------------------|
| Creates a vocabulary from your dataset | Loads the pretrained model's vocabulary |
| Token IDs depend on your data | Token IDs are fixed |
| Used when training your own model | Used with pretrained Transformer models |
| Vocabulary changes for different datasets | Vocabulary never changes for a given pretrained model |
| Does not match BERT embeddings | Perfectly matches the pretrained model's embedding layer |

---

# Key Takeaways

- `Tokenizer` from TensorFlow/Keras creates a new vocabulary from your dataset.
- `AutoTokenizer` downloads the vocabulary that was used to pretrain the Transformer model.
- The token IDs generated by `AutoTokenizer` exactly match the embedding layer of the pretrained model.
- Using a different tokenizer with a pretrained Transformer will produce incompatible token IDs and lead to incorrect model behavior.

### `return_tensors`

By default, the tokenizer returns Python lists. The `return_tensors` parameter converts these outputs into tensors that can be passed directly to a deep learning model.

```python
inputs = tokenizer(text, return_tensors="pt")
```

- `"pt"` → Returns PyTorch (`torch.Tensor`) objects.
- `"tf"` → Returns TensorFlow (`tf.Tensor`) objects (supported in Transformers versions that include TensorFlow support).
- `"np"` → Returns NumPy arrays.

> **Note:** See the **PyTorch Tensors vs TensorFlow Tensors** chapter for a detailed explanation of framework-specific tensors.

---

### What Does the Tokenizer Return?

The tokenizer returns a **BatchEncoding** object, which is a dictionary-like container holding one or more encoded inputs required by the transformer model.

Example:

```python
inputs = tokenizer("I love India", return_tensors="pt")
```

Output:

```python
{
    'input_ids': tensor([[101, 1045, 2293, 2634, 102]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1]])
}
```

The most common fields are:

- **`input_ids`** → Tensor containing the numerical IDs of the input tokens.
- **`attention_mask`** → Tensor indicating which tokens are real (`1`) and which are padding (`0`).
- **`token_type_ids`** *(optional)* → Tensor identifying which sentence each token belongs to in sentence-pair tasks (used by models such as BERT).
