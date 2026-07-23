# AutoModel

After converting text into Token IDs using `AutoTokenizer`, the next step is to load the actual Transformer model.

This is done using the `AutoModel` class.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
```

At first glance, this looks like a single line of code.

However, a lot happens internally.

---

# Why Do We Need AutoModel?

Suppose we already have the token IDs.

```
Sentence

↓

AutoTokenizer

↓

[101, 1045, 2293, 2023, 102]
```

Can these numbers directly give us predictions?

No.

These numbers are only integer IDs.

They still need to pass through the Transformer model.

The Transformer performs:

- Embedding Lookup
- Positional Encoding
- Self-Attention
- Feed Forward Networks
- Layer Normalization
- Residual Connections

to produce meaningful representations.

Therefore, we need to load the Transformer model itself.

That is the purpose of `AutoModel`.

---

# What Does AutoModel Do?

`AutoModel` automatically loads the **base Transformer architecture** along with its pretrained weights.

For example,

```python
model = AutoModel.from_pretrained("bert-base-uncased")
```

Hugging Face automatically downloads:

- Model configuration
- Transformer architecture
- Pretrained weights
- Model parameters

and constructs the complete neural network.

---

# What Does "Pretrained" Mean?

Before being released, BERT was trained by Google on massive amounts of text.

During training, the model learned:

- Grammar
- Sentence structure
- Word relationships
- Language patterns
- Context

All this knowledge is stored inside millions of learned parameters called **weights**.

Instead of training again from scratch, we simply download these learned weights.

That is why it is called a **pretrained model**.

---

# How Does AutoModel Work Internally?

When we execute:

```python
model = AutoModel.from_pretrained("bert-base-uncased")
```

the following steps happen internally.

```
Model Name

↓

"bert-base-uncased"

↓

Search Hugging Face Hub

↓

Download Configuration

↓

Build BERT Architecture

↓

Download Pretrained Weights

↓

Load Weights into the Architecture

↓

Return Model Object
```

The developer never has to manually build the Transformer.

---

# What Does AutoModel Return?

This is one of the most misunderstood concepts.

`AutoModel` **does not perform classification, translation, or text generation.**

Instead, it returns the **base Transformer model**.

Its output is called **Hidden States**.

```
Sentence

↓

AutoTokenizer

↓

Token IDs

↓

AutoModel

↓

Hidden States
```

These hidden states are rich numerical representations of the input text.

They capture the meaning and context learned during pretraining.

---

# What Are Hidden States?

Suppose our sentence is:

```text
I love Transformers.
```

After tokenization:

```
[1045, 2293, 19081]
```

These IDs enter the Transformer.

Inside the Transformer:

- Embeddings are created.
- Attention layers exchange information.
- Feed Forward Networks refine the representations.
- This process repeats through multiple Transformer layers.

Finally, each token is represented by a dense vector.

Example:

```
Token          Hidden Representation

I         → [0.12, -0.41, 0.85, ...]

love      → [-0.77, 0.33, 0.11, ...]

Transformers → [0.58, 1.02, -0.64, ...]
```

These vectors are called **Hidden States**.

They contain contextual information about each token.

---

# Why Doesn't AutoModel Produce Predictions?

Because the base Transformer was pretrained to learn language representations, **not a specific downstream task**.

For example:

- Sentiment Analysis
- Question Answering
- Text Generation
- Named Entity Recognition

Each task requires an additional output layer.

Therefore, `AutoModel` stops after producing hidden states.

---

# Complete Flow

```
Sentence

↓

AutoTokenizer

↓

Token IDs

↓

Embedding Layer

↓

Transformer Layers

↓

Hidden States
```

Notice that the pipeline stops at **Hidden States**.

No prediction has been made yet.

---

# Example

```python
from transformers import AutoTokenizer
from transformers import AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("I love Transformers.", return_tensors="pt")

outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
```

Example output:

```
torch.Size([1, 5, 768])
```

Meaning:

- 1 sentence
- 5 tokens
- Each token represented by a 768-dimensional vector

These are the hidden states produced by BERT.

---

# Key Takeaways

- `AutoModel` loads the base Transformer model.
- It automatically downloads the correct architecture and pretrained weights.
- The model starts processing from the Embedding layer.
- The output of `AutoModel` is **Hidden States**, not predictions.
- Hidden states are contextual vector representations learned by the Transformer.
- Additional task-specific layers are required for classification, question answering, or text generation.
