# AutoModelForSequenceClassification

## What is AutoModelForSequenceClassification?

`AutoModelForSequenceClassification` is a Hugging Face class that automatically loads a **pretrained Transformer model** (such as BERT, RoBERTa, or DistilBERT) and **attaches a sequence classification head** on top of it.

It is used for tasks where an **entire sentence or document** belongs to one category.

Examples:

- Sentiment Analysis
- Spam Detection
- Fake News Detection
- Topic Classification
- Intent Classification

---

# Why Do We Need AutoModelForSequenceClassification?

Earlier, we learned about `AutoModel`.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
```

`AutoModel` loads only the **base Transformer architecture** with pretrained weights.

```
Sentence
     ↓
Tokenizer
     ↓
Token IDs
     ↓
BERT
(Embedding + Transformer Layers)
     ↓
Hidden States
```

The output of `AutoModel` is **hidden states**.

It does **not** predict:

- Positive / Negative
- Spam / Not Spam
- Sports / Politics
- Any other class

It only produces contextual representations of the input.

---

# The Problem

Suppose we want to classify a review.

```
"This movie is amazing!"
```

The Transformer understands the sentence and produces hidden states.

But hidden states are just numerical vectors.

```
[0.21, -0.45, 1.02, ..., 0.38]
```

A user does not want vectors.

The user wants:

```
Positive
```

or

```
Negative
```

So we need another neural network that converts hidden states into class predictions.

---

# Solution

Hugging Face provides:

```python
AutoModelForSequenceClassification
```

This class automatically builds:

```
                BERT
                     Sentence
                         │
                         ▼
                    AutoTokenizer
                         │
                         ▼
                 Input IDs + Attention Mask
                         │
                         ▼
                   Embedding Layer
                         │
                         ▼
                 Transformer Layer 1
                         │
                         ▼
                 Transformer Layer 2
                         │
                         ▼
                        ...
                         │
                         ▼
                Transformer Layer 12
                         │
                         ▼
                   Hidden States
                         │
                         ▼
             Sequence Classification Head
                         │
                         ▼
                    Linear Layer
                         │
                         ▼
                 Logits (Raw Scores)
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
  torch.argmax()                Softmax (Optional)
          │                             │
          ▼                             ▼
      Class ID                  Probabilities
          │                             │
          └──────────────┬──────────────┘
                         ▼
              model.config.id2label
                         │
                         ▼
            Final Prediction (Positive / Negative)
```

Now the model can classify text.

---

# Is It a New BERT?

This is one of the most common beginner questions.

Many people think:

```
AutoModel

↓

AutoModelForSequenceClassification
```

This is **NOT** how it works.

Suppose we write:

```python
bert = AutoModel.from_pretrained("bert-base-uncased")
```

and later:

```python
classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased"
)
```

These are **two independent Python objects**.

The second model **does not reuse** the first object.

Instead, Hugging Face builds another model internally.

```
Object 1

BERT
```

```
Object 2

BERT
+
Classification Head
```

Normally, we **do not load both**.

---

# What Happens in Real Applications?

If your goal is text classification, you simply write:

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
```

Notice that we never load `AutoModel`.

Why?

Because `AutoModelForSequenceClassification` already contains the Transformer.

---

# Internal Architecture

Internally, it looks similar to this:

```python
class BertForSequenceClassification(nn.Module):

    def __init__(self):

        self.bert = BertModel(...)

        self.classifier = nn.Linear(768, num_labels)
```

Notice something important.

```
self.bert
```

is the pretrained Transformer.

```
self.classifier
```

is a new neural network added for classification.

---

# Does the Classification Head Replace BERT?

No.

The classification head is connected to BERT.

```
Sentence
      ↓
Tokenizer
      ↓
BERT
      ↓
Hidden States
      ↓
Classification Head
      ↓
Prediction
```

The classification head cannot work without BERT because it needs the hidden states produced by the Transformer.

---

# What Does the Classification Head Contain?

The classification head is usually very small.

```
Hidden States
      ↓
CLS Token (or pooled output)
      ↓
Linear Layer
      ↓
Logits
      ↓
Softmax
      ↓
Predicted Class
```

The Linear layer converts the hidden representation into class scores.

Softmax converts these scores into probabilities.

Example:

```
Positive : 97%

Negative : 3%
```

---

# Why Doesn't AutoModel Predict Classes?

The pretrained Transformer was trained to understand language.

It was **not trained** specifically for:

- Sentiment Analysis
- Spam Detection
- Topic Classification

Therefore, it only learns rich language representations.

Different tasks require different prediction layers.

---

# Hugging Face Design Philosophy

Instead of creating a different Transformer for every task, Hugging Face reuses the same pretrained Transformer and adds a task-specific head.

```
Requirement

        │

        ▼

Base Transformer

        │

        ▼

Task-Specific Head

        │

        ▼

Prediction
```

---

# Examples

## Feature Extraction

```
AutoModel
```

Output:

```
Hidden States
```

---

## Sentiment Analysis

```
AutoModelForSequenceClassification
```

Output:

```
Positive
```

---

## Named Entity Recognition

```
AutoModelForTokenClassification
```

Output:

```
John → PERSON

London → LOCATION
```

---

## Question Answering

```
AutoModelForQuestionAnswering
```

Output:

```
Start Position

End Position
```

---

## Text Generation

```
AutoModelForCausalLM
```

Output:

```
Next Word Prediction
```

---

# Key Takeaways

- `AutoModel` loads only the pretrained Transformer.
- It produces hidden states but does not make task-specific predictions.
- `AutoModelForSequenceClassification` contains the pretrained Transformer plus a classification head.
- The classification head converts hidden states into class predictions.
- For text classification, we directly use `AutoModelForSequenceClassification`.
- We do **not** need to load `AutoModel` separately.
- Hugging Face selects the appropriate architecture based on the task by attaching the required prediction head to the base Transformer.
