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

# Understanding `num_labels` in AutoModelForSequenceClassification

One of the most confusing concepts for beginners is understanding why the model sometimes predicts only **2 classes** even when the dataset contains **3 classes**.

Let's understand this step by step.

---

# Is `distilbert-base-uncased` a Binary Classification Model?

**No.**

`distilbert-base-uncased` is **not** a sentiment classification model.

It is a **pretrained language model**.

During pretraining, it learns:

- English grammar
- Vocabulary
- Sentence structure
- Contextual word representations
- Relationships between words

It does **not** learn any downstream task such as:

- Sentiment Analysis
- Spam Detection
- News Classification
- Topic Classification

Its purpose is to understand language, not classify it.

---

# What happens when we use AutoModelForSequenceClassification?

When we write

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)
```

Hugging Face automatically adds a **classification head** on top of the pretrained DistilBERT encoder.

```
Input Text
      │
      ▼
DistilBERT Encoder
      │
      ▼
Classification Head
      │
      ▼
Predicted Class
```

The DistilBERT encoder comes from the pretrained checkpoint.

The classification head is created specifically for sequence classification.

---

# Why does the model show `num_labels = 2`?

If we execute

```python
print(model.config.num_labels)
```

the output is

```python
2
```

Many beginners think this means

> "DistilBERT is a binary classification model."

This is **incorrect**.

The reason is much simpler.

When `num_labels` is not specified,

Hugging Face creates the classification head using the default configuration.

The default value is

```python
num_labels = 2
```

Therefore, the classification head contains **2 output neurons**.

```
Classification Head

      │
      ├── Class 0
      └── Class 1
```

This is only the default configuration.

It does **not** describe what the pretrained DistilBERT encoder has learned.

---

# Why doesn't Hugging Face automatically know the number of classes?

Consider the following code:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)
```

At this point,

the model knows only

- which pretrained checkpoint to download
- which architecture to build

It does **not** know

- which dataset will be used
- how many classes exist
- what task will be performed

For example,

later we may use

```python
load_dataset("zeroshot/twitter-financial-news-sentiment")
```

or

```python
load_dataset("ag_news")
```

or

```python
load_dataset("imdb")
```

These datasets contain different numbers of classes.

Since the model has no information about the dataset yet,

it uses the default value

```python
num_labels = 2
```

---

# Specifying the Number of Labels

If our dataset contains three sentiment classes

```
0 → Negative

1 → Neutral

2 → Positive
```

then we should create the model as

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)
```

Now

```python
print(model.config.num_labels)
```

returns

```python
3
```

The classification head becomes

```
Classification Head

      │
      ├── Class 0
      ├── Class 1
      └── Class 2
```

---

# Why is `num_labels` important?

During training,

the model predicts one score (logit) for each output neuron.

Suppose the dataset contains

```
Labels

0

1

2
```

The model must therefore produce

```
Output

Logit for Class 0

Logit for Class 1

Logit for Class 2
```

If the model has only two output neurons,

```
Class 0

Class 1
```

but the dataset contains

```
Label = 2
```

PyTorch cannot compute the loss because there is no output corresponding to class **2**.

The training fails with

```text
IndexError: Target 2 is out of bounds.
```

---

# Difference Between the Base Model and the Classification Head

```
distilbert-base-uncased

↓

Learns English Language

✓ Grammar

✓ Vocabulary

✓ Context

✓ Sentence Meaning

✗ No Sentiment Classification
```

```
AutoModelForSequenceClassification

↓

Adds Classification Head

↓

Number of output neurons depends on

num_labels
```

---

# Complete Flow

```
distilbert-base-uncased
        │
        ▼
Pretrained Language Model
        │
        ▼
AutoModelForSequenceClassification
        │
        ▼
Adds Classification Head
        │
        ▼
num_labels decides
the number of output neurons
        │
        ▼
Fine-Tune on Your Dataset
```

---

# Common Misconceptions

### ❌ `distilbert-base-uncased` is a binary classification model.

**Wrong.**

It is a pretrained language model.

---

### ❌ `num_labels=2` means DistilBERT learned only two classes.

**Wrong.**

Only the newly created classification head has two output neurons by default.

---

### ❌ Hugging Face automatically detects the number of dataset classes.

**Wrong.**

The model is created before it sees the dataset.

Therefore, the user must specify `num_labels` whenever the default does not match the task.

---

# Key Takeaways

- `distilbert-base-uncased` is a pretrained language model.
- It understands English but is not trained for sentiment classification.
- `AutoModelForSequenceClassification` adds a classification head.
- The number of output neurons is controlled by `num_labels`.
- If `num_labels` is not specified, the default value is **2**.
- The number of output neurons must match the number of classes in the dataset.
- Otherwise, training fails with errors such as `IndexError: Target 2 is out of bounds`.
