# AutoModelForQuestionAnswering

## What is AutoModelForQuestionAnswering?

`AutoModelForQuestionAnswering` is a Hugging Face class that automatically loads a **pretrained Transformer model** (such as BERT, RoBERTa, or DistilBERT) and attaches a **Question Answering (QA) head** on top of it.

It is used to answer questions from a given context.

Example:

**Context**

```
The capital of France is Paris.
```

**Question**

```
What is the capital of France?
```

**Answer**

```
Paris
```

---

# Why Do We Need AutoModelForQuestionAnswering?

Earlier, we learned that `AutoModel` produces only hidden states.

```
Sentence
     ↓
Tokenizer
     ↓
Token IDs
     ↓
Transformer
     ↓
Hidden States
```

The hidden states represent the meaning of every token.

Example:

```
The      → [0.12, ...]
capital  → [0.56, ...]
of       → [0.09, ...]
France   → [1.20, ...]
is       → [0.31, ...]
Paris    → [2.11, ...]
```

These vectors are useful for computers but they are **not answers**.

If we ask:

```
What is the capital of France?
```

`AutoModel` cannot answer it directly.

---

# The Problem

We want the model to return:

```
Paris
```

But the Transformer only gives hidden representations.

Therefore, we need another neural network that can determine **which words in the context form the answer**.

---

# Solution

Hugging Face provides:

```python
AutoModelForQuestionAnswering
```

It automatically builds:

```
                    Transformer
Embedding
      ↓
Transformer Layer 1
      ↓
Transformer Layer 2
      ↓
...
      ↓
Last Transformer Layer
      ↓
Hidden States
      ↓
──────────────────────────────
Question Answering Head
──────────────────────────────
      ↓
Start Position
End Position
      ↓
Extract Answer
```

---

# How Does It Find the Answer?

Instead of predicting:

```
Positive

Negative
```

the QA head predicts:

```
Where does the answer begin?

Where does the answer end?
```

Suppose the context is:

```
The capital of France is Paris.
```

Token positions:

```
0  The
1  capital
2  of
3  France
4  is
5  Paris
6  .
```

The QA head predicts:

```
Start = 5

End = 5
```

Therefore:

```
Answer = Paris
```

---

# Another Example

Context

```
Sachin Tendulkar was born in Mumbai.
```

Question

```
Where was Sachin Tendulkar born?
```

Token positions

```
0 Sachin
1 Tendulkar
2 was
3 born
4 in
5 Mumbai
6 .
```

Prediction

```
Start = 5

End = 5
```

Answer

```
Mumbai
```

---

# Multi-word Answers

Context

```
The Taj Mahal is located in Agra, India.
```

Question

```
Where is the Taj Mahal located?
```

Token positions

```
0 The
1 Taj
2 Mahal
3 is
4 located
5 in
6 Agra
7 ,
8 India
```

Prediction

```
Start = 6

End = 8
```

Answer

```
Agra, India
```

Notice that the model predicts a **span of tokens**, not a class label.

---

# Internal Architecture

Internally, the model looks similar to this:

```python
class BertForQuestionAnswering(nn.Module):

    def __init__(self):

        self.bert = BertModel(...)

        self.qa_outputs = nn.Linear(768, 2)
```

The two outputs represent:

- Start Position
- End Position

---

# Does It Replace BERT?

No.

Exactly like sequence classification, the Transformer remains unchanged.

Only the prediction head changes.

```
Question + Context
          ↓
Tokenizer
          ↓
Transformer
          ↓
Hidden States
          ↓
QA Head
          ↓
Start Position
End Position
          ↓
Answer
```

---

# Difference Between Classification Head and QA Head

Sequence Classification

```
Hidden States
      ↓
Classification Head
      ↓
Positive
```

Question Answering

```
Hidden States
      ↓
QA Head
      ↓
Start Position
End Position
      ↓
Extract Answer
```

The Transformer is the same.

Only the final prediction layer changes.

---

# Example Code

```python
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-cased-distilled-squad"
)

model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-cased-distilled-squad"
)
```

Notice that we never load:

```python
AutoModel(...)
```

because `AutoModelForQuestionAnswering` already contains the pretrained Transformer.

---

# Hugging Face Design Philosophy

Instead of building a completely different Transformer for every NLP task, Hugging Face reuses the same pretrained Transformer and only changes the prediction head.

```
Task
      ↓
Base Transformer
      ↓
Task-Specific Head
      ↓
Output
```

Examples

```
AutoModel
            ↓
Hidden States
```

```
AutoModelForSequenceClassification
            ↓
Class Label
```

```
AutoModelForQuestionAnswering
            ↓
Answer Span
```

```
AutoModelForTokenClassification
            ↓
Label for Every Token
```

```
AutoModelForCausalLM
            ↓
Next Token
```

---

# Key Takeaways

- `AutoModelForQuestionAnswering` contains the pretrained Transformer plus a Question Answering head.
- The Transformer understands the language.
- The QA head predicts the **start** and **end** positions of the answer in the context.
- It does not predict labels like Positive or Negative.
- The Transformer architecture remains the same; only the task-specific prediction head changes.
- We directly use `AutoModelForQuestionAnswering` for question answering tasks without loading `AutoModel` separately.
