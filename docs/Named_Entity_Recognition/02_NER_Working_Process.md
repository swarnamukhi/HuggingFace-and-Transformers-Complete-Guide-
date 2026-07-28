# Named Entity Recognition (NER) - Chapter 2
# How NER Models Work

---

# Learning Objectives

After completing this chapter, you will be able to:

- Understand why NER is a task and not a model.
- Learn why Encoder-only Transformer models are used for NER.
- Understand the role of BERT in NER.
- Learn why a Token Classification Head is required.
- Understand how logits are generated for every token.
- Learn how the number of output neurons is determined.
- Understand the difference between pretrained BERT and pretrained NER models.

---

# Prerequisites

Before reading this chapter, you should understand:

- Transformer Architecture
- Encoder, Decoder and Encoder-Decoder models
- Hidden States
- Feed Forward Networks
- Classification Heads

---

# NER is a Task, Not a Model

Named Entity Recognition (NER) is an NLP task.

Its objective is to identify and classify entities such as:

- Person
- Organization
- Location
- Date
- Product
- Disease
- Custom entities

NER does not define a new neural network architecture.

Instead, existing Transformer models are adapted for this task.

---

# Which Transformer Architecture is Used?

Transformer models can be divided into three categories.

```text
Transformer
│
├── Encoder Only
├── Decoder Only
└── Encoder-Decoder
```

Examples

Encoder Only

- BERT
- RoBERTa
- DeBERTa
- DistilBERT

Decoder Only

- GPT
- LLaMA
- Mistral

Encoder-Decoder

- T5
- BART
- FLAN-T5

---

# Why Encoder Models?

The objective of NER is to understand the meaning of every word in the input sentence.

Example

```text
Satya Nadella works at Microsoft.
```

The model is **not generating new text**.

It is only identifying entities already present in the sentence.

Encoder models are specifically designed to learn contextual representations of the input text.

Therefore, Encoder-only models are the preferred architecture for NER.

---

# Role of BERT

BERT is a pretrained Encoder model.

Its job is to convert every input token into a contextual representation called a hidden state.

Example

```text
Sentence

↓

Tokenizer

↓

BERT Encoder

↓

Hidden States
```

These hidden states contain contextual information about every token.

However, hidden states are **not predictions**.

---

# Why is a Token Classification Head Needed?

BERT only produces hidden representations.

Applications such as NER require predictions.

Therefore, a task-specific Token Classification Head is added on top of BERT.

```text
Sentence

↓

Tokenizer

↓

BERT

↓

Hidden States

↓

Token Classification Head

↓

Logits

↓

Predicted Labels
```

The Token Classification Head converts hidden representations into logits.

---

# What is the Token Classification Head?

The Token Classification Head is typically a Linear (Fully Connected) layer.

Example

```text
Hidden Size

768

↓

Linear Layer

↓

Number of Labels
```

Unlike BERT, this layer depends entirely on the task.

---

# How are Predictions Made?

Suppose the sentence is

```text
Satya Nadella works at Microsoft.
```

After tokenization

```text
Satya
Nadella
works
at
Microsoft
```

BERT produces one hidden vector for every token.

Each hidden vector passes through the Token Classification Head independently.

```text
Hidden State

↓

Linear Layer

↓

Logits
```

The predicted label is obtained by selecting the largest logit.

---

# Why Does NER Produce Many Logits?

Sequence Classification predicts one label for the entire sentence.

NER predicts one label for every token.

Suppose

- Tokens = 5
- Labels = 9

Then

```text
5 Tokens

×

9 Labels

=

45 Logits
```

This can also be viewed as a matrix.

```text
Rows    → Tokens

Columns → Labels
```

Each row contains the logits for one token.

---

# How is the Number of Output Neurons Decided?

The output layer size depends on the number of labels.

Example

```text
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
B-DATE
I-DATE
O
```

Total labels

```text
9
```

Therefore

```text
Linear Layer

768

↓

9 Output Neurons
```

Different datasets may require different numbers of labels.

---

# Pretrained BERT vs Pretrained NER Model

These are not the same.

## Pretrained BERT

Training

```text
Books

+

Wikipedia

↓

BERT
```

Learns

- Grammar
- Context
- Language representation

It does not learn entity labels.

---

## Pretrained NER Model

```text
Pretrained BERT

↓

Token Classification Head

↓

NER Dataset

↓

Fine-tuning

↓

NER Model
```

The model learns entity labels during fine-tuning.

---

# Can a Pretrained NER Model Predict New Labels?

No.

The Token Classification Head is trained for a fixed set of labels.

Example

```text
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
O
```

If a company introduces a new label such as

```text
B-EMPLOYEE_ID
```

the existing model cannot predict it because the output layer does not contain neurons for that label.

A new classification head must be created and the model must be fine-tuned on the new dataset.

---

# Complete NER Architecture

```text
Sentence

↓

Tokenizer

↓

Encoder (BERT)

↓

Hidden States

↓

Token Classification Head

↓

Logits

↓

Predicted Entity Labels
```

---

# Key Takeaways

- NER is an NLP task, not a Transformer architecture.
- Encoder-only models are most suitable for NER.
- BERT produces contextual hidden states.
- Hidden states are converted into logits by the Token Classification Head.
- The output layer size depends on the number of labels.
- Pretrained BERT and pretrained NER models are different.
- New entity labels require additional fine-tuning.

---

# Interview Questions

### Q1. Why are Encoder-only models preferred for NER?

Because NER requires understanding the input text rather than generating new text.

---

### Q2. Does BERT directly predict entity labels?

No.

BERT produces contextual hidden states. The Token Classification Head converts them into logits for entity prediction.

---

### Q3. Why is a Token Classification Head required?

Because BERT only generates hidden representations and not task-specific predictions.

---

### Q4. What determines the number of output neurons in an NER model?

The number of entity labels in the dataset.

---

### Q5. Can a pretrained NER model recognize completely new entity types without retraining?

No.

New labels require modifying the output layer and fine-tuning the model on annotated data.

---

## Next Chapter

In the next chapter, we will study **BIO Tagging**, which explains how the predicted labels are represented and how multi-word entities are identified.
