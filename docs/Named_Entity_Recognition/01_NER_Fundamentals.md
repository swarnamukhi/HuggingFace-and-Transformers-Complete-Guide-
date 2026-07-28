# Named Entity Recognition (NER) - Fundamentals

---

# Learning Objectives

After completing this chapter, you will be able to:

- Understand what Named Entity Recognition (NER) is.
- Understand why NER is an important NLP task.
- Learn how NER evolved over the years.
- Understand traditional NER approaches.
- Learn how modern Transformer models solve NER.
- Understand where NER is used in real industries.
- Learn how ML engineers choose NER models.

---

# Prerequisites

Before learning NER, you should know:

- Python Basics
- Natural Language Processing (NLP)
- Tokenization
- Transformer Basics

---

# What is Named Entity Recognition (NER)?

Named Entity Recognition (NER) is a **Natural Language Processing (NLP)** task that automatically identifies and extracts **named entities** from unstructured text and classifies them into predefined categories.

Common entity types include:

- Person
- Organization
- Location
- Date
- Time
- Product
- Currency
- Event
- Disease
- Medication

Example

```text
Satya Nadella is the CEO of Microsoft.
```

NER Output

| Word | Entity |
|------|---------|
| Satya Nadella | PERSON |
| Microsoft | ORGANIZATION |

Another Example

```text
Apple opened a new office in Hyderabad on Monday.
```

NER Output

| Word | Entity |
|------|---------|
| Apple | ORGANIZATION |
| Hyderabad | LOCATION |
| Monday | DATE |

Unlike sentiment analysis, NER does not determine whether a sentence is positive or negative. Its objective is to identify and classify important entities mentioned in the text.

---

# Why Was NER Introduced?

As organizations started storing millions of documents, manually identifying important information became impossible.

Examples of such documents include:

- News articles
- Emails
- Medical records
- Legal contracts
- Customer reviews
- Government documents

Imagine searching one million news articles to answer questions such as:

- Which companies are mentioned?
- Which countries appear most frequently?
- Which people are involved?

Reading every document manually is impractical.

NER was introduced to automatically extract these important entities from large collections of text.

---

# Evolution of Named Entity Recognition

NER has evolved significantly over the past three decades.

```text
Rule-Based Systems
        │
        ▼
Dictionary (Gazetteers)
        │
        ▼
Statistical Machine Learning
(HMM)
        │
        ▼
Conditional Random Fields (CRF)
        │
        ▼
BiLSTM + CRF
        │
        ▼
Transformer-Based Models
(BERT, RoBERTa, DeBERTa)
```

Each generation attempted to overcome the limitations of the previous one.

---

# 1. Rule-Based NER

The earliest NER systems relied entirely on handwritten rules.

Example rules

```text
IF word starts with "Mr."

↓

Next word is PERSON
```

```text
Words ending with

Ltd

Inc

Corporation

↓

ORGANIZATION
```

Example

```text
Mr. John Smith works at Google.
```

Output

| Word | Entity |
|------|---------|
| John Smith | PERSON |
| Google | ORGANIZATION |

## Limitations

- Thousands of rules must be written manually.
- Difficult to maintain.
- Does not generalize well.
- Fails on unseen patterns.

---

# 2. Dictionary-Based NER (Gazetteers)

Instead of writing rules, researchers built dictionaries containing known entities.

Example

Company Dictionary

```text
Microsoft
Google
Amazon
Apple
```

Country Dictionary

```text
India
Japan
Australia
```

Sentence

```text
Google opened a new office in India.
```

Output

Google → Organization

India → Location

## Limitations

- New companies are missed.
- New people are missed.
- Dictionaries require continuous updates.
- Cannot recognize unknown entities.

---

# 3. Statistical Machine Learning

Researchers shifted from manually writing rules to allowing algorithms to learn patterns from labeled data.

Popular approaches included:

- Hidden Markov Models (HMM)
- Maximum Entropy Models

Instead of memorizing names, the model learned statistical relationships from training examples.

Although these models performed better than rule-based systems, they still struggled with complex language and long-range context.

---

# 4. Conditional Random Fields (CRF)

Conditional Random Fields (CRF) became the dominant sequence labeling algorithm for many years.

Instead of predicting each word independently, CRF considers the relationship between neighboring words.

Example

```text
New York University
```

Rather than assigning labels independently, CRF predicts the labels for the entire sequence, improving consistency.

For nearly fifteen years, CRF represented the state of the art in NER.

However, CRFs still depended heavily on manually engineered features.

---

# 5. Deep Learning (BiLSTM + CRF)

The next breakthrough combined deep learning with CRF.

Architecture

```text
Sentence

↓

Word Embeddings

↓

BiLSTM

↓

CRF

↓

Entity Labels
```

The BiLSTM automatically learned contextual features from the sentence, while the CRF produced consistent entity predictions.

This architecture significantly improved NER performance.

---

# 6. Transformer-Based NER

In 2017, Google introduced the Transformer architecture through the paper:

**Attention Is All You Need**

Soon after, models such as:

- BERT
- RoBERTa
- DeBERTa

became the standard approach for NER.

Instead of processing words sequentially like an LSTM, Transformers use **self-attention** to understand relationships between all words in a sentence simultaneously.

Example

```text
Apple released a new iPhone.
```

Transformer understands

Apple → Organization

Example

```text
I ate an apple.
```

Transformer understands

apple → Fruit

The surrounding context determines the correct meaning.

This contextual understanding is the primary reason Transformer-based models dominate modern NLP.

---

# Why Transformer Models Dominate Modern NLP

Today, most NLP applications begin with pretrained Transformer models rather than traditional machine learning techniques.

Examples

| NLP Task | Traditional Methods | Modern Approach |
|-----------|--------------------|-----------------|
| Sentiment Analysis | SVM, LSTM | BERT, RoBERTa |
| NER | CRF, BiLSTM | BERT, DeBERTa |
| Translation | Seq2Seq LSTM | Transformer, T5 |
| Question Answering | Rule-based | BERT |
| Text Generation | RNN | GPT, LLaMA |

Transformer models achieve higher accuracy because they capture contextual relationships much more effectively than earlier approaches.

---

# Real Industry Applications

NER is widely used across many industries.

## Banking

Extract

- Customer names
- Account numbers
- Organizations

Example

```text
My SBI credit card is blocked.
```

Output

SBI → Organization

Credit Card → Product

---

## Healthcare

Example

```text
Patient has diabetes and was prescribed Metformin.
```

Output

Diabetes → Disease

Metformin → Medication

---

## News Analytics

Example

```text
Tesla announced a new factory in India.
```

Output

Tesla → Organization

India → Location

---

## Human Resources

Resume

```text
Worked at TCS for 10 years.

Python

SQL

Azure
```

NER extracts

- Organization
- Skills
- Technologies

---

## Legal Industry

Example

```text
The agreement between Infosys and ABC Bank expires on January 1, 2028.
```

Output

Infosys → Organization

ABC Bank → Organization

January 1, 2028 → Date

---

# How ML Engineers Choose an NER Model

A common misconception is that engineers first choose a model.

In practice, they first identify the business problem.

```text
Business Problem

↓

NLP Task

↓

Model Type

↓

Fine-Tuned Model
```

Example

Business Requirement

> Extract company names from news articles.

↓

Task

Named Entity Recognition

↓

Model Type

Token Classification

↓

Pretrained Model

BERT fine-tuned for NER

---

# Popular Hugging Face NER Models

Examples include:

- dslim/bert-base-NER
- dbmdz/bert-large-cased-finetuned-conll03-english
- Jean-Baptiste/roberta-large-ner-english
- Davlan/xlm-roberta-base-ner-hrl

These are not entirely new architectures.

They are Transformer models that have been **fine-tuned specifically for the NER task**.

---

# Key Takeaways

- NER is an NLP task for identifying and classifying named entities.
- It evolved from rule-based systems to Transformer-based models.
- Traditional approaches relied on handwritten rules and statistical learning.
- Modern NER systems use pretrained Transformer models.
- NER is widely used in banking, healthcare, legal, HR, finance, and news analytics.
- ML engineers first identify the business task, then select an appropriate pretrained model.

---

# Interview Questions

### Q1. What is Named Entity Recognition?

NER is an NLP task that identifies and classifies named entities such as people, organizations, locations, dates, products, and other important entities from unstructured text.

---

### Q2. Why was NER introduced?

NER was introduced to automatically extract important information from large collections of unstructured text, reducing the need for manual processing.

---

### Q3. How has NER evolved over time?

Rule-Based Systems

↓

Dictionary-Based Systems

↓

Statistical Machine Learning

↓

Conditional Random Fields (CRF)

↓

BiLSTM + CRF

↓

Transformer-Based Models (BERT, RoBERTa, DeBERTa)

---

### Q4. Why are Transformer models preferred for NER?

Because they understand context using self-attention, resulting in significantly better accuracy than earlier approaches.

---

### Q5. How do ML engineers select an NER model?

They first identify the business problem, map it to the NLP task (NER), and then choose a pretrained model that has been fine-tuned for token classification.

---

## Next Chapter

In the next document, we will learn the technical implementation of NER using Hugging Face, including:

- Token Classification
- BIO Tagging (B-PER, I-PER, O)
- AutoModelForTokenClassification
- NER Pipeline
- Popular NER Datasets
- Fine-tuning BERT for NER
- Inference using Hugging Face
