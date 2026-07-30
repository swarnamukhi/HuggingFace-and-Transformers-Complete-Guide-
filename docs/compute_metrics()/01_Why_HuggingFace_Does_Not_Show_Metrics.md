# Chapter 1: Why Hugging Face Doesn't Show Accuracy, Precision, Recall, and F1 Automatically

> **Repository:** HuggingFace From First Principles

---

# Table of Contents

1. Introduction
2. The Beginner's Question
3. What Happens During Training?
4. What Happens During Validation?
5. Why Doesn't Trainer Print Accuracy?
6. Isn't Hugging Face Smart Enough?
7. The Real Design Philosophy
8. Understanding Different NLP Tasks
9. Why Accuracy Is Not Always the Right Metric
10. The Birth of `compute_metrics()`
11. Where Does `evaluate` Come In?
12. Summary

---

# Introduction

One of the first surprises beginners face while using Hugging Face's `Trainer` is this:

```python
trainer.train()
```

After training finishes, they see something like

```text
Training Loss

Validation Loss
```

But they do **not** see

- Accuracy
- Precision
- Recall
- F1-score

The immediate question becomes

> **"Why doesn't Hugging Face calculate these automatically?"**

At first glance, this feels strange.

Surely Hugging Face already has

- model predictions
- actual labels

So why not calculate everything?

To answer this question, we first need to understand what Hugging Face is trying to be.

---

# The Beginner's Thought Process

Most beginners think like this.

```
Model predicts labels.

↓

Dataset already contains labels.

↓

Trainer can compare them.

↓

Trainer should calculate Accuracy.
```

This sounds perfectly logical.

So why doesn't it?

Because there is one assumption hidden inside this reasoning.

The assumption is

> **Every Machine Learning task should use Accuracy.**

This assumption is wrong.

---

# What Happens During Training?

During training,

the Trainer repeatedly performs

```
Training Batch

↓

Model

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Update Weights
```

Notice something.

The Trainer only needs one value.

```
Loss
```

Loss tells the optimizer

how bad the prediction was.

Nothing else is required.

---

# What Happens During Validation?

Validation is different.

During validation,

weights are **not updated**.

Instead,

the Trainer simply asks

> "How well does the current model perform on unseen data?"

Conceptually,

the flow becomes

```
Validation Dataset

↓

Validation Batch

↓

Model

↓

Logits

↓

Store Predictions

↓

Repeat

↓

Store Labels

↓

Evaluation
```

At this point,

the Trainer now has

```
Predictions

AND

True Labels
```

So why stop here?

Why not calculate Accuracy?

---

# Isn't Hugging Face Smart Enough?

Imagine Hugging Face automatically calculated

```
Accuracy
```

Would that always make sense?

Let's test that idea.

---

# Example 1

Sentiment Analysis

Classes

```
Positive

Negative

Neutral
```

Accuracy is useful.

No problem.

---

# Example 2

Machine Translation

Input

```
I love AI.
```

Output

```
J'aime l'IA.
```

What is Accuracy here?

Should we compare

```
Word by Word?

Sentence by Sentence?

Character by Character?
```

Accuracy suddenly becomes meaningless.

Instead,

people use

```
BLEU Score
```

---

# Example 3

Text Summarization

Input

```
Long Article
```

Output

```
Summary
```

Should Accuracy compare

every generated word?

No.

Researchers instead use

```
ROUGE
```

---

# Example 4

Question Answering

Question

```
Where was Einstein born?
```

Prediction

```
Ulm
```

Reference

```
Ulm, Germany
```

Accuracy says

Wrong.

Humans say

Correct.

So researchers use

```
Exact Match

F1 Score
```

---

# Example 5

Named Entity Recognition

Sentence

```
John lives in Paris.
```

Predictions happen

for every token.

Accuracy is usually less informative.

Instead,

people often use

```
Token-level F1
```

---

# Example 6

Speech Recognition

Prediction

```
Hello World
```

Reference

```
Hello Word
```

Do we calculate Accuracy?

No.

Researchers use

```
Word Error Rate (WER)
```

---

# Same Trainer

Notice something interesting.

Every one of these tasks uses

the exact same

```python
Trainer
```

But every task needs

different metrics.

If Trainer automatically calculated

Accuracy,

then

Translation,

Summarization,

Speech,

Question Answering

would all receive

a useless metric.

---

# Another Problem

Even inside

Text Classification,

there is no universal metric.

Imagine

Fraud Detection.

Dataset

```
99%

Normal

1%

Fraud
```

Suppose the model predicts

```
Everything is Normal.
```

Accuracy becomes

```
99%
```

Looks amazing.

Reality?

The model never detects fraud.

It is completely useless.

Here,

Recall matters much more than Accuracy.

---

Medical Diagnosis

Suppose

```
999 Healthy

1 Cancer
```

The model predicts

```
Healthy

Healthy

Healthy

...

Healthy
```

Accuracy

```
99.9%
```

Would you deploy this model?

Of course not.

The model misses every cancer patient.

Recall is the important metric.

---

Spam Detection

Sometimes

Precision matters more.

News Classification

Sometimes

Accuracy is enough.

Recommendation Systems

Sometimes

Ranking metrics matter.

Again,

there is no universal answer.

---

# Hugging Face's Philosophy

Instead of forcing one metric,

Hugging Face follows a different philosophy.

The Trainer says

> "I will produce predictions."

The developer says

> "I know my business problem."

Therefore,

the developer decides

which metrics matter.

This is one of the reasons Hugging Face is so flexible.

---

# The Birth of `compute_metrics()`

The Hugging Face developers asked themselves

> "How can we let every user calculate their own metrics without rewriting the evaluation loop?"

Their solution was brilliant.

Instead of hardcoding Accuracy,

they added a hook.

That hook is

```python
compute_metrics()
```

The Trainer performs all the difficult work.

It

- runs validation
- collects predictions
- collects labels

Then it simply asks

> "Developer, here are the predictions."

> "You decide how to evaluate them."

That single callback is

```python
compute_metrics()
```

---

# Why Another Library?

Another question appears.

Why do we write

```python
import evaluate
```

instead of

```python
import transformers
```

The answer is

because

**training**

and

**evaluation**

are two completely different responsibilities.

The

```
transformers
```

library is responsible for

- Models
- Tokenizers
- Trainer
- Pipelines

The

```
evaluate
```

library is responsible for

metrics.

Think of them as two departments inside the same company.

```
Transformers

↓

Train Models
```

```
Evaluate

↓

Measure Models
```

Keeping them separate makes both libraries easier to maintain.

---

# Summary

The Trainer does **not** calculate Accuracy automatically because it cannot know which metric is appropriate for your task.

Different machine learning problems require different evaluation metrics.

Instead of making assumptions,

the Trainer focuses only on producing predictions.

The responsibility for evaluating those predictions belongs to the developer.

To make this easy,

Hugging Face provides the `compute_metrics()` callback together with the `evaluate` library.

In the next chapter, we will follow the evaluation process step by step and answer the next important question:

> **How does the Trainer create the predictions that are eventually passed to `compute_metrics()`?**
