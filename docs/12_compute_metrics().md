# Understanding `compute_metrics()` in Hugging Face Trainer

## Introduction

When we fine-tune a Hugging Face model using `Trainer`, the training output only shows:

- Training Loss
- Validation Loss

It does **not** automatically show metrics like:

- Accuracy
- Precision
- Recall
- F1-score

Many beginners wonder:

> "Why doesn't Hugging Face calculate these metrics automatically?"

To answer that question, we first need to understand how evaluation works internally.

---

# Why Doesn't Hugging Face Calculate Metrics Automatically?

Hugging Face is designed to support many different machine learning tasks.

Examples include:

- Text Classification
- Question Answering
- Named Entity Recognition
- Machine Translation
- Text Summarization
- Speech Recognition
- Image Classification

Each task requires different evaluation metrics.

For example:

| Task | Common Metrics |
|-------|----------------|
| Sentiment Analysis | Accuracy, Precision, Recall, F1 |
| Machine Translation | BLEU |
| Summarization | ROUGE |
| Question Answering | Exact Match, F1 |
| Text Generation | Perplexity, BLEU, Human Evaluation |

Even for text classification, there is no single correct metric.

For example:

Fraud Detection

```
99% Normal
1% Fraud
```

If a model predicts:

```
Everything is Normal
```

Accuracy becomes:

```
99%
```

But the model completely fails to detect fraud.

In this case, Recall and F1-score are much more important than Accuracy.

Since only the developer knows the business problem, Hugging Face cannot assume which metric should be calculated.

Therefore, Hugging Face only provides:

- Model predictions
- True labels

The developer decides how to evaluate them.

---

# What Happens During Validation?

Suppose we call:

```python
trainer.evaluate()
```

Internally, the Trainer performs the following steps.

---

## Step 1

Load one validation batch.

```
Validation Dataset
        │
        ▼
Validation Batch
```

---

## Step 2

Send the batch to the model.

Conceptually:

```python
outputs = model(**batch)
```

The model returns:

```python
outputs.loss
outputs.logits
```

Notice:

The model returns **both** loss and logits.

---

## Step 3

The Trainer repeats this process for every validation batch.

Conceptually:

```python
all_logits = []
all_labels = []

for batch in validation_dataloader:

    outputs = model(**batch)

    all_logits.append(outputs.logits)

    all_labels.append(batch["labels"])
```

We never write this code.

The Trainer does it internally.

---

## Step 4

After processing the entire validation dataset,

the Trainer combines everything.

Example:

```
all_logits.shape

(1061,3)

all_labels.shape

(1061,)
```

Now the Trainer has:

- Predictions for every validation sample
- True labels for every validation sample

---

# Why Doesn't Trainer Calculate Accuracy Now?

Because the Trainer still doesn't know which metric we want.

Maybe we want:

- Accuracy
- F1-score
- Precision
- Recall
- BLEU
- ROUGE

Only we know.

Therefore, the Trainer simply passes the collected outputs to our function.

Conceptually:

```python
compute_metrics(
    (
        all_logits,
        all_labels
    )
)
```

---

# What Is `eval_pred`?

Inside our function,

```python
def compute_metrics(eval_pred):
```

the variable

```python
eval_pred
```

already contains

```python
(
    logits,
    labels
)
```

We never create this tuple.

The Trainer creates it internally.

---

# Understanding

```python
logits, labels = eval_pred
```

This line simply performs tuple unpacking.

Before:

```python
eval_pred = (
    logits,
    labels
)
```

After:

```python
logits = ...

labels = ...
```

No computation happens here.

We are simply separating the two values.

---

# What Are Logits?

Suppose our model has three classes.

```
0 → Negative

1 → Neutral

2 → Positive
```

For one sentence,

the model produces

```python
[0.2,0.4,3.1]
```

These numbers are called **logits**.

They are raw scores.

Notice:

These are **not probabilities**.

---

# Converting Logits into Predicted Labels

The model predicts one score for every class.

Example:

```python
[
 [0.2,0.4,3.1],
 [0.5,2.8,0.3],
 [4.0,0.2,0.1]
]
```

We convert these scores into predicted class IDs using

```python
predictions = np.argmax(logits, axis=-1)
```

Result:

```python
[2,1,0]
```

Meaning

```
Positive

Neutral

Negative
```

---

# Why `np.argmax()`?

The Trainer passes NumPy arrays into `compute_metrics()`.

Therefore,

```python
np.argmax()
```

is commonly used.

If logits were PyTorch tensors,

we could also write

```python
torch.argmax()
```

---

# What About `model.config.id2label`?

Many beginners think

```python
model.config.id2label
```

converts logits into labels.

It does **not**.

It is only a dictionary.

Example:

```python
{
0:"Negative",
1:"Neutral",
2:"Positive"
}
```

If

```python
prediction = 2
```

then

```python
model.config.id2label[prediction]
```

returns

```
Positive
```

It is simply a lookup table.

It is **not** used when calculating metrics.

---

# Why Doesn't `accuracy.compute()` Use Loss?

The model returns

```python
outputs.loss
```

and

```python
outputs.logits
```

Why don't we calculate Accuracy using Loss?

Because Loss is only one aggregated number.

Example:

```
Loss = 0.42
```

From this value we cannot know:

- Which predictions were correct
- Which predictions were wrong
- Which class each sample belongs to

Therefore,

Accuracy cannot be calculated from Loss.

Instead,

we need

- Predicted labels
- Actual labels

These come from

```
Logits
↓

Argmax

↓

Predicted Labels

+

True Labels
```

---

# Computing Metrics

After obtaining predicted labels,

we calculate

```python
accuracy.compute(...)
precision.compute(...)
recall.compute(...)
f1.compute(...)
```

These functions belong to the Hugging Face **Evaluate** library.

Example:

```python
accuracy = evaluate.load("accuracy")
```

creates an Accuracy calculator.

Later,

```python
accuracy.compute(
    predictions=predictions,
    references=labels
)
```

calculates the metric.

Conceptually,

Accuracy performs something similar to

```python
correct = 0

for pred,true in zip(predictions,labels):

    if pred == true:
        correct += 1

accuracy = correct / len(labels)
```

---

# Who Writes Which Code?

## Code Written By Us

```python
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(
            predictions=predictions,
            references=labels
        )["accuracy"],

        "precision": precision.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["precision"],

        "recall": recall.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["recall"],

        "f1": f1.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["f1"],
    }
```

Everything inside this function is written by us.

---

## Code Managed Internally by Trainer

Conceptually:

```python
for batch in validation_dataloader:

    outputs = model(**batch)

    collect outputs.logits

    collect labels

combine logits

combine labels

compute_metrics(
    (
        logits,
        labels
    )
)
```

We never write this.

The Trainer handles it automatically.

---

# Complete Evaluation Flow

```
Validation Dataset
        │
        ▼
Validation DataLoader
        │
        ▼
Model
        │
        ▼
outputs.loss
outputs.logits
        │
        ▼
Trainer collects logits
Trainer collects labels
        │
        ▼
Creates

eval_pred = (
    logits,
    labels
)
        │
───────────────
        │
        ▼
Our compute_metrics()

        │
        ▼
Tuple Unpacking

logits, labels

        │
        ▼
np.argmax()

        │
        ▼
Predicted Labels

        │
        ▼
Accuracy

Precision

Recall

F1-score

        │
        ▼
Return Dictionary

        │
        ▼
Trainer Displays Metrics
```

---

# Key Takeaways

- The Trainer automatically performs the evaluation loop.
- The Trainer collects logits and labels from every validation batch.
- The Trainer creates `eval_pred` internally.
- `compute_metrics()` is written entirely by the developer.
- Hugging Face does not assume which metrics should be calculated because different tasks require different evaluation metrics.
- Metrics such as Accuracy and F1-score require predicted labels and true labels, not the loss value.
- `model.config.id2label` is only a dictionary for converting class IDs into readable class names and is **not** used when calculating metrics.
