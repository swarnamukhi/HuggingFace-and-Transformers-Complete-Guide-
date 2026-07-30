# Chapter 7: The Complete Evaluation Pipeline — From One Sentence to Printed Metrics

> Repository: **HuggingFace From First Principles**

---

# Introduction

Throughout this repository, we've studied each component separately.

- Why Hugging Face doesn't calculate metrics automatically.
- What the Trainer does.
- How `compute_metrics()` is called.
- Why NumPy is used.
- What logits are.
- How `argmax()` creates predictions.
- How the Evaluate library calculates metrics.

Now we'll connect everything into one continuous story.

Nothing will be skipped.

---

# Step 1 — Validation Dataset

Suppose our validation dataset contains one sentence.

```
"I love this movie."
```

Its correct label is

```
Positive

↓

2
```

The validation dataset already knows the answer.

---

# Step 2 — Trainer Starts Evaluation

We execute

```python
trainer.evaluate()
```

At this point, our code pauses.

The Trainer now takes control.

---

# Step 3 — Trainer Creates Validation Batches

Conceptually,

```
Validation Dataset

↓

Validation DataLoader

↓

Batch
```

The batch contains

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

---

# Step 4 — Trainer Calls the Model

The Trainer executes

```python
outputs = model(**batch)
```

The model performs a forward pass.

---

# Step 5 — Model Returns Logits

Instead of returning

```
Positive
```

the model returns

```python
[0.2, 0.4, 3.1]
```

These are logits.

They are raw scores.

---

# Step 6 — Trainer Collects Results

The Trainer stores

```python
outputs.logits
```

and

```python
batch["labels"]
```

It repeats this process for every validation batch.

---

# Step 7 — Evaluation Finishes

Suppose there are

```
1061 validation samples.
```

After processing every batch,

the Trainer has

```
logits

shape

(1061,3)
```

and

```
labels

shape

(1061,)
```

---

# Step 8 — Trainer Converts Tensors

The model produced PyTorch tensors.

The Trainer now

- moves them to the CPU
- converts them into NumPy arrays

Conceptually,

```
GPU Tensor

↓

CPU Tensor

↓

NumPy Array
```

---

# Step 9 — Trainer Creates `eval_pred`

Conceptually,

```python
eval_pred = (
    logits,
    labels
)
```

This tuple contains everything our metric function needs.

---

# Step 10 — Trainer Calls Our Function

The Trainer executes

```python
compute_metrics(eval_pred)
```

This is where our code begins.

---

# Step 11 — Tuple Unpacking

Inside our function,

```python
logits, labels = eval_pred
```

This simply unpacks the tuple.

No calculations happen here.

---

# Step 12 — Convert Logits to Predictions

The logits still look like

```python
[
 [0.2,0.4,3.1],
 [0.5,2.8,0.3],
 ...
]
```

We convert them into predicted class IDs.

```python
predictions = np.argmax(logits, axis=-1)
```

Now we have

```python
[2,1,0,...]
```

---

# Step 13 — Calculate Metrics

We pass the predictions and true labels to the Evaluate library.

```python
accuracy.compute(
    predictions=predictions,
    references=labels
)
```

Likewise,

```python
precision.compute(...)
```

```python
recall.compute(...)
```

```python
f1.compute(...)
```

Each metric compares the predictions against the true labels and returns a score.

---

# Step 14 — Return the Results

Our function returns

```python
return {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}
```

---

# Step 15 — Trainer Displays the Metrics

The Trainer receives the dictionary and prints

```
eval_accuracy

eval_precision

eval_recall

eval_f1
```

along with the validation loss.

Evaluation is complete.

---

# Complete Pipeline

```
Validation Dataset
        │
        ▼
Validation DataLoader
        │
        ▼
Validation Batch
        │
        ▼
Model Forward Pass
        │
        ▼
Logits (PyTorch Tensor)
        │
        ▼
Trainer Collects Logits & Labels
        │
        ▼
Move to CPU
        │
        ▼
Convert to NumPy
        │
        ▼
Create eval_pred
        │
        ▼
compute_metrics(eval_pred)
        │
        ▼
Tuple Unpacking
        │
        ▼
np.argmax()
        │
        ▼
Predicted Class IDs
        │
        ▼
Evaluate Library
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        └── F1
        │
        ▼
Dictionary of Metrics
        │
        ▼
Trainer Prints Results
```

---

# Final Takeaways

By the end of this journey, you should understand that:

- The **Trainer** manages the evaluation loop.
- The **model** produces logits, not labels.
- The **Trainer** gathers predictions across all validation batches.
- The **Trainer** converts tensors to NumPy before calling `compute_metrics()`.
- `compute_metrics()` is **your code**, but it is **called by the Trainer**.
- `np.argmax()` converts logits into predicted class IDs.
- The **Evaluate** library computes Accuracy, Precision, Recall, and F1 from those predictions.
- The **Trainer** displays whatever metrics your `compute_metrics()` function returns.

---

# Conclusion

The Hugging Face ecosystem is made up of several focused components that work together:

- **Datasets** prepares the data.
- **Tokenizer** converts text into model inputs.
- **Model** produces logits.
- **Trainer** orchestrates training and evaluation.
- **Evaluate** computes performance metrics.

Once you understand the responsibility of each component and how data flows between them, the evaluation pipeline is no longer a black box—it becomes a sequence of understandable, predictable steps.
