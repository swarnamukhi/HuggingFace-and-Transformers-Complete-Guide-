# Chapter 2: Inside the Trainer Evaluation Loop

> Repository: HuggingFace From First Principles

---

# Table of Contents

1. Introduction
2. What Happens When We Call `trainer.evaluate()`?
3. Who Runs the Validation Dataset?
4. Where Does the Model Predict?
5. What Does the Model Return?
6. How Does Trainer Collect Predictions?
7. Why Doesn't Trainer Call `compute_metrics()` Immediately?
8. Combining Every Batch
9. Creating `eval_pred`
10. Where Our Code Starts
11. Trainer Code vs Our Code
12. Complete Flow Diagram
13. Summary

---

# Introduction

In Chapter 1 we answered one important question.

> Why doesn't Hugging Face automatically calculate Accuracy?

The answer was

Because Hugging Face doesn't know which metric is correct for your task.

Now another question naturally appears.

Suppose we write

```python
trainer.evaluate()
```

How does our

```python
compute_metrics()
```

function receive

```
logits

and

labels
```

We never create them.

We never pass them.

Yet somehow they magically appear.

Nothing magical is happening.

The Trainer creates them.

Let's see how.

---

# Step 1

When we execute

```python
trainer.evaluate()
```

Python enters Hugging Face's Trainer.

Notice something.

Our program stops here.

Everything after this point is Hugging Face code.

Conceptually

```
Our Code

↓

trainer.evaluate()

────────────────────────────

Trainer Code Starts
```

This boundary is extremely important.

Many beginners think

their code is still running.

It isn't.

Trainer has completely taken control.

---

# Step 2

Trainer Creates a Validation DataLoader

Earlier,

during preprocessing,

we created

```python
train_dataset

validation_dataset
```

The Trainer now creates a DataLoader for the validation dataset.

Conceptually,

```
Validation Dataset

↓

Validation DataLoader
```

A DataLoader does not send the entire dataset to the model.

Instead,

it creates mini-batches.

For example

Suppose

```
Validation Dataset

1061 samples
```

Batch size

```
16
```

The DataLoader produces

```
Batch 1

16 samples

↓

Batch 2

16 samples

↓

Batch 3

16 samples

...

↓

Last Batch

5 samples
```

The Trainer processes one batch at a time.

---

# Step 3

One Validation Batch Goes to the Model

Conceptually,

the Trainer performs

```python
for batch in validation_dataloader:

    outputs = model(**batch)
```

Did we write this code?

No.

The Trainer wrote it.

This is one of the biggest responsibilities of Trainer.

It manages the entire evaluation loop.

---

# What Is Inside One Batch?

A batch is usually a dictionary.

Example

```python
batch = {

"input_ids": ...,

"attention_mask": ...,

"labels": ...

}
```

Notice something interesting.

The labels are still present.

The validation dataset already knows the correct answers.

The Trainer simply forwards them to the model.

---

# Step 4

The Model Performs a Forward Pass

The Trainer executes

```python
outputs = model(**batch)
```

Now the model predicts.

Suppose

our batch contains

16 sentences.

The model predicts

16 outputs.

For a 3-class sentiment model

the logits look like

```
(16,3)
```

Meaning

```
16 sentences

3 scores for every sentence
```

---

# What Does the Model Return?

The model does not return

only predictions.

Instead,

it returns a

SequenceClassifierOutput.

Conceptually

```python
outputs

↓

loss

logits
```

Example

```python
outputs.loss

↓

0.42
```

```python
outputs.logits

↓

tensor(...)

shape

(16,3)
```

Notice

the logits are still

PyTorch tensors.

Nothing has been converted yet.

---

# Step 5

Trainer Stores Everything

Now the Trainer starts collecting outputs.

Conceptually

```python
all_logits = []

all_labels = []
```

Then

inside every iteration

```python
all_logits.append(outputs.logits)

all_labels.append(batch["labels"])
```

Notice

Trainer stores

both

Predictions

and

True Labels.

---

# Why Doesn't Trainer Call compute_metrics() Here?

This is an excellent question.

Suppose

after Batch 1

the Trainer immediately calculated Accuracy.

Would that be useful?

No.

It has only seen

16 samples.

The remaining

1045 samples

have not been evaluated.

Metrics calculated now would be incomplete.

Therefore

Trainer waits.

---

Conceptually

```
Batch 1

↓

Store

↓

Batch 2

↓

Store

↓

Batch 3

↓

Store

↓

...

↓

Last Batch

↓

Store

↓

NOW evaluate
```

---

# Step 6

Combining Every Batch

Suppose

we have

1061 validation samples.

Earlier

every batch looked like

```
(16,3)

(16,3)

(16,3)

...

(5,3)
```

Trainer combines everything into

```
(1061,3)
```

Likewise

labels become

```
(1061,)
```

Now Trainer has

every prediction

and

every true label

for the entire validation dataset.

---

# Step 7

Creating eval_pred

This is the part that confuses almost everyone.

Trainer now creates

a tuple.

Conceptually

```python
eval_pred = (

all_logits,

all_labels

)
```

Notice something.

We never wrote this line.

Trainer created it.

---

# Step 8

Trainer Calls Our Function

Remember

during Trainer creation

we wrote

```python
trainer = Trainer(

...

compute_metrics=compute_metrics

)
```

Notice carefully.

There are

NO parentheses.

We are not executing the function.

We are simply giving the Trainer a reference to our function.

Think of it like giving someone your phone number.

You are not calling them.

You are simply saying

"If you ever need me,

here is how you can reach me."

The Trainer remembers this function internally.

Later,

after validation finishes,

the Trainer finally executes

```python
compute_metrics(eval_pred)
```

This is the first line of OUR code.

Everything before this line belonged to Trainer.

Everything after this line belongs to us.

---

# Trainer Code vs Our Code

Trainer Code

```
Validation Dataset

↓

Validation DataLoader

↓

Model

↓

Loss

↓

Logits

↓

Collect Logits

↓

Collect Labels

↓

Combine Batches

↓

Create eval_pred

↓

Call compute_metrics()
```

Our Code

```
def compute_metrics(eval_pred):

↓

logits,labels=eval_pred

↓

argmax()

↓

Accuracy

↓

Precision

↓

Recall

↓

F1

↓

Return Dictionary
```

Notice the boundary.

Trainer's responsibility ends

the moment

our function starts executing.

---

# Complete Evaluation Flow

```
Validation Dataset

↓

Validation DataLoader

↓

Batch 1

↓

Model

↓

Outputs

↓

Store Logits

↓

Store Labels

↓

Repeat

↓

Repeat

↓

Repeat

↓

Last Batch

↓

Combine All Logits

↓

Combine All Labels

↓

Create

eval_pred

↓

compute_metrics(eval_pred)

↓

Our Code Starts
```

---

# Summary

The Trainer is responsible for the entire evaluation loop.

It creates the validation DataLoader.

It sends every batch to the model.

It collects logits.

It collects labels.

It waits until the entire validation dataset has been processed.

Only then does it create

```python
eval_pred = (
    logits,
    labels
)
```

and finally call our

```python
compute_metrics()
```

function.

Notice that our code never performs the evaluation loop.

Our responsibility begins only after the Trainer has already collected every prediction and every true label.

---

## Next Chapter

In the next chapter, we answer one of the biggest questions beginners have:

> **How does `compute_metrics(eval_pred)` receive `eval_pred` even though we never call the function ourselves?**

We'll also explain:

- Why `logits, labels = eval_pred` works.
- What tuple unpacking really is.
- How Python functions receive arguments.
- Why `compute_metrics` is passed without parentheses.
- How the Trainer stores our function internally.
