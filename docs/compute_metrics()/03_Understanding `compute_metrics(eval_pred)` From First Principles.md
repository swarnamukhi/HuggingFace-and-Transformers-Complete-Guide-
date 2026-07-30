# Chapter 3: Understanding `compute_metrics(eval_pred)` From First Principles

> Repository: **HuggingFace From First Principles**

---

# Table of Contents

1. Introduction
2. The Biggest Beginner Question
3. We Never Call `compute_metrics()`
4. Then Who Calls It?
5. Passing a Function vs Calling a Function
6. How Trainer Remembers Our Function
7. What Exactly is `eval_pred`?
8. How Trainer Creates `eval_pred`
9. Why is `eval_pred` a Tuple?
10. Understanding Tuple Unpacking
11. What Happens After Unpacking?
12. Where Does Our Code Actually Begin?
13. Complete Execution Trace
14. Summary

---

# Introduction

In the previous chapter we learned something important.

The Trainer performs the entire evaluation loop.

It

- loads validation batches
- sends them through the model
- collects logits
- collects labels

Eventually it calls

```python
compute_metrics(...)
```

But another question immediately appears.

> **Who called `compute_metrics()`?**

We certainly didn't.

There is no line in our notebook that says

```python
compute_metrics(...)
```

Yet somehow the function executes.

Let's investigate.

---

# The Biggest Beginner Question

Most beginners imagine something like this.

```python
trainer.evaluate()

↓

Trainer

↓

Magic

↓

compute_metrics()
```

It feels like Python somehow "knows" about our function.

Python doesn't.

Someone has to explicitly call it.

Who?

The Trainer.

---

# Let's Look at Our Code

Earlier we wrote

```python
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics

)
```

Look carefully.

Notice something interesting.

We did **not** write

```python
compute_metrics()
```

Instead we wrote

```python
compute_metrics
```

No parentheses.

This tiny difference changes everything.

---

# Passing a Function vs Calling a Function

These two lines are completely different.

Calling a function

```python
compute_metrics()
```

means

> Execute the function immediately.

Passing a function

```python
compute_metrics
```

means

> Do not execute it.

> Just give someone the function.

Think of it like this.

Imagine your friend asks

"Give me your phone number."

You do **not** call them.

You simply hand them your number.

Later,

they decide when to call.

That is exactly what happens here.

We are giving Trainer the function itself.

Trainer decides when to execute it.

---

# What Does Trainer Do With It?

Conceptually,

inside Trainer,

something similar happens.

```python
self.compute_metrics = compute_metrics
```

Nothing runs.

Nothing executes.

Trainer simply stores a reference.

Think of it like this.

```
Trainer

↓

Memory

↓

compute_metrics

↓

Stored
```

Later,

during evaluation,

Trainer looks into its memory.

It finds

```
compute_metrics
```

and finally executes it.

Conceptually

```python
self.compute_metrics(eval_pred)
```

---

# When Does Trainer Execute It?

Not during training.

Not after Batch 1.

Not after Batch 2.

Not after Batch 50.

Only after

the **entire validation dataset**

has finished.

Flow

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

Combine Everything

↓

Call compute_metrics()
```

---

# What Exactly Is `eval_pred`?

Now another mystery appears.

Trainer executes

```python
compute_metrics(eval_pred)
```

But

where did

```
eval_pred
```

come from?

We never created it.

Trainer did.

---

# Earlier We Learned

After every batch,

Trainer stores

```
Logits

↓

all_logits
```

and

```
Labels

↓

all_labels
```

After evaluation finishes,

Trainer has

```
all_logits

shape

(1061,3)
```

and

```
all_labels

shape

(1061,)
```

Now Trainer needs to send

both objects

to our function.

Python functions receive arguments.

Trainer could have written

```python
compute_metrics(

all_logits,

all_labels

)
```

But Hugging Face chose another design.

Instead,

Trainer creates one object.

```python
eval_pred = (

all_logits,

all_labels

)
```

This object contains

two values.

```
(

logits,

labels

)
```

That object is called

```
eval_pred
```

---

# Why a Tuple?

Python allows multiple values to be grouped together.

Example

```python
person = (

"John",

25

)
```

Now

```
person
```

contains

two values.

Likewise,

Trainer creates

```python
eval_pred = (

logits,

labels

)
```

One variable.

Two values.

---

# Now Trainer Calls Our Function

Conceptually,

Trainer executes

```python
compute_metrics(

eval_pred

)
```

Now Python enters our function.

```python
def compute_metrics(eval_pred):
```

Notice something.

The parameter

```python
eval_pred
```

receives

whatever Trainer passed.

Since Trainer passed

```python
(

logits,

labels

)
```

our variable becomes

```python
eval_pred = (

logits,

labels

)
```

Nothing mysterious happened.

Python simply copied the argument into the parameter.

---

# Understanding Tuple Unpacking

Now comes one of the most confusing lines.

```python
logits, labels = eval_pred
```

Many beginners think

this is calculating something.

It isn't.

Suppose

```python
numbers = (

10,

20

)
```

Now

```python
a,b = numbers
```

After unpacking

```
a

↓

10
```

```
b

↓

20
```

Exactly the same thing happens here.

Before

```python
eval_pred = (

logits,

labels

)
```

After

```python
logits = eval_pred[0]

labels = eval_pred[1]
```

Python simply performs the unpacking automatically.

Nothing new is created.

Nothing is calculated.

---

# Visualizing the Process

Before unpacking

```
eval_pred

↓

(

logits,

labels

)
```

After

```
logits

↓

(1061,3)

labels

↓

(1061,)
```

Both variables now point to the same objects that were already inside the tuple.

---

# Where Does Our Code Actually Begin?

Everything before this point belonged to Trainer.

```
Validation

↓

Model

↓

Collect Logits

↓

Collect Labels

↓

Create Tuple

↓

Call Function
```

Now,

our code finally starts.

```python
def compute_metrics(eval_pred):

    logits, labels = eval_pred
```

Everything after this line

is written by us.

Everything before this line

was written by Hugging Face.

This boundary is extremely important.

---

# Complete Execution Trace

```
trainer.evaluate()

↓

Trainer Starts

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

Combine Everything

↓

eval_pred = (

all_logits,

all_labels

)

↓

compute_metrics(eval_pred)

────────────────────────────

Our Code Starts

↓

logits,labels=eval_pred

↓

Ready to Calculate Metrics
```

---

# What We Have NOT Done Yet

Notice something.

After unpacking,

we still have

```
Logits
```

We do **not** yet have

```
Predicted Labels
```

The model outputs

raw scores,

not class IDs.

The next question becomes

> How do raw logits become

```
Positive

Negative

Neutral
```

The answer is

```python
np.argmax()
```

That will be the focus of the next chapter.

---

# Summary

- We never call `compute_metrics()` ourselves.
- We pass the function to Trainer during initialization.
- Trainer stores the function internally.
- After validation finishes, Trainer creates

```python
eval_pred = (
    logits,
    labels
)
```

- Trainer calls

```python
compute_metrics(eval_pred)
```

- Python places that tuple into the parameter `eval_pred`.
- The line

```python
logits, labels = eval_pred
```

is simply **tuple unpacking**.
- No calculations happen during unpacking.
- After unpacking, our code has access to every prediction and every true label produced during evaluation.

---

## Next Chapter

In the next chapter we answer another important question:

> **The model returns PyTorch tensors. Why does `compute_metrics()` receive NumPy arrays?**

We'll also explain:

- GPU vs CPU
- Why tensors become NumPy arrays
- Why `np.argmax()` is used instead of `torch.argmax()`
- Where the conversion happens inside the Trainer
- How Hugging Face saves GPU memory during evaluation
