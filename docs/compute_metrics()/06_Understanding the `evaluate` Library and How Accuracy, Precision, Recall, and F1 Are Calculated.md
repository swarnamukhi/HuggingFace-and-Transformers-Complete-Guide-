# Chapter 6: Understanding the `evaluate` Library and How Accuracy, Precision, Recall, and F1 Are Calculated

> Repository: **HuggingFace From First Principles**

---

# Table of Contents

1. Introduction
2. Why Does Hugging Face Have Another Library?
3. What Happens When We Write `evaluate.load("accuracy")`?
4. What Does `accuracy` Become?
5. Why Doesn't It Calculate Immediately?
6. What Happens When We Call `accuracy.compute()`?
7. How Accuracy Is Calculated
8. How Precision Is Calculated
9. How Recall Is Calculated
10. How F1 Is Calculated
11. Why Do We Return a Dictionary?
12. Complete Metric Pipeline
13. Summary

---

# Introduction

By now our evaluation pipeline looks like this.

```
Validation Dataset

↓

Trainer

↓

Model

↓

Logits

↓

NumPy

↓

argmax()

↓

Predicted Labels
```

Now another important question appears.

We write

```python
import evaluate

accuracy = evaluate.load("accuracy")
```

What exactly is happening here?

Is Hugging Face calculating Accuracy?

No.

Let's understand why.

---

# Why Another Library?

Remember,

Transformers already contains

- Models
- Trainer
- Tokenizers
- Pipelines

So why create another library?

Because

**training**

and

**evaluation**

are completely different responsibilities.

Think of a university.

```
Department

↓

Teaching
```

Another department

```
↓

Examinations
```

Both belong to the same university.

But they have different jobs.

Likewise,

```
Transformers

↓

Training
```

```
Evaluate

↓

Measuring
```

Keeping them separate makes the code cleaner.

---

# What Happens Here?

```python
accuracy = evaluate.load("accuracy")
```

Many beginners think

```
↓

Accuracy calculated
```

Wrong.

Nothing has been calculated.

Absolutely nothing.

Instead,

Hugging Face simply gives us an

**Accuracy Metric Object**.

Think of it like this.

Suppose

you buy a calculator.

Buying the calculator

does not perform a calculation.

It simply gives you

a tool.

Later,

you decide

what numbers to calculate.

Exactly the same thing happens here.

---

# What Does `load()` Mean?

Conceptually,

```
evaluate.load("accuracy")
```

means

> Give me the implementation of the Accuracy metric.

Not

> Calculate Accuracy.

---

# What Is Stored Inside `accuracy`?

Suppose we print

```python
print(type(accuracy))
```

We do not get

```
float
```

or

```
int
```

Instead,

we get

an object.

Conceptually,

```
Accuracy Object

↓

Knows

↓

How to calculate Accuracy
```

Think of it as

a machine waiting for data.

---

# Nothing Has Been Calculated Yet

At this point,

Trainer has not even started evaluation.

There are no predictions.

There are no labels.

So

what would Accuracy calculate?

Nothing.

That's why

`load()`

only prepares the metric.

---

# When Does Accuracy Actually Calculate?

Only here.

```python
accuracy.compute(

predictions=predictions,

references=labels

)
```

Now,

finally,

the metric has

everything it needs.

```
Predictions

+

True Labels
```

Now it can calculate Accuracy.

---

# Where Do These Predictions Come From?

Remember our previous chapters.

```
Model

↓

Logits

↓

argmax()

↓

Predictions
```

Example

```python
predictions

=

[2,1,0,2]
```

Labels

```python
labels

=

[2,0,0,1]
```

These are exactly what

Accuracy receives.

---

# How Does Accuracy Work?

Internally,

Accuracy is surprisingly simple.

Conceptually,

it performs

```python
correct = 0

for prediction,true_label in zip(predictions,labels):

    if prediction==true_label:

        correct+=1

accuracy = correct/len(labels)
```

Let's do it manually.

Predictions

```
2

1

0

2
```

Labels

```
2

0

0

1
```

Compare

```
2==2

✓
```

```
1==0

✗
```

```
0==0

✓
```

```
2==1

✗
```

Correct

```
2
```

Total

```
4
```

Accuracy

```
2/4

=

0.50
```

Exactly what

```
accuracy.compute()
```

returns.

---

# Why Don't We Write This Ourselves?

We certainly could.

But imagine implementing

- Accuracy
- Precision
- Recall
- F1
- BLEU
- ROUGE
- WER
- CER
- MCC

correctly,

handling

- binary classification
- multiclass classification
- multilabel classification

It becomes a huge amount of work.

Instead,

Hugging Face already provides

tested implementations.

We simply use them.

---

# Precision

Next,

we load

```python
precision = evaluate.load("precision")
```

Again,

nothing is calculated.

Later,

```python
precision.compute(...)
```

is executed.

Conceptually,

Precision asks

> Every time the model predicted a class,

how often was it correct?

Formula

```
Correct Positive Predictions

/

Total Positive Predictions
```

Suppose

```
Predicted Positive

10
```

Correct

```
8
```

Precision

```
8/10

=

0.8
```

---

# Recall

Recall asks

> Of all the actual positives,

how many did the model find?

Formula

```
Correct Positive Predictions

/

Actual Positives
```

Suppose

Actual Positives

```
20
```

Model Found

```
15
```

Recall

```
15/20

=

0.75
```

---

# F1 Score

F1 combines

Precision

and

Recall.

Think of it as

a balance.

A model should not only

predict correctly,

it should also

find all important examples.

The Evaluate library calculates

the harmonic mean internally.

We simply call

```python
f1.compute(...)
```

---

# Why Do We Use `average="weighted"`?

Our sentiment dataset contains

```
Negative

Neutral

Positive
```

Three classes.

Precision,

Recall

and

F1

must first calculate

each class separately.

Conceptually,

```
Negative

↓

Precision
```

```
Neutral

↓

Precision
```

```
Positive

↓

Precision
```

Now

three numbers exist.

How should they become

one number?

This is what

```
average
```

controls.

---

# Weighted Average

When we write

```python
average="weighted"
```

classes with

more samples

receive

more importance.

Suppose

```
Negative

800 samples
```

```
Neutral

150 samples
```

```
Positive

111 samples
```

Weighted averaging says

```
Negative

should influence the final score

more than

Positive
```

because

there are many more examples.

This is usually a good default

for multiclass classification.

---

# Why Does compute_metrics Return a Dictionary?

Our function ends with

```python
return {

"accuracy":...,

"precision":...,

"recall":...,

"f1":...

}
```

Why a dictionary?

Because Trainer doesn't know

how many metrics

we want.

Maybe

```
Accuracy
```

only.

Maybe

```
Accuracy

+

F1
```

Maybe

10 different metrics.

A dictionary lets us return

any number of metrics.

Trainer simply prints

every key.

Example

```
{

accuracy:0.91,

precision:0.90,

recall:0.89,

f1:0.89

}
```

Trainer displays

```
eval_accuracy

eval_precision

eval_recall

eval_f1
```

automatically.

---

# Complete Metric Pipeline

```
Validation Dataset

↓

Trainer

↓

Model

↓

Logits

↓

NumPy

↓

argmax()

↓

Predictions

↓

Accuracy.compute()

↓

Precision.compute()

↓

Recall.compute()

↓

F1.compute()

↓

Dictionary

↓

Trainer Prints Metrics
```

---

# One Important Observation

Notice something.

Trainer never knows

how Accuracy works.

Trainer simply executes

```
accuracy.compute(...)
```

The actual implementation belongs to

the

**Evaluate Library**.

Likewise,

Evaluate

knows nothing about

Transformers.

It simply receives

Predictions

and

Labels.

This separation is one of the reasons Hugging Face libraries remain modular and reusable.

---

# Summary

- `evaluate` is a separate Hugging Face library dedicated to evaluation metrics.
- `evaluate.load("accuracy")` does **not** calculate Accuracy.
- It returns an **Accuracy Metric Object**.
- Metrics are calculated only when we call `.compute()`.
- `accuracy.compute()` compares predicted labels with true labels.
- `precision.compute()`, `recall.compute()`, and `f1.compute()` use their own implementations inside the Evaluate library.
- We usually use `average="weighted"` for multiclass classification so that larger classes contribute proportionally to the final metric.
- `compute_metrics()` returns a dictionary because the Trainer can display any number of user-defined metrics.

---

# Next Chapter

Now that we understand every individual component, we are ready for the final chapter.

We will trace **one sentence** from the validation dataset all the way to the printed metrics.

Nothing will be skipped.

Every object,

every function,

every conversion,

every library,

and every line of code will be connected into one complete execution flow.
