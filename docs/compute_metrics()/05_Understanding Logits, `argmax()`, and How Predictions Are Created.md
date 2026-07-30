# Chapter 5: Understanding Logits, `argmax()`, and How Predictions Are Created

> Repository: **HuggingFace From First Principles**

---

# Table of Contents

1. Introduction
2. What Are Logits?
3. Why Doesn't the Model Return Labels?
4. Why Aren't Logits Probabilities?
5. One Sentence, One Prediction
6. Multiple Sentences
7. Understanding the Shape of Logits
8. Why Do We Need `argmax()`?
9. Understanding `axis=-1`
10. What Does `argmax()` Return?
11. Where Does `id2label` Fit?
12. Why Doesn't `accuracy.compute()` Use `id2label`?
13. Complete Prediction Flow
14. Summary

---

# Introduction

In the previous chapter we learned

```
Model

↓

GPU Tensor(Logits)

↓

CPU

↓

NumPy Array

↓

compute_metrics()
```

Inside

```python
compute_metrics(eval_pred)
```

we receive

```python
logits
```

But another question appears.

> What exactly are logits?

Many beginners think

```
Logits

↓

Predicted Labels
```

This is not true.

Let's understand what logits actually are.

---

# What Are Logits?

Suppose our sentiment analysis model has three classes.

```
0 → Negative

1 → Neutral

2 → Positive
```

Now imagine we give the model one sentence.

```
"I love this movie."
```

The model processes the sentence.

Instead of predicting

```
Positive
```

it returns

```python
[0.2, 0.4, 3.1]
```

These three numbers are called

**logits**.

---

# Why Three Numbers?

Because the model has

three possible classes.

One score for every class.

```
Negative

↓

0.2
```

```
Neutral

↓

0.4
```

```
Positive

↓

3.1
```

Notice

these are

scores,

not labels.

---

# Why Doesn't the Model Return "Positive"?

Excellent question.

Imagine the model returned

```
Positive
```

only.

Would we know

how confident it was?

No.

Consider these two situations.

Prediction A

```
Negative

↓

0.1

Neutral

↓

0.2

Positive

↓

8.7
```

Prediction B

```
Negative

↓

2.4

Neutral

↓

2.5

Positive

↓

2.6
```

Both predict

```
Positive
```

But they are very different.

The first prediction is extremely confident.

The second prediction is barely confident.

If the model returned only

```
Positive
```

this important information would be lost.

Therefore,

the model returns logits.

---

# Are Logits Probabilities?

No.

This is another common misunderstanding.

Logits are simply

raw scores.

Notice

```
0.2

+

0.4

+

3.1

=

3.7
```

Probabilities must add up to

```
1
```

These do not.

Therefore,

they are not probabilities.

---

# What Happens Next?

Now imagine

our validation dataset contains

1061 sentences.

Instead of producing

one row,

the model produces

1061 rows.

Example

```python
[
 [0.2,0.4,3.1],

 [0.5,2.8,0.3],

 [4.0,0.2,0.1]
]
```

Each row belongs to one sentence.

---

# Understanding the Shape

Suppose

```
1061 validation samples

3 classes
```

Then

```python
logits.shape
```

becomes

```
(1061,3)
```

Meaning

```
1061 rows

↓

Sentences
```

```
3 columns

↓

Class Scores
```

Visualized

```
Sentence 1

↓

[0.2 0.4 3.1]

Sentence 2

↓

[0.5 2.8 0.3]

Sentence 3

↓

[4.0 0.2 0.1]

...
```

Every row contains

scores,

not predictions.

---

# But Accuracy Needs Labels

Accuracy cannot compare

```
[0.2,0.4,3.1]
```

with

```
2
```

One is

a vector.

The other is

a class ID.

We first need to convert

the scores

into predicted labels.

---

# Enter `argmax()`

The simplest idea is

> Pick the largest score.

Why?

Because

the largest score represents

the class the model believes is most likely.

Example

```
[0.2,0.4,3.1]
```

Largest value

```
3.1
```

Position

```
2
```

Prediction

```
Class 2
```

Exactly what

`argmax()`

does.

---

# Understanding `np.argmax()`

We write

```python
predictions = np.argmax(

logits,

axis=-1

)
```

Most beginners only memorize this line.

Let's understand every part.

---

# What Does `argmax` Mean?

The name comes from

```
Argument

of

Maximum
```

It does **not** return

the largest value.

It returns

the **position**

of the largest value.

Example

```python
[0.2,0.4,3.1]
```

Largest value

```
3.1
```

Position

```
2
```

Therefore

```python
np.argmax(...)
```

returns

```
2
```

not

```
3.1
```

This is a very important distinction.

---

# What Does `axis=-1` Mean?

Suppose

```python
[
 [0.2,0.4,3.1],

 [0.5,2.8,0.3],

 [4.0,0.2,0.1]
]
```

Think of this as a table.

```
Rows

↓

Sentences
```

```
Columns

↓

Classes
```

We want

one prediction

for every sentence.

Therefore,

for every row,

find the largest column.

Sentence 1

```
0.2

0.4

3.1

↓

2
```

Sentence 2

```
0.5

2.8

0.3

↓

1
```

Sentence 3

```
4.0

0.2

0.1

↓

0
```

Final output

```python
[2,1,0]
```

This is exactly what

```python
axis=-1
```

tells NumPy to do.

It means

> Find the maximum along the last dimension.

In our case,

the last dimension is

the class dimension.

---

# What Does `argmax()` Return?

Notice

before

```python
logits
```

looked like

```
(1061,3)
```

After

```python
argmax()
```

we obtain

```
(1061,)
```

Why?

Because

every sentence now has

one predicted class.

Instead of

```
Three scores
```

we now have

```
One prediction
```

---

# Where Does `id2label` Fit?

Suppose

our prediction is

```python
2
```

Humans don't like

```
2
```

We prefer

```
Positive
```

Earlier,

during model creation,

we defined

```python
model.config.id2label
```

Example

```python
{
0:"Negative",

1:"Neutral",

2:"Positive"
}
```

Now

```python
model.config.id2label[2]
```

returns

```
Positive
```

Notice something.

Nothing is calculated.

The dictionary simply performs

a lookup.

---

# Does Accuracy Use `id2label`?

No.

This surprises many beginners.

Suppose

Predictions

```python
[2,1,0]
```

Labels

```python
[2,1,0]
```

Accuracy only checks

```
2==2

✓
```

```
1==1

✓
```

```
0==0

✓
```

It never converts

```
2

↓

Positive
```

because

integers are enough.

Using strings would only make

the computation slower.

Therefore,

`id2label`

is intended

for humans,

not for metric calculation.

---

# Complete Prediction Flow

```
Sentence

↓

Tokenizer

↓

Model

↓

Logits

↓

NumPy Array

↓

argmax()

↓

Predicted Class IDs

↓

Accuracy

↓

(Optional)

id2label

↓

Human Readable Labels
```

Notice

`id2label`

comes **after**

metric calculation,

not before.

---

# Summary

- The model never predicts labels directly.
- It predicts **logits**, which are raw scores.
- Each class receives one score.
- Logits are **not** probabilities.
- `np.argmax()` finds the **index** of the largest score.
- The returned index is the predicted class ID.
- `axis=-1` tells NumPy to choose the largest score **within each row**.
- After `argmax()`, the output changes from `(1061,3)` to `(1061,)`.
- `model.config.id2label` is only a lookup dictionary that converts class IDs into human-readable names.
- Metrics such as Accuracy, Precision, Recall, and F1 work directly with integer class IDs and **do not** use `id2label`.

---

# Next Chapter

In the next chapter we finally answer the question we postponed from the beginning:

> **What is the `evaluate` library?**

We'll explain:

- Why Hugging Face created a separate `evaluate` library.
- What `evaluate.load("accuracy")` actually returns.
- Where `accuracy.compute()` is implemented.
- How Accuracy, Precision, Recall, and F1 are calculated internally.
- Why we don't implement these metrics ourselves.
