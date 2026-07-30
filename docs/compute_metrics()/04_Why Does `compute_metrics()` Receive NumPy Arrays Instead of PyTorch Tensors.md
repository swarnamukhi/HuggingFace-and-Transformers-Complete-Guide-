# Chapter 4: Why Does `compute_metrics()` Receive NumPy Arrays Instead of PyTorch Tensors?

> Repository: **HuggingFace From First Principles**

---

# Table of Contents

1. Introduction
2. The Beginner's Observation
3. The Model Returns Tensors
4. Then Why Does `compute_metrics()` Receive NumPy Arrays?
5. Understanding GPU vs CPU
6. Why Can't NumPy Live on the GPU?
7. Where Does the Conversion Happen?
8. Why Doesn't Hugging Face Keep Tensors?
9. Why `np.argmax()` Instead of `torch.argmax()`?
10. What Happens If We Use `torch.argmax()`?
11. Complete Data Flow
12. Summary

---

# Introduction

By now we know the Trainer eventually calls

```python
compute_metrics(eval_pred)
```

Inside our function we usually write

```python
logits, labels = eval_pred
```

and then

```python
predictions = np.argmax(logits, axis=-1)
```

Most beginners immediately ask

> Wait...

> The model is a PyTorch model.

> Doesn't PyTorch return tensors?

Yes.

It absolutely does.

So another question naturally appears.

> Then why am I using NumPy?

Let's answer that from first principles.

---

# The Model Returns Tensors

Remember

our model is a PyTorch model.

When Trainer executes

```python
outputs = model(**batch)
```

the model returns

```python
outputs.logits
```

These logits are

```python
torch.Tensor
```

You can verify this yourself.

```python
outputs = model(**batch)

print(type(outputs.logits))
```

Output

```text
<class 'torch.Tensor'>
```

Nothing surprising.

PyTorch models always produce PyTorch tensors.

---

# Where Are These Tensors?

Another question.

Where do these tensors live?

If training uses a GPU,

then

```
outputs.logits
```

also live on the GPU.

Conceptually

```
GPU

↓

outputs.logits

↓

torch.Tensor
```

Nothing has been moved to the CPU yet.

---

# Why GPU?

GPUs are extremely fast at

- Matrix Multiplication
- Tensor Operations
- Deep Learning Computation

Our model performs millions of mathematical operations.

Those operations belong on the GPU.

That's why Hugging Face automatically moves

- model
- batches

to the GPU.

The forward pass happens entirely there.

---

# Then Why Doesn't `compute_metrics()` Receive GPU Tensors?

Excellent question.

Let's think about what happens after evaluation.

Suppose

our validation dataset contains

```
1061 samples
```

After evaluation,

Trainer now has

```
1061 predictions
```

Does it still need the GPU?

No.

Evaluation is finished.

The GPU has completed its job.

Now we only want to calculate things like

```
Prediction == Label
```

Example

```
2 == 2

True
```

or

```
1 == 0

False
```

These are very small computations.

A CPU is perfectly capable of doing them.

Keeping everything on the GPU would simply waste GPU memory.

---

# Why Not Leave Everything on the GPU?

Imagine this.

Your GPU contains

```
Model

+

Optimizer

+

Gradients

+

Validation Logits
```

Suppose

the validation dataset contains

100,000 samples.

Keeping all logits on the GPU would consume a huge amount of memory.

Instead,

Trainer does something smarter.

After evaluation,

it says

> My GPU work is finished.

Let's move everything back to the CPU.

---

# Moving from GPU to CPU

In PyTorch,

moving a tensor to the CPU looks like

```python
tensor.cpu()
```

Example

```python
gpu_tensor = outputs.logits

cpu_tensor = gpu_tensor.cpu()
```

Notice

it is still

a tensor.

Only its location changed.

```
GPU Tensor

↓

CPU Tensor
```

---

# But `compute_metrics()` Doesn't Receive Tensors

Correct.

One more step happens.

After moving to the CPU,

Trainer converts

the tensor

into a NumPy array.

Conceptually

```python
numpy_logits = cpu_tensor.numpy()
```

Now

```python
type(numpy_logits)
```

becomes

```text
numpy.ndarray
```

This is exactly what arrives inside

```python
compute_metrics()
```

---

# Why Convert to NumPy?

Because metric computation

does not require PyTorch.

Think about Accuracy.

Conceptually,

Accuracy simply does

```python
prediction == label
```

Nothing here requires

- Autograd
- Gradients
- GPU kernels
- Neural Networks

NumPy is lightweight.

Fast.

Simple.

So Hugging Face converts tensors into NumPy arrays.

---

# Where Does This Happen?

We never write

```python
tensor.cpu().numpy()
```

So who does?

The Trainer.

Conceptually,

Trainer performs something similar to

```python
all_logits = all_logits.cpu().numpy()

all_labels = all_labels.cpu().numpy()
```

Then

```python
eval_pred = (

all_logits,

all_labels

)
```

Then

```python
compute_metrics(eval_pred)
```

Notice

the conversion happens

BEFORE

our function starts.

---

# Visualizing the Flow

```
Validation Batch

↓

GPU

↓

Model

↓

GPU Tensor(Logits)

↓

Trainer finishes evaluation

↓

Move Tensor to CPU

↓

Convert to NumPy

↓

Create eval_pred

↓

compute_metrics()
```

---

# Why Do Hugging Face Examples Use `np.argmax()`?

Now the answer becomes obvious.

Inside

```python
compute_metrics()
```

our variable

```python
logits
```

is already

```python
numpy.ndarray
```

Therefore

```python
np.argmax()
```

is the natural choice.

Example

```python
predictions = np.argmax(

logits,

axis=-1

)
```

---

# Can We Use `torch.argmax()`?

Yes.

But only if

```
logits
```

is still a tensor.

For example

```python
predictions = torch.argmax(

logits,

dim=-1

)
```

works perfectly

if

```python
type(logits)
```

is

```python
torch.Tensor
```

In the Trainer,

however,

the conversion has already happened.

So

```python
torch.argmax()
```

would no longer work directly.

---

# Verify It Yourself

Inside

```python
compute_metrics()
```

temporarily write

```python
print(type(logits))

print(type(labels))
```

You will most likely see

```text
<class 'numpy.ndarray'>

<class 'numpy.ndarray'>
```

This proves

the Trainer already converted them.

---

# Complete Flow

```
Validation Batch

↓

GPU Tensor

↓

Model

↓

GPU Tensor(Logits)

↓

Trainer Stores Tensor

↓

Evaluation Finishes

↓

tensor.cpu()

↓

tensor.numpy()

↓

NumPy Array

↓

compute_metrics()

↓

np.argmax()
```

---

# One Important Observation

Notice something interesting.

The model

never knows

NumPy exists.

The model only works with tensors.

The conversion happens

AFTER

the model has completely finished its work.

This is why

changing

```python
np.argmax()
```

to

```python
torch.argmax()
```

inside

```python
compute_metrics()
```

usually doesn't make sense.

By the time our function begins,

the tensors have already become NumPy arrays.

---

# Summary

- PyTorch models always return **PyTorch tensors**.
- During evaluation, those tensors usually live on the GPU.
- After evaluation finishes, the Trainer no longer needs GPU computation.
- The Trainer moves tensors from the GPU to the CPU.
- The Trainer converts CPU tensors into NumPy arrays.
- The Trainer creates

```python
eval_pred = (
    logits,
    labels
)
```

using those NumPy arrays.
- Therefore, inside `compute_metrics()`, `logits` is usually a `numpy.ndarray`.
- This is why Hugging Face examples use `np.argmax()` instead of `torch.argmax()`.

---

# Next Chapter

In the next chapter we finally answer one of the most important questions in this repository:

> **What are logits?**

We'll learn:

- What logits actually represent.
- Why they are not probabilities.
- Why we need `argmax()`.
- What `axis=-1` means.
- How logits become predicted class IDs.
- Why `model.config.id2label` is **not** used to calculate metrics.
