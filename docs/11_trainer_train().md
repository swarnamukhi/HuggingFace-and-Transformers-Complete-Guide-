# trainer.train()

`trainer.train()` is the most important method in the Hugging Face training pipeline.

Although it looks like a single function call,

```python
trainer.train()
```

internally it performs dozens of operations required to train a deep learning model.

Understanding this method means understanding how the entire Hugging Face training pipeline works.

---

# Why do we call trainer.train()?

Before this point we have only prepared everything required for training.

We created

- Dataset
- Tokenizer
- Tokenized Dataset
- Data Collator
- Model
- TrainingArguments
- Trainer

However,

nothing has been learned yet.

The model weights are still unchanged.

Training actually begins only after

```python
trainer.train()
```

---

# High-Level Flow

```
trainer.train()

        │

        ▼

Create DataLoader

        │

        ▼

Read One Batch

        │

        ▼

DataCollator Pads Batch

        │

        ▼

Convert to Tensors

        │

        ▼

Send Batch to Model

        │

        ▼

Forward Pass

        │

        ▼

Calculate Loss

        │

        ▼

Backward Pass

        │

        ▼

Compute Gradients

        │

        ▼

Optimizer Updates Weights

        │

        ▼

Learning Rate Scheduler

        │

        ▼

Clear Old Gradients

        │

        ▼

Next Batch

        │

        ▼

Repeat Until Epoch Ends

        │

        ▼

Evaluation

        │

        ▼

Checkpoint Saving

        │

        ▼

Next Epoch

        │

        ▼

Training Finished
```

---

# Step 1 : Create DataLoader

The Trainer first creates a PyTorch DataLoader.

```
Tokenized Dataset

        │

        ▼

DataLoader
```

The DataLoader is responsible for

- shuffling the dataset
- creating batches
- sending batches one by one

Suppose

```
9543 examples
Batch Size = 8
```

Then

```
9543

↓

1193 batches

↓

1193 training steps
```

---

# Step 2 : Read One Batch

Instead of loading all data into memory,

the DataLoader reads only one batch.

```
Dataset

↓

Example 1

Example 2

...

Example 8

↓

Batch
```

---

# Step 3 : DataCollatorWithPadding

Different sentences have different lengths.

Example

```
Sentence 1

12 tokens

Sentence 2

20 tokens

Sentence 3

15 tokens
```

The DataCollator dynamically pads them.

```
12 → 20

20 → 20

15 → 20
```

Now all sentences have equal length.

---

# Step 4 : Convert to PyTorch Tensors

The collated batch becomes

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "labels": tensor(...)
}
```

These tensors are ready for the neural network.

---

# Step 5 : Forward Pass

The Trainer sends the batch into the model.

```
Input Batch

↓

DistilBERT

↓

Classification Head

↓

Logits
```

Example

```
[-2.15, 0.75, 3.91]
```

These numbers are called **logits**.

They are raw prediction scores.

---

# Step 6 : Calculate Loss

The dataset already contains the correct label.

Example

```
Sentence

↓

"The company reported record profits."

Correct Label

↓

Positive

↓

2
```

The model predicts

```
[-2.15, 0.75, 3.91]
```

Trainer computes

```
CrossEntropyLoss
```

using

```
Prediction

vs

Ground Truth
```

The result might be

```
Loss = 0.82
```

---

# Step 7 : Backward Pass

Now PyTorch calculates

```
How much did every weight contribute to the loss?
```

This process is called

```
Backpropagation
```

Internally

```python
loss.backward()
```

is executed.

Every trainable parameter now receives gradients.

---

# Step 8 : Optimizer Step

The optimizer updates the weights.

```
Old Weights

↓

Optimizer

↓

New Weights
```

For example

```
Before

Weight

0.512

↓

After

0.497
```

Now the model has learned slightly.

---

# Step 9 : Learning Rate Scheduler

If a scheduler exists,

Trainer adjusts the learning rate.

Example

```
Learning Rate

0.00005

↓

0.000049

↓

0.000048
```

This helps stable training.

---

# Step 10 : Clear Old Gradients

PyTorch accumulates gradients.

Therefore Trainer clears them.

Internally

```python
optimizer.zero_grad()
```

is called.

Without this,

gradients from previous batches would accumulate incorrectly.

---

# Step 11 : Next Batch

Trainer repeats

```
Read Batch

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer

↓

Zero Grad

↓

Next Batch
```

until every batch has been processed.

---

# End of One Epoch

After all batches are processed,

one epoch is complete.

```
1193 batches

↓

Epoch 1 Complete
```

---

# Evaluation

If

```python
eval_strategy="epoch"
```

Trainer evaluates the validation dataset.

Notice

```
Training

↓

Weight Updates

Validation

↓

No Weight Updates
```

Validation only measures performance.

---

# Saving Checkpoints

If

```python
save_strategy="epoch"
```

Trainer saves

```
checkpoint-1193
```

after Epoch 1.

The checkpoint contains

- model weights
- optimizer state
- scheduler state
- trainer state
- configuration

---

# Next Epoch

Trainer immediately starts

```
Epoch 2

↓

Repeat Entire Process
```

---

# Training Ends

After

```python
num_train_epochs
```

epochs,

training stops.

The final model contains updated weights learned from the new dataset.

---

# Complete Flow Diagram

```
Tokenized Dataset

        │

        ▼

DataLoader

        │

        ▼

Batch

        │

        ▼

DataCollatorWithPadding

        │

        ▼

PyTorch Tensors

        │

        ▼

Forward Pass

        │

        ▼

Logits

        │

        ▼

Loss

        │

        ▼

Backward Pass

        │

        ▼

Gradients

        │

        ▼

Optimizer

        │

        ▼

Updated Weights

        │

        ▼

Scheduler

        │

        ▼

Zero Gradients

        │

        ▼

Next Batch

        │

        ▼

Evaluation

        │

        ▼

Checkpoint

        │

        ▼

Next Epoch
```

---

# Common Misconceptions

### ❌ trainer.train() only trains the model.

Incorrect.

It also

- creates DataLoaders
- batches data
- pads inputs
- computes loss
- performs backpropagation
- updates weights
- evaluates
- saves checkpoints
- logs metrics
- manages epochs

---

### ❌ One call to trainer.train() trains only one epoch.

Incorrect.

It trains for

```python
num_train_epochs
```

epochs unless interrupted.

---

### ❌ Trainer updates weights during validation.

Incorrect.

Validation only measures model performance.

No learning occurs during validation.

---

# Key Takeaways

- `trainer.train()` is the central method of Hugging Face training.
- It coordinates every stage of the training pipeline.
- Training happens batch by batch.
- One optimizer update occurs after every batch.
- After each epoch, Trainer can evaluate and save checkpoints.
- Training continues until `num_train_epochs` is completed.
