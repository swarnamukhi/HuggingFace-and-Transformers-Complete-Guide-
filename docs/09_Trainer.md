# 🤖 Trainer in Hugging Face

## What is Trainer?

`Trainer` is a high-level API provided by Hugging Face that manages the entire model training process.

Instead of writing your own training loop, validation loop, optimizer updates, checkpoint saving, logging, and evaluation, you simply provide the required components, and the `Trainer` performs these tasks automatically.

---

# Why do we need Trainer?

Training a transformer model manually requires writing a large amount of boilerplate code.

Without `Trainer`, we would have to manually:

- Create DataLoaders
- Create mini-batches
- Apply dynamic padding
- Convert data into tensors
- Send batches to the model
- Calculate loss
- Perform backpropagation
- Update model weights
- Save checkpoints
- Evaluate the model
- Log training progress

`Trainer` automates all of these tasks.

---

# Where does Trainer fit in the pipeline?

```text
Raw Dataset
      │
      ▼
AutoTokenizer
      │
      ▼
dataset.map()
      │
      ▼
Tokenized Dataset
      │
      ▼
DataCollatorWithPadding
      │
      ▼
Trainer
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Prediction
```

---

# Creating a Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    args=training_args
)
```

Creating the Trainer **does not start training**.

It only prepares everything required for training.

---

# Components passed to Trainer

## 1. model

```python
model=model
```

The pretrained model that will be fine-tuned.

Example:

```python
model = AutoModelForSequenceClassification.from_pretrained(...)
```

Trainer uses this model for:

- Forward pass
- Loss calculation
- Backpropagation
- Weight updates

---

## 2. train_dataset

```python
train_dataset=tokenized_dataset["train"]
```

Contains the training examples.

Each example already contains

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "label": 0
}
```

Trainer reads this dataset during training.

---

## 3. eval_dataset

```python
eval_dataset=tokenized_dataset["validation"]
```

Used for model evaluation.

The validation dataset is **never used to update model weights**.

Its purpose is to measure how well the model performs on unseen data.

---

## 4. data_collator

```python
data_collator=data_collator
```

Trainer does **not** pad sequences itself.

Instead, before each batch is sent to the model, Trainer passes that batch to the DataCollator.

The DataCollator:

- Dynamically pads the batch
- Converts lists into tensors
- Returns tensors to the Trainer

---

## 5. args

```python
args=training_args
```

Contains the training configuration.

Examples include:

- Number of epochs
- Learning rate
- Batch size
- Logging frequency
- Checkpoint saving
- Evaluation strategy

These settings will be discussed separately in the **TrainingArguments** document.

---

# What happens after creating Trainer?

When we write

```python
trainer = Trainer(...)
```

nothing is trained yet.

The Trainer only stores all required objects.

Conceptually:

```text
Trainer

├── Model
├── Training Dataset
├── Validation Dataset
├── Data Collator
└── Training Arguments
```

The actual training begins only when

```python
trainer.train()
```

is called.

---

# What happens internally?

Conceptually, Trainer performs the following steps:

```text
Read Training Dataset
          │
          ▼
Create Mini Batch
          │
          ▼
Call DataCollatorWithPadding
          │
          ▼
Pad Sequences
          │
          ▼
Convert to PyTorch Tensors
          │
          ▼
Send Batch to Model
          │
          ▼
Compute Loss
          │
          ▼
Backpropagation
          │
          ▼
Update Model Weights
          │
          ▼
Repeat Until All Epochs Complete
```

All of these steps are automatically managed by the Trainer.

---

# Trainer internally creates a DataLoader

We never create a DataLoader manually.

Trainer creates one internally.

Conceptually:

```text
Tokenized Dataset
        │
        ▼
Trainer
        │
        ▼
DataLoader
        │
        ▼
Batch 1
Batch 2
Batch 3
...
```

Each batch is then sent to the DataCollator.

---

# Dynamic Padding inside Trainer

Suppose the DataLoader creates a batch containing:

```text
18 tokens
32 tokens
11 tokens
25 tokens
```

Trainer sends this batch to

```text
DataCollatorWithPadding
```

The collator pads the batch to

```text
32
32
32
32
```

and converts it into tensors before sending it back to the Trainer.

---

# Does Trainer tokenize the data?

No.

Tokenizer works **before** Trainer.

Trainer expects an already tokenized dataset.

---

# Does Trainer perform padding?

No.

Padding is handled by `DataCollatorWithPadding`.

Trainer only calls the collator.

---

# Does Trainer loop through the dataset?

Yes.

Trainer automatically iterates through the dataset by creating mini-batches using an internal DataLoader.

The user does not need to write any loops.

Conceptually:

```text
for every batch:

    Create Batch

    Apply Dynamic Padding

    Convert to Tensor

    Send to Model

    Compute Loss

    Update Weights
```

This entire process is hidden inside the Trainer.

---

# Difference between Trainer() and trainer.train()

Creating the Trainer:

```python
trainer = Trainer(...)
```

Only prepares the training pipeline.

Nothing is trained.

---

Starting training:

```python
trainer.train()
```

Starts the complete training process.

---

# Summary

- Trainer is a high-level API for model training.
- It manages the complete training pipeline.
- It automatically creates DataLoaders.
- It automatically creates mini-batches.
- It calls `DataCollatorWithPadding` for dynamic padding.
- It sends padded tensors to the model.
- It computes loss and updates model weights.
- Creating a Trainer does **not** start training.
- Training begins only after calling `trainer.train()`.

---

# Overall Hugging Face Training Pipeline

```text
Raw Dataset
      │
      ▼
AutoTokenizer
      │
      ▼
dataset.map()
      │
      ▼
Tokenized Dataset
      │
      ▼
DataCollatorWithPadding
      │
      ▼
Trainer
      │
      ▼
trainer.train()
      │
      ▼
Fine-Tuned Model
      │
      ▼
Prediction
```
