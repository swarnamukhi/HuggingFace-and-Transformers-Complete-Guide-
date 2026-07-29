# 📦 DataCollatorWithPadding in Hugging Face

## What is DataCollatorWithPadding?

`DataCollatorWithPadding` is a Hugging Face utility that **pads tokenized sequences within a batch to the same length** and **converts them into PyTorch tensors** before they are sent to the model.

It works **after tokenization** and **before the model**.

---

# Why do we need DataCollatorWithPadding?

After tokenization, every sentence has a different number of tokens.

Example:

Sentence 1

```text
I love NLP
```

```python
input_ids = [101, 1045, 2293, 17953, 2361, 102]
```

Length = **6**

---

Sentence 2

```text
Transformers are amazing models
```

```python
input_ids = [101, 19081, 2024, 6429, 4275, 102]
```

Length = **12**

---

Sentence 3

```text
Machine Learning
```

```python
input_ids = [101, 3698, 4083, 102]
```

Length = **4**

---

Neural networks (PyTorch/TensorFlow) require every sequence in a batch to have the **same length**.

So before sending data to the model, shorter sequences must be padded.

---

# What does DataCollatorWithPadding do?

Suppose a batch contains:

| Sentence | Length |
|-----------|--------|
| Sentence 1 | 6 |
| Sentence 2 | 12 |
| Sentence 3 | 4 |

The longest sequence has length **12**.

The collator pads the remaining sequences.

Before padding:

```text
6
12
4
```

After padding:

```text
12
12
12
```

Now they can be converted into tensors.

---

# Where does it fit in the pipeline?

```text
Raw Dataset
      │
      ▼
Tokenizer
      │
      ▼
Tokenized Dataset
(No Padding Yet)
      │
      ▼
DataCollatorWithPadding
(Padding + Tensor Conversion)
      │
      ▼
PyTorch Tensor
      │
      ▼
Model
```

---

# Why don't we use padding during tokenization?

Suppose the dataset contains 10,000 sentences.

Longest sentence:

```text
190 tokens
```

Average sentence:

```text
25 tokens
```

If padding is applied during tokenization:

Every sentence becomes

```text
190 tokens
```

This wastes a large amount of memory.

Instead, Hugging Face performs **Dynamic Padding**.

Example:

Batch 1

```text
18
32
11
25
```

↓

Pad to **32**

---

Batch 2

```text
70
81
74
68
```

↓

Pad to **81**

---

Batch 3

```text
9
14
10
16
```

↓

Pad to **16**

Only the current batch is padded.

This is called **Dynamic Padding**.

---

# Creating the Data Collator

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

---

# Why do we pass the tokenizer?

The collator does **not tokenize** text.

It uses the tokenizer only to obtain padding information.

Different tokenizers have different padding settings.

Examples:

- Pad Token
- Pad Token ID
- Padding Side (left/right)

The tokenizer already knows these values.

Instead of manually specifying them, we simply pass the tokenizer.

---

# What input does DataCollatorWithPadding expect?

It expects a **list of tokenized examples**.

Example:

```python
features = [
    {
        "input_ids": [101, 1045, 2293, 102],
        "attention_mask": [1, 1, 1, 1],
        "label": 1
    },
    {
        "input_ids": [101, 2023, 2003, 1037, 2143, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "label": 0
    }
]
```

Notice that each example already contains:

- input_ids
- attention_mask
- label

The collator does **not** require raw text.

---

# Applying the Data Collator

```python
batch = data_collator(features)
```

---

# Output of the Data Collator

The output is a dictionary containing PyTorch tensors.

Example:

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "labels": tensor(...)
}
```

All sequences now have equal length.

---

# Example

Before DataCollator

```python
[
    [101,1045,2293,102],
    [101,2023,2003,1037,2143,102]
]
```

Lengths

```text
4
6
```

After DataCollator

```python
tensor([
    [101,1045,2293,102,0,0],
    [101,2023,2003,1037,2143,102]
])
```

Both sequences now have length **6**.

---

# Does DataCollator iterate over the dataset?

No.

The collator only processes **one batch at a time**.

It does **not** loop through the dataset.

Batch creation is handled by the training pipeline (DataLoader/Trainer).

Conceptually:

```text
Tokenized Dataset
        │
        ▼
Batch 1
        │
        ▼
DataCollatorWithPadding
        │
        ▼
Padded Tensor

Batch 2
        │
        ▼
DataCollatorWithPadding

Batch 3
        │
        ▼
DataCollatorWithPadding
```

The collator's only responsibility is:

1. Pad sequences within the current batch.
2. Convert lists into tensors.
3. Return the padded batch.

---

# Key Points

- Works **after tokenization**.
- Works **before the model**.
- Does **not** tokenize text.
- Does **not** iterate through the dataset.
- Performs **Dynamic Padding**.
- Converts Python lists into PyTorch tensors.
- Uses the tokenizer only to obtain padding information.
