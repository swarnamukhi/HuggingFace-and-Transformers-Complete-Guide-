# Hugging Face Pipeline - Chapter 2
## Important Technical Concepts Every ML Engineer Should Know

---

# Learning Objectives

After completing this chapter, you will be able to:

- Process single and multiple inputs using a pipeline.
- Use your own pretrained model from Hugging Face Hub.
- Run inference on a GPU.
- Understand batching, padding, and truncation.
- Choose between PyTorch and TensorFlow.
- Know when to use and when to avoid the Pipeline API.

---

# Prerequisites

Before reading this chapter, you should understand:

- Pipeline Basics
- AutoTokenizer
- AutoModel
- PyTorch Tensors
- Model Inference

---

# 1. Processing a Single Sentence

The simplest use case is classifying a single sentence.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I love this laptop.")
```

Output

```python
[
    {
        "label": "POSITIVE",
        "score": 0.9998
    }
]
```

Internally

```text
Sentence
    │
    ▼
Tokenizer
    │
    ▼
Tensor
    │
    ▼
Model
    │
    ▼
Prediction
```

---

# 2. Processing Multiple Sentences (Batch Inference)

One of the advantages of Pipeline is that it accepts a list of sentences.

```python
reviews = [
    "Amazing movie",
    "Worst phone",
    "Excellent battery"
]

result = classifier(reviews)
```

Output

```python
[
    {"label":"POSITIVE","score":0.998},
    {"label":"NEGATIVE","score":0.997},
    {"label":"POSITIVE","score":0.999}
]
```

Internally

```text
Review 1
Review 2
Review 3
      │
      ▼
Batch Tokenization
      │
      ▼
Single Forward Pass
      │
      ▼
Three Predictions
```

Instead of calling the model three separate times, Pipeline groups the inputs together, making inference more efficient.

---

# 3. Using a Specific Model

When only the task is specified,

```python
classifier = pipeline("sentiment-analysis")
```

Pipeline automatically downloads the default model.

Sometimes you want to use a specific model.

```python
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
```

Internally

```text
Pipeline

↓

Downloads Config

↓

Downloads Tokenizer

↓

Downloads Model Weights

↓

Creates Pipeline Object
```

This is useful when your team has trained or fine-tuned a custom model.

---

# 4. Specifying Both Tokenizer and Model

Normally, the tokenizer is loaded automatically.

However, you can explicitly provide both.

```python
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
```

This is commonly used when the model has already been loaded into memory.

---

# 5. Choosing the Deep Learning Framework

Pipeline supports multiple frameworks.

PyTorch

```python
classifier = pipeline(
    "sentiment-analysis",
    framework="pt"
)
```

TensorFlow

```python
classifier = pipeline(
    "sentiment-analysis",
    framework="tf"
)
```

If you don't specify anything,

```python
pipeline(...)
```

automatically chooses the available supported framework.

---

# 6. Running on GPU

By default,

```python
classifier = pipeline("sentiment-analysis")
```

runs on the CPU.

To use GPU

```python
classifier = pipeline(
    "sentiment-analysis",
    device=0
)
```

Meaning

```text
device = 0

↓

GPU 0
```

If your machine has multiple GPUs

```python
device=1
```

means

```text
GPU 1
```

If no GPU exists,

```python
device=-1
```

uses the CPU.

---

# 7. Batch Size

Suppose you have

```text
100,000 Reviews
```

Bad approach

```python
for review in reviews:
    classifier(review)
```

This makes one prediction at a time.

Better approach

```python
classifier(
    reviews,
    batch_size=32
)
```

Internally

```text
Reviews

↓

32 Reviews

↓

32 Reviews

↓

32 Reviews

↓

Prediction
```

Batching significantly improves GPU utilization.

---

# 8. Truncation

Transformer models have a maximum sequence length.

Suppose BERT supports

```text
Maximum Tokens = 512
```

What if a review contains

```text
900 Tokens?
```

Without truncation

```python
classifier(review)
```

may produce an error.

Instead

```python
classifier(
    review,
    truncation=True
)
```

Pipeline keeps only the first supported tokens.

```text
900 Tokens

↓

512 Tokens

↓

Model
```

---

# 9. Padding

Suppose we have

Sentence 1

```text
I love AI
```

4 tokens

Sentence 2

```text
The movie was absolutely fantastic
```

7 tokens

The model cannot process tensors with different lengths.

Pipeline automatically pads the shorter sentence.

```text
Sentence 1

I love AI PAD PAD PAD

Sentence 2

The movie was absolutely fantastic
```

Now both sequences have equal length.

The attention mask ensures the model ignores the PAD tokens during inference.

---

# 10. Device Placement

Instead of

```text
CPU

↓

GPU
```

for every prediction,

Pipeline transfers the model to the selected device once.

```text
Model

↓

GPU

↓

Prediction

↓

Prediction

↓

Prediction
```

This avoids repeatedly copying the model between devices.

---

# 11. Pipeline Return Type

The pipeline returns

```python
[
    {
        "label":"POSITIVE",
        "score":0.998
    }
]
```

It does **not** return

- logits
- hidden states
- embeddings
- attention scores

If your application needs those values,

use

```python
AutoTokenizer

AutoModel
```

instead.

---

# 12. When Should You Use Pipeline?

Pipeline is ideal for

- Learning
- Demonstrations
- Rapid Prototyping
- Research
- Small Applications

Example

```python
classifier("Amazing product")
```

Very little code is required.

---

# 13. When Should You Avoid Pipeline?

Most production systems do not rely on Pipeline directly.

Instead they use

```python
AutoTokenizer

AutoModel
```

because they need

- Custom preprocessing
- Custom postprocessing
- Raw logits
- Hidden states
- Business rules
- Advanced batching
- Logging
- Monitoring
- API integration
- Model optimization

Example

```text
User Request

↓

FastAPI

↓

Authentication

↓

Custom Tokenizer Logic

↓

AutoTokenizer

↓

AutoModel

↓

Business Rules

↓

Database

↓

JSON Response
```

Pipeline hides many of these implementation details, making it less flexible for large production systems.

---

# Summary

| Feature | Supported |
|----------|-----------|
| Single Prediction | ✅ |
| Batch Prediction | ✅ |
| GPU Execution | ✅ |
| CPU Execution | ✅ |
| Custom Model | ✅ |
| Custom Tokenizer | ✅ |
| Automatic Softmax | ✅ |
| Automatic Padding | ✅ |
| Automatic Truncation | ✅ |
| Raw Logits | ❌ |
| Hidden States | ❌ |
| Attention Scores | ❌ |

---

# Key Takeaways

- Pipeline is designed for quick and easy inference.
- It supports both single and batch predictions.
- You can load any model from Hugging Face Hub.
- GPU execution requires only the `device` parameter.
- Padding and truncation are handled automatically.
- Pipeline hides low-level implementation details.
- For production applications requiring maximum flexibility, `AutoTokenizer` and `AutoModel` are generally preferred.

---

# Interview Questions

### Q1. Can Pipeline process multiple sentences?

**Answer**

Yes.

Pass a Python list instead of a single string.

---

### Q2. How do you use your own pretrained model?

```python
pipeline(
    "sentiment-analysis",
    model="your-model-name"
)
```

---

### Q3. How do you run Pipeline on GPU?

```python
pipeline(
    "sentiment-analysis",
    device=0
)
```

---

### Q4. Why is `batch_size` important?

It processes multiple samples together, improving throughput and GPU utilization.

---

### Q5. What is the purpose of `truncation=True`?

It prevents errors by shortening inputs that exceed the model's maximum supported sequence length.

---

### Q6. Why is padding necessary?

Models require all sequences in a batch to have the same length. Padding adds placeholder tokens to shorter sequences, while the attention mask ensures those tokens are ignored.

---

### Q7. Why do many production systems avoid using Pipeline?

Because production systems often need direct access to model internals, custom preprocessing/postprocessing, advanced batching, monitoring, and API-specific business logic that the high-level Pipeline API abstracts away.

---

## Next Chapter

The next topic is **Hugging Face Datasets Library**, where we'll learn how to efficiently load, preprocess, tokenize, and prepare datasets for training and inference with Transformers.
