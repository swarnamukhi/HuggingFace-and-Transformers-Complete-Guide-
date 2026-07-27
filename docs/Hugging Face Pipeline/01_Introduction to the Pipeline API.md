# Hugging Face Pipeline - Chapter 1
## Introduction to the Pipeline API

---

# Learning Objectives

After completing this chapter, you will be able to:

- Understand what the Hugging Face `pipeline()` API is.
- Explain why the Pipeline API was introduced.
- Understand how `pipeline()` differs from manually using `AutoTokenizer` and `AutoModel`.
- Create a basic sentiment analysis pipeline.
- Understand what happens internally when a prediction is made.
- Interpret the output returned by a pipeline.

---

# Prerequisites

Before learning this chapter, you should already know:

- Python Basics
- PyTorch Tensors
- AutoTokenizer
- AutoModel
- AutoModelForSequenceClassification
- Model Inference
- Logits
- Softmax

---

# What is a Pipeline?

The **Pipeline** is a high-level API provided by the Hugging Face `transformers` library.

Instead of manually performing every inference step, a pipeline combines all the common operations into a single object.

Without Pipeline, the developer is responsible for:

- Loading the tokenizer
- Loading the pretrained model
- Tokenizing the input text
- Creating tensors
- Running model inference
- Converting logits into probabilities
- Mapping prediction IDs to labels

Pipeline automates all these operations.

> **Important**
>
> Pipeline is **not** a model.
>
> Pipeline is a wrapper around several components that work together to perform inference.

---

# Why was Pipeline Introduced?

Suppose every developer had to write the following code for every NLP project.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

review = "I love this phone."

inputs = tokenizer(review, return_tensors="pt")

outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1)

label = model.config.id2label[prediction.item()]

print(label)
```

Although this approach provides complete control, it becomes repetitive because almost every NLP project follows the same inference workflow.

To simplify common use cases, Hugging Face introduced the Pipeline API.

---

# Manual Inference Workflow

```text
                Review
                   │
                   ▼
          AutoTokenizer
                   │
                   ▼
            BatchEncoding
                   │
                   ▼
AutoModelForSequenceClassification
                   │
                   ▼
                Logits
                   │
                   ▼
           torch.argmax()
                   │
                   ▼
      model.config.id2label
                   │
                   ▼
          Final Prediction
```

---

# Pipeline Workflow

Using a pipeline, the developer simply writes

```python
classifier(review)
```

Internally the pipeline performs

```text
                Review
                   │
                   ▼
               Pipeline
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
 Tokenizer      Model     Post-processing
      │            │            │
      └────────────┼────────────┘
                   ▼
           Final Prediction
```

Notice that the tokenizer and model are still being used.

Pipeline simply hides these implementation details.

---

# Creating a Pipeline

Import the pipeline class.

```python
from transformers import pipeline
```

Create a sentiment analysis pipeline.

```python
classifier = pipeline("sentiment-analysis")
```

At this stage, no prediction has been performed.

Instead, Hugging Face prepares everything required for inference.

---

# What Happens Internally?

When Python executes

```python
classifier = pipeline("sentiment-analysis")
```

the following sequence of operations occurs.

## Step 1

The requested task is identified.

```text
sentiment-analysis
```

---

## Step 2

The default pretrained model for that task is selected.

Example

```text
distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

---

## Step 3

The tokenizer corresponding to that model is loaded.

Internally this is equivalent to

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## Step 4

The pretrained sequence classification model is loaded.

Internally this is equivalent to

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

## Step 5

Both objects are stored inside the pipeline.

Conceptually

```text
classifier

├── tokenizer
├── model
├── task
├── preprocess()
├── forward()
└── postprocess()
```

The pipeline object is now ready to perform predictions.

---

# Running Inference

Predict the sentiment of a review.

```python
result = classifier("I love this phone.")
```

Only one line of code is written by the developer.

Internally, however, the pipeline performs several operations.

```text
Sentence
     │
     ▼
Tokenizer
     │
     ▼
Input IDs
Attention Mask
     │
     ▼
Model
     │
     ▼
Logits
     │
     ▼
Softmax
     │
     ▼
Highest Probability
     │
     ▼
Class Label
     │
     ▼
Dictionary Output
```

---

# Pipeline Output

The returned value looks like

```python
[
    {
        "label": "POSITIVE",
        "score": 0.9998
    }
]
```

Unlike our manual implementation, the pipeline automatically applies post-processing before returning the result.

---

# Why Does Pipeline Return a List?

Even if only one sentence is provided,

```python
classifier("Amazing movie")
```

the output is still

```python
[
    {
        "label": "POSITIVE",
        "score": 0.9998
    }
]
```

This design keeps the API consistent because the same pipeline also supports batch predictions.

Example

```python
classifier([
    "Amazing movie",
    "Worst phone",
    "Excellent camera"
])
```

returns

```python
[
    {"label":"POSITIVE","score":0.999},
    {"label":"NEGATIVE","score":0.998},
    {"label":"POSITIVE","score":0.997}
]
```

---

# Does Pipeline Apply Softmax?

Yes.

When using the model directly,

```python
outputs.logits
```

returns raw logits.

Pipeline automatically performs

```text
Logits

↓

Softmax

↓

Probability

↓

Highest Probability

↓

Class Label
```

Therefore, the developer directly receives

```python
{
    "label": "POSITIVE",
    "score": 0.9998
}
```

instead of raw logits.

---

# Manual Inference vs Pipeline

| Manual Implementation | Pipeline |
|-----------------------|----------|
| Load tokenizer | Automatic |
| Load model | Automatic |
| Tokenize input | Automatic |
| Create tensors | Automatic |
| Run inference | Automatic |
| Apply Softmax | Automatic |
| Map prediction ID | Automatic |
| Return confidence | Automatic |

---

# When Should You Use Pipeline?

Pipeline is an excellent choice for:

- Learning Hugging Face
- Research notebooks
- Rapid prototyping
- Demonstrations
- Small applications

For production systems requiring custom preprocessing, batching strategies, advanced logging, or direct access to logits and hidden states, engineers often use `AutoTokenizer` and `AutoModel` directly.

---

# Key Takeaways

- Pipeline is a high-level inference API.
- Pipeline is **not** a pretrained model.
- Pipeline automatically loads the tokenizer and model.
- Pipeline performs preprocessing, inference, and post-processing.
- Pipeline returns labels and confidence scores instead of raw logits.
- Pipeline simplifies inference by hiding repetitive implementation details.

---

# Interview Questions

### Q1. Is Pipeline a model?

**Answer**

No.

Pipeline is a wrapper that combines preprocessing, model inference, and post-processing into a single interface.

---

### Q2. Does Pipeline internally use AutoTokenizer?

**Answer**

Yes.

Pipeline automatically loads the tokenizer associated with the selected pretrained model.

---

### Q3. Does Pipeline always use AutoModel?

**Answer**

No.

Pipeline selects the appropriate model class based on the task.

Examples:

- Sentiment Analysis → `AutoModelForSequenceClassification`
- Question Answering → `AutoModelForQuestionAnswering`
- Text Generation → `AutoModelForCausalLM`

---

### Q4. Does Pipeline return logits?

**Answer**

No.

Pipeline returns processed predictions such as labels and confidence scores.

To access raw logits, use the model directly.

---

## Next Chapter

In the next chapter, we will cover the most important technical concepts of the Pipeline API, including:

- Batch Processing
- GPU Execution
- Custom Models
- Padding
- Truncation
- Framework Selection (PyTorch vs TensorFlow)
- When to Use and When to Avoid Pipeline
