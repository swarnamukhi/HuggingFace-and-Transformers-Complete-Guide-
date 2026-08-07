# Hugging Face Local Models with LangChain (Pipeline Integration)

## Objective

Learn how to integrate a Hugging Face model with LangChain using the **Transformers Pipeline** and understand the warnings that may appear during text generation.

---

# Architecture

```text
                  Hugging Face Hub
                        │
        (Download model if not available locally)
                        │
                        ▼
                Transformers Pipeline
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
   AutoTokenizer                  AutoModel
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
              HuggingFacePipeline
                        │
                        ▼
                 LangChain LLM
                        │
                        ▼
                    invoke()
```

---

# Step 1: Install Libraries

```python
!pip install -q transformers langchain langchain-huggingface accelerate sentencepiece
```

---

# Step 2: Import Required Libraries

```python
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
```

---

# Step 3: Create a Transformers Pipeline

```python
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)
```

### What happens internally?

```text
pipeline()

↓

Checks Local Cache

↓

Model Found?

├── Yes → Load locally
│
└── No
      │
      ▼
Connect to Hugging Face Hub

↓

Download

• Model Weights
• Tokenizer
• Configuration

↓

Store in Local Cache

↓

Create Pipeline Object
```

> **Note**
>
> `pipeline()` downloads the model only the **first time**.
> Afterwards it loads everything from the local cache.

---

# Step 4: Wrap the Pipeline with LangChain

```python
llm = HuggingFacePipeline(
    pipeline=pipe
)
```

Now the Transformers pipeline behaves like a LangChain LLM.

You can use

```python
response = llm.invoke("Explain Retrieval-Augmented Generation.")
print(response)
```

---

# Overall Flow

```text
Choose Model

↓

pipeline()

↓

Download (only first time)

↓

Load Tokenizer

↓

Load Model

↓

Create Pipeline

↓

HuggingFacePipeline

↓

LangChain

↓

invoke()

↓

Generated Response
```

---

# Warning Encountered

```text
Both max_new_tokens (=256) and max_length (=20) seem to have been set.

max_new_tokens will take precedence.
```

---

# Why did this happen?

We never wrote

```python
max_length=20
```

So where did it come from?

Internally, Transformers loads a **Generation Configuration**.

```text
pipeline()

↓

GenerationConfig

↓

max_length = 20 (default configuration)

↓

LangChain invoke()

↓

max_new_tokens = 256

↓

Conflict
```

Transformers now sees

```text
max_length = 20

AND

max_new_tokens = 256
```

Both parameters control the output length.

Therefore it prints a warning.

---

# Why is it a contradiction?

Suppose

Prompt contains

```text
10 tokens
```

### max_length = 20

means

```text
Prompt Tokens + Generated Tokens = 20
```

So only

```text
10 new tokens
```

can be generated.

---

### max_new_tokens = 256

means

```text
Generate 256 new tokens
```

regardless of prompt length.

Now both cannot be true simultaneously.

```text
Total Tokens = 20

AND

Generate 256 new tokens
```

Hence the warning.

---

# What does Transformers do?

Instead of throwing an error, Transformers decides

```text
Ignore max_length

↓

Use max_new_tokens
```

Therefore

```text
max_new_tokens takes precedence
```

---

# Is this a LangChain Warning?

❌ No.

The warning comes from the **Transformers library**.

Flow

```text
Your Code

↓

Transformers Pipeline

↓

Generation Configuration

↓

Warning Printed

↓

LangChain receives Pipeline
```

LangChain is **not** generating the warning.

---

# Second Warning

```text
Ignoring clean_up_tokenization_spaces=True
for BPE tokenizer Qwen2Tokenizer
```

---

# Why?

Qwen uses a

```text
Byte Pair Encoding (BPE)
```

tokenizer.

The option

```python
clean_up_tokenization_spaces=True
```

was originally designed for

```text
WordPiece Tokenizers

(BERT)
```

Using it on BPE tokenizers can modify the generated text incorrectly.

Therefore Transformers ignores it.

---

# Is this an error?

No.

It is simply informing us that

```text
Cleanup is skipped because Qwen uses BPE.
```

---

# Which component is responsible?

| Warning | Responsible |
|----------|-------------|
| max_length vs max_new_tokens | Transformers Generation Configuration |
| clean_up_tokenization_spaces | Transformers + Qwen Tokenizer |
| LangChain | Not responsible |

---

# How to Remove the Warnings

## Solution 1 (Recommended)

Explicitly specify

```python
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=256,
    clean_up_tokenization_spaces=False
)
```

This avoids relying on the default generation configuration and suppresses the BPE cleanup warning.

---

## Solution 2

Specify generation parameters during invocation.

```python
response = llm.invoke(
    "Explain Retrieval-Augmented Generation.",
    max_new_tokens=256
)
```

---

# Best Practice

For modern Decoder-only LLMs

- GPT
- Llama
- Gemma
- Qwen
- Mistral

Prefer

```python
max_new_tokens
```

instead of

```python
max_length
```

Reason

```text
max_new_tokens

↓

Controls only the generated response.

max_length

↓

Controls

Prompt + Generated Tokens

which is often not what we want.
```

---

# Key Learning

## Hosted Inference API

```text
LangChain

↓

ChatHuggingFace

↓

Hugging Face Server

↓

Model
```

No model is downloaded locally.

Most generation settings are handled by Hugging Face.

---

## Transformers Pipeline

```text
Transformers

↓

Download Model

↓

Run Locally

↓

Generation Configuration

↓

Warnings (if any)

↓

LangChain
```

The model runs on your own machine after downloading.

---

# Summary

- `pipeline()` downloads the model only the first time and caches it locally.
- `HuggingFacePipeline` wraps the Transformers pipeline so it can be used with LangChain.
- The warning about `max_length` is caused by the **Transformers Generation Configuration**, not by LangChain.
- `max_new_tokens` overrides `max_length` when both are present.
- The BPE cleanup warning occurs because Qwen uses a **BPE tokenizer**, where that cleanup option is not appropriate.
- For modern LLMs, prefer using `max_new_tokens` and set `clean_up_tokenization_spaces=False` when creating the pipeline.
