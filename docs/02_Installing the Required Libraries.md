# Installing the Required Libraries

Before we start working with Hugging Face models, we first need to install a few Python libraries.

```bash
!pip install -q transformers datasets torch sentencepiece sacremoses librosa soundfile accelerate
```

At first glance, this command may look like a simple installation command, but it actually installs several powerful libraries developed by different organizations. Understanding what each library provides will help us understand the Hugging Face ecosystem.

---

# What is `pip`?

`pip` stands for **Pip Installs Packages**.

It is Python's official package manager used to download and install external libraries.

Instead of writing everything from scratch, developers install libraries that already contain the required functionality.

For example, rather than implementing the complete Transformer architecture ourselves, we simply install the required library and use it.

When we execute:

```bash
pip install transformers
```

Python performs the following steps:

```
Your Computer
      │
      ▼
     pip
      │
      ▼
Python Package Index (PyPI)
      │
      ▼
Downloads the package
      │
      ▼
Installs it into your Python environment
```

After installation, the package becomes available for import in Python programs.

Example:

```python
from transformers import AutoTokenizer
```

---

# Where Do These Libraries Come From?

Many beginners assume that every library in the installation command is developed by Hugging Face.

This is **not true**.

Different organizations have developed different libraries, and Hugging Face integrates many of them into a single ecosystem.

| Library | Developed By | Purpose |
|----------|--------------|---------|
| `transformers` | Hugging Face | Provides pretrained Transformer models and APIs |
| `datasets` | Hugging Face | Loads and processes datasets |
| `accelerate` | Hugging Face | Simplifies training on CPU, GPU, TPU, and multiple GPUs |
| `torch` | Meta (Facebook AI Research) | Deep Learning framework used to build and train neural networks |
| `sentencepiece` | Google | Tokenization algorithm used by models such as T5 and LLaMA |
| `sacremoses` | Open Source Community | Text preprocessing and tokenization utilities |
| `librosa` | Brian McFee & Contributors | Audio processing library |
| `soundfile` | Open Source Community | Reads and writes audio files |

From the table above, we can see that Hugging Face develops some libraries, while other libraries are developed by different organizations.

Together, they form a complete ecosystem for building AI applications.

---

# What Has Hugging Face Developed?

Hugging Face is an AI company that develops open-source tools for Machine Learning and Natural Language Processing.

Some of its most popular libraries include:

- `transformers`
- `datasets`
- `accelerate`
- `tokenizers`
- `diffusers`
- `evaluate`
- `peft`
- `trl`

These libraries work together to simplify developing, training, fine-tuning, and deploying AI models.

---

# Which Library Are We Going to Learn First?

Among all the installed libraries, the most important one for this course is:

```python
transformers
```

The `transformers` library provides ready-to-use classes for working with pretrained Transformer models.

Some of the most commonly used classes include:

| Class | Purpose |
|--------|---------|
| `AutoTokenizer` | Automatically loads the correct tokenizer for a pretrained model |
| `AutoModel` | Loads the base Transformer model |
| `AutoModelForSequenceClassification` | Loads models for text classification |
| `AutoModelForQuestionAnswering` | Loads models for question answering |
| `AutoModelForCausalLM` | Loads models for text generation |
| `Trainer` | Simplifies model training |
| `TrainingArguments` | Configures the training process |
| `pipeline` | High-level API for inference |

Notice that we are **not importing a model**.

We are importing **Python classes** provided by the `transformers` library.

For example,

```python
from transformers import AutoTokenizer
```

does **not** download BERT or GPT.

It simply imports the `AutoTokenizer` class into our program.

Later, when we execute:

```python
AutoTokenizer.from_pretrained("bert-base-uncased")
```

the class automatically downloads and loads the correct tokenizer for the selected model.

Similarly,

```python
AutoModel.from_pretrained("bert-base-uncased")
```

downloads the pretrained BERT model.

---

# Summary

- `pip` is Python's package manager.
- Libraries are downloaded from the Python Package Index (PyPI).
- Not every installed library is developed by Hugging Face.
- Hugging Face develops libraries such as `transformers`, `datasets`, and `accelerate`.
- The `transformers` library provides classes for loading and using pretrained Transformer models.
- Importing a class is different from downloading a pretrained model.
- The first class we will study in detail is **AutoTokenizer**.

---

## Next Topic

**AutoTokenizer**
- Why do we need it?
- Why can't a Transformer process raw text directly?
- How does AutoTokenizer work internally?
- What happens inside `from_pretrained()`?
