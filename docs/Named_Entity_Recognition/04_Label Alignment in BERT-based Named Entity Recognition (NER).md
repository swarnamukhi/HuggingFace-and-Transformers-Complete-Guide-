# Label Alignment in BERT-based Named Entity Recognition (NER)

## The Question

A common question while learning NER is:

> If the dataset already tells us that **"Washington"** belongs to the label **B-LOC**, then why does BERT tokenize it into **"Wash"** and **"##ington"**?

The answer lies in how BERT was pretrained.

---

# Step 1: Human Annotation

Consider the sentence:

```
I visited Washington last summer.
```

Before any machine learning model is involved, a **human annotator** reads the sentence and assigns NER labels.

| Word | NER Label |
|------|-----------|
| I | O |
| visited | O |
| Washington | B-LOC |
| last | O |
| summer | O |

The dataset stores:

```python
tokens = [
    "I",
    "visited",
    "Washington",
    "last",
    "summer"
]

ner_tags = [
    0,
    0,
    5,
    0,
    0
]
```

At this stage:

- No BERT
- No Transformer
- No Tokenizer

Only human annotation.

---

# Step 2: Why Doesn't BERT Use "Washington" Directly?

BERT was pretrained using **WordPiece Tokenization**.

Its vocabulary does not necessarily contain every possible English word.

Instead, BERT may split

```
Washington
```

into

```
Wash
##ington
```

because those subwords already exist in its vocabulary.

This allows BERT to represent millions of words using a much smaller vocabulary.

---

# Step 3: The Label Does Not Change

Originally,

```
Washington → B-LOC
```

After BERT tokenization,

```
Washington
      │
      ▼
Wash
##ington
```

The entity is still **Washington**.

The tokenizer only split the word because of BERT's vocabulary.

The NER label remains the same entity.

---

# Step 4: Label Alignment

Now we have two BERT tokens but only one original label.

We must align the labels.

### Original Dataset

| Token | Label |
|-------|-------|
| Washington | B-LOC |

### After BERT Tokenization

| BERT Token | Label |
|------------|-------|
| Wash | B-LOC |
| ##ington | ? |

This process is called **Label Alignment**.

---

# Common Label Alignment Strategies

## Strategy 1 (Most Common - Hugging Face)

Assign the original label only to the first subword.

Ignore the remaining subwords during loss calculation.

| Token | Label |
|--------|-------|
| Wash | B-LOC |
| ##ington | -100 |

Here,

```
-100
```

means

> Ignore this token while computing the training loss.

This is the strategy used in most Hugging Face NER examples.

---

## Strategy 2

Assign labels to every subword.

| Token | Label |
|--------|-------|
| Wash | B-LOC |
| ##ington | I-LOC |

Some research papers and implementations follow this approach.

---

# Why Doesn't Hugging Face Label Every Subword?

Even though the second token receives **-100**, BERT still processes it.

```
Wash --------\
              \
               ---> BERT ---> Prediction
              /
##ington ----/
```

Both subwords contribute to the contextual representation.

Only the first subword contributes to the loss calculation.

---

# Important Clarification

The tokenizer **never decides** the NER label.

The tokenizer only decides how to split the word.

Example:

```
Washington
```

↓

```
Wash
##ington
```

The NER label

```
B-LOC
```

was already assigned by the human annotator before tokenization.

---

# Complete Pipeline

```
Original Sentence
        │
        ▼
Human Annotation
        │
        ▼
Word Tokens + NER Labels
        │
        ▼
BERT WordPiece Tokenizer
        │
        ▼
Subword Tokens
        │
        ▼
Label Alignment
        │
        ▼
BERT Training
```

---

# Key Takeaways

- Human annotators assign NER labels to complete words.
- The dataset stores word-level tokens and labels.
- BERT uses WordPiece tokenization because of its pretrained vocabulary.
- Some words are split into multiple subwords.
- Label Alignment transfers the original word label to the BERT tokens.
- Hugging Face usually assigns the original label to the first subword and **-100** to the remaining subwords.
- The tokenizer never predicts labels; it only converts words into BERT-compatible tokens.

---

# Summary

Human Annotation

```
Washington
      │
      ▼
B-LOC
```

↓

BERT Tokenizer

```
Washington
      │
      ▼
Wash
##ington
```

↓

Label Alignment (Hugging Face)

```
Wash      → B-LOC
##ington  → -100
```

↓

Fine-tune BERT for Named Entity Recognition.
