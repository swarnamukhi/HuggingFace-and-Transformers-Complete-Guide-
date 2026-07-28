# Named Entity Recognition (NER) - Chapter 3
# Choosing, Customizing, and Fine-Tuning NER Models

---

# Learning Objectives

After completing this chapter, you will be able to:

- Understand when a pretrained NER model is sufficient.
- Learn when fine-tuning is required.
- Understand why companies create custom entity labels.
- Learn how BIO annotation fits into the fine-tuning process.
- Know how to evaluate whether a Hugging Face NER model matches your business problem.

---

# Prerequisites

Before reading this chapter, you should understand:

- NER Fundamentals
- How NER Models Work
- Transformer Encoder Models
- Token Classification

---

# The First Question in Every NER Project

The first question is **not**

> Which model should I download?

Instead, it is

> What entities does my business want to extract?

Everything else depends on the answer.

---

# Step 1: Identify the Business Problem

Suppose a company wants to process customer complaints.

Example

```text
Customer Ram Kumar's Airtel SIM stopped working on 10 June.
```

The business wants to extract

- Customer Name
- Company
- Product
- Date

The required entities determine whether an existing model can be used.

---

# Can I Use a Pretrained NER Model?

The answer depends on the required entity types.

## Example

Business Requirements

- Person
- Organization
- Location
- Date

Many pretrained NER models already support these labels.

Workflow

```text
Business Text

↓

Pretrained NER Model

↓

Extract Entities
```

No additional training is required.

---

# When is Fine-Tuning Required?

Fine-tuning becomes necessary when the business requires entity types that are not supported by the pretrained model.

Telecom Example

Business wants to identify

- Customer Name
- SIM Number
- IMEI Number
- Tower ID
- Plan Name

Most general NER models do not contain these labels.

Therefore,

the model must be customized.

---

# Why Create Custom Labels?

The output layer of an NER model can only predict labels that it learned during training.

If the business introduces new entity types,

the model must also learn these new labels.

Example

```text
B-CUSTOMER_NAME
I-CUSTOMER_NAME

B-SIM_NUMBER

B-IMEI

B-TOWER_ID

B-PLAN_NAME

O
```

These labels are designed by the engineering team according to business requirements.

---

# Who Creates These Labels?

Usually,

Business Experts identify what information needs to be extracted.

Machine Learning Engineers design the label schema.

Example

Healthcare

```text
B-DISEASE
I-DISEASE

B-DRUG
I-DRUG

B-SYMPTOM
```

Telecom

```text
B-CUSTOMER_NAME
I-CUSTOMER_NAME

B-SIM_NUMBER

B-IMEI

B-TOWER_ID
```

Banking

```text
B-ACCOUNT_NUMBER

B-CREDIT_CARD

B-IFSC_CODE
```

Every domain defines labels according to its business needs.

---

# Why BIO Annotation is Required

During fine-tuning,

the model must know the correct answer.

Suppose the sentence is

```text
Customer Ram Kumar's Airtel SIM stopped working.
```

The annotated training data becomes

| Token | Label |
|--------|-------|
| Customer | O |
| Ram | B-CUSTOMER_NAME |
| Kumar | I-CUSTOMER_NAME |
| Airtel | B-ORG |
| SIM | B-PRODUCT |
| stopped | O |
| working | O |

These BIO labels become the ground truth used during training.

Without labeled examples,

the model cannot learn the new entity types.

---

# Industry Workflow

```text
Business Requirement

↓

Identify Required Entities

↓

Check Existing NER Models

↓

Are All Required Labels Available?

↓

Yes ----------------------► Use Pretrained Model

No

↓

Define Custom Labels

↓

Annotate Dataset (BIO)

↓

Fine-Tune Pretrained Model

↓

Evaluate Model

↓

Deploy
```

---

# Structured vs Unstructured Data

## Structured Data

Example

| CustomerName | Company | Date |

The information is already stored in separate columns.

NER is generally unnecessary.

---

## Unstructured Data

Example

```text
Customer Ram Kumar's Airtel SIM stopped working yesterday.
```

The required information is embedded inside free text.

NER becomes useful.

---

# Can NER be Used with CSV Files?

Yes.

The deciding factor is **not the file format**.

The deciding factor is **the content inside the columns**.

Example

CSV

| Complaint |

```text
My Airtel broadband stopped working yesterday.
```

Although the data is stored in a CSV,

the column contains unstructured text.

NER can extract entities from this text.

---

# How Do I Know What a Pretrained Model Can Predict?

Every Hugging Face model publishes its supported labels.

Example

```python
model.config.id2label
```

Possible output

```text
O
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
```

If the required labels are missing,

the model cannot predict them.

---

# Questions Every ML Engineer Should Ask

Before selecting an NER model,

always ask:

- What entities does the business require?
- Does a pretrained model already support these entities?
- Is the model trained on a similar domain?
- Are custom labels required?
- Do we need additional annotation?
- Is fine-tuning necessary?

These questions should be answered before writing any code.

---

# Key Takeaways

- Business requirements determine the entity labels.
- Pretrained NER models are used whenever possible.
- Fine-tuning is required when new entity types are introduced.
- BIO annotation provides the training labels.
- CSV files can contain structured or unstructured data.
- The file format does not determine whether NER is applicable.
- Always verify the labels supported by a pretrained model before selecting it.

---

# Interview Questions

### Q1. Should every NER project be fine-tuned?

No.

If a pretrained model already supports the required entity types and performs well, it can be used directly.

---

### Q2. Why do companies create custom entity labels?

Because business-specific entities such as SIM Number, IMEI, or Account Number are usually not part of general NER datasets.

---

### Q3. Why is BIO annotation required?

BIO labels provide the ground truth that the model learns from during fine-tuning.

---

### Q4. Can NER be applied to CSV files?

Yes, if the CSV contains unstructured text.

If the information is already organized into columns, NER is usually unnecessary.

---

### Q5. What is the first step before choosing an NER model?

Understand the business requirements and identify the entities that need to be extracted.

---

## Next Chapter

In the next chapter, we will study **BIO Tagging**, where we will understand how entity boundaries are represented and why Transformer-based NER models predict BIO labels for every token.
