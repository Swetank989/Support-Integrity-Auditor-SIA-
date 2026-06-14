# Support Integrity Auditor (SIA)

Have you ever submitted a critical support ticket only for it to sit in a queue marked as "Low" priority? Or seen a simple password reset ticket labeled as "Critical"? 

In customer support, priority mistakes happen all the time due to agent fatigue or favoritism. The **Support Integrity Auditor (SIA)** is a smart assistant designed to audit support tickets automatically, identifying priority mismatches:
*   🚨 **Hidden Crises**: Urgent tickets mislabeled as Low or Medium priority.
*   ⚠️ **False Alarms**: Simple, non-urgent tickets inflated to High or Critical priority.

This system achieves **91% accuracy** in detecting these mismatches, helping support teams prioritize the right tickets at the right time.

---

## 🛠️ How It Works (In Simple Terms)

The system works in three main stages:

### Stage 1: Finding the "True" Severity of a Ticket
Because we don't have human-labeled mismatch examples, the system calculates its own "true severity" score for each ticket by averaging three signals:
1.  **Key Phrases**: Looking for urgent words (like "outage", "broken", "blocked") and customer frustration phrases (like "cancel my subscription", "waiting for days").
2.  **Sentence Similarity**: Comparing the ticket text to example "extreme" sentences (like *"system is completely down"*) to see how closely they match.
3.  **Resolution Time**: Using historical data to predict how long the ticket would take to resolve. Complex tickets usually take longer.

We combine these three signals into a score from 0 to 3: **Low, Medium, High, or Critical**.

### Stage 2: Training the AI
We use the true severity scores from Stage 1 to train a fine-tuned language model (**DistilBERT**). This model reads the ticket's subject and description and learns to classify whether a ticket is **Urgent** or **Standard**.

### Stage 3: Creating the Evidence Dossier
Whenever the AI detects a priority mismatch, it generates an **Evidence Dossier** (a structured JSON report). This report outlines exactly *why* the ticket was flagged, listing matched keywords, expected resolution times, and a plain-English explanation—giving support managers the confidence to re-triage the ticket.

---

## 💡 The "XOR Bottleneck" & Our Smart Fix
When we first trained the AI to predict "mismatch" directly, it struggled heavily, achieving only ~75% accuracy. 

**Why did this happen?**
Mismatch is a logical puzzle (an XOR operation):
*   If a ticket is Urgent AND priority is Low $\rightarrow$ Mismatch
*   If a ticket is Standard AND priority is High $\rightarrow$ Mismatch
*   If they match $\rightarrow$ No Mismatch

Forcing the AI to learn both the text meaning and this logical puzzle at the same time was too confusing for the model. 

**Our Fix:**
We trained the AI to do one simple thing: predict if the text is **Urgent or Standard**. Then, during inference, we compare the AI's prediction with the assigned priority in code. This simple design change bypassed the bottleneck and raised our accuracy from **75% to 91%**!

---

## 📊 Performance Results
On our evaluation dataset, the system achieved the following scores:
*   **Accuracy**: **91.1%** (How often the system correctly flags mismatches)
*   **Macro F1**: **91.0%** (Balanced accuracy across both classes)
*   **Consistent Recall**: **91.0%** (Correctly identifying consistent tickets)
*   **Mismatch Recall**: **90.0%** (Correctly identifying mismatched tickets)

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
To preprocess the data and train the classifier, run:
```bash
python train_pipeline.py
```
*This will train the model on your GPU (or CPU if GPU is unavailable) and save it to the `./sia_model/` folder.*

### 3. Run Predictions on a File
To run predictions on a CSV file of new tickets and output the results and evidence dossiers:
```bash
python predict.py --input eval_processed.csv --output-csv predictions.csv --output-json dossiers.json
```

### 4. Open the Web Dashboard
To run the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
This opens a dashboard in your browser where you can:
- Type in a custom ticket to test the AI.
- Upload a batch CSV to view a **Severity Delta Heatmap** (showing which ticket channels have the most priority mismatches).
