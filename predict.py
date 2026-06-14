import os
import pandas as pd
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Preprocessing function
def preprocess_row(row):
    desc = str(row['Ticket Description'])
    prod = str(row['Product Purchased'])
    desc = desc.replace('{product_purchased}', prod)
    subject = str(row['Ticket Subject']) if pd.notna(row['Ticket Subject']) else ""
    return subject, desc, subject + " " + desc

# NLP Score function
def calculate_nlp_score(text, words, phrases):
    text = text.lower()
    score = 0.0
    for w in words:
        if w in text:
            score += 2.0
    for p in phrases:
        if p in text:
            score += 1.5
    negations = ["not", "no", "never", "cannot", "doesn't", "don't", "wasn't"]
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in negations and i + 1 < len(tokens):
            next_tok = tokens[i+1]
            if next_tok in ["working", "charging", "turning", "responding", "connecting", "opening"]:
                score += 1.5
    return score

def main():
    parser = argparse.ArgumentParser(description="Inference script for Support Integrity Auditor")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Path to output predictions CSV")
    parser.add_argument("--output-json", type=str, default="dossiers.json", help="Path to output evidence dossiers JSON")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
        
    print("Loading pipeline metadata...")
    if not os.path.exists("pipeline_metadata.joblib"):
        print("Error: pipeline_metadata.joblib not found. Run train_pipeline.py first.")
        return
    meta = joblib.load("pipeline_metadata.joblib")
    
    print("Loading fine-tuned classifier...")
    if not os.path.exists("./sia_model"):
        print("Error: ./sia_model directory not found. Run train_pipeline.py first.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("./sia_model")
    classifier = AutoModelForSequenceClassification.from_pretrained("./sia_model")
    classifier.to(device)
    classifier.eval()
    
    print("Loading sentence encoder...")
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Read input CSV
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tickets from {args.input}.")
    
    # Preprocess
    df['Cleaned Subject'] = df['Ticket Subject'].fillna("").astype(str)
    
    processed_texts = []
    full_texts = []
    cleaned_descs = []
    
    for idx, row in df.iterrows():
        subj, desc, full_txt = preprocess_row(row)
        cleaned_descs.append(desc)
        full_texts.append(full_txt)
        # Format input for classifier
        input_text = f"Channel: {row['Ticket Channel']} | Product: {row['Product Purchased']} | Subject: {subj} | Description: {desc}"
        processed_texts.append(input_text)
        
    df['Cleaned Description'] = cleaned_descs
    df['Full Text'] = full_texts
    df['InputText'] = processed_texts
    
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    
    # Run classifier predictions in batches
    print("Running classification...")
    mismatch_predictions = []
    confidences = []
    
    batch_size = 32
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i+batch_size]
        batch_rows = df.iloc[i:i+batch_size]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = classifier(**encodings)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Predict Mismatch using XOR identity
            for j, (_, row) in enumerate(batch_rows.iterrows()):
                prio_name = row['Ticket Priority']
                prio_high = 1 if priority_map.get(prio_name, 1) >= 2 else 0
                
                prob_urgent_0 = probs[j, 0]
                prob_urgent_1 = probs[j, 1]
                
                # Mismatch = Urgent ^ Priority High
                if prio_high == 0:
                    prob_mismatch_0 = prob_urgent_0
                    prob_mismatch_1 = prob_urgent_1
                else:
                    prob_mismatch_0 = prob_urgent_1
                    prob_mismatch_1 = prob_urgent_0
                    
                pred_mismatch = 1 if prob_mismatch_1 > prob_mismatch_0 else 0
                confidence = prob_mismatch_1 if pred_mismatch == 1 else prob_mismatch_0
                
                mismatch_predictions.append(pred_mismatch)
                confidences.append(confidence)
                
    df['Mismatch_Pred'] = mismatch_predictions
    df['Confidence'] = confidences
    
    # Run Stage 1 logic to compute inferred severity for dossiers
    print("Extracting signals for evidence grounding...")
    # Signal 1
    df['Signal_1_Raw'] = df['Full Text'].apply(lambda x: calculate_nlp_score(x, meta['high_urgency_words'], meta['escalation_phrases']))
    df['Signal_1'] = (df['Signal_1_Raw'] - meta['s1_min']) / (meta['s1_max'] - meta['s1_min'] + 1e-9)
    df['Signal_1'] = df['Signal_1'].clip(0.0, 1.0)
    
    # Signal 2
    embeddings = model_st.encode(df['Full Text'].tolist(), show_progress_bar=False)
    clusters = meta['kmeans'].predict(embeddings)
    df['Signal_2_Cluster'] = pd.Series(clusters).map(meta['cluster_rank_map']).fillna(0.0).values
    
    cos_sim_high = np.dot(embeddings, meta['ref_high_emb']) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(meta['ref_high_emb']) + 1e-9)
    cos_sim_low = np.dot(embeddings, meta['ref_low_emb']) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(meta['ref_low_emb']) + 1e-9)
    df['Signal_2_Sim'] = (cos_sim_high - cos_sim_low + 1.0) / 2.0
    df['Signal_2'] = 0.5 * df['Signal_2_Cluster'] + 0.5 * df['Signal_2_Sim']
    df['Signal_2'] = df['Signal_2'].clip(0.0, 1.0)
    
    # Signal 3
    df['Pred_Resolution_Duration'] = meta['reg'].predict(embeddings)
    df['Signal_3'] = (df['Pred_Resolution_Duration'] - meta['s3_min']) / (meta['s3_max'] - meta['s3_min'] + 1e-9)
    df['Signal_3'] = df['Signal_3'].clip(0.0, 1.0)
    
    # Fusion
    df['Fusion_Score'] = (df['Signal_1'] + df['Signal_2'] + df['Signal_3']) / 3.0
    
    def get_inferred_severity(score):
        if score < meta['q25']:
            return "Low"
        elif score < meta['q50']:
            return "Medium"
        elif score < meta['q75']:
            return "High"
        else:
            return "Critical"
            
    df['Inferred_Severity'] = df['Fusion_Score'].apply(get_inferred_severity)
    
    # Generate Evidence Dossiers for flagged tickets
    dossiers = []
    mismatch_types = []
    
    severity_levels = ["Low", "Medium", "High", "Critical"]
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    
    for idx, row in df.iterrows():
        prio_name = row['Ticket Priority']
        inf_name = row['Inferred_Severity']
        
        prio_val = priority_map.get(prio_name, 1)
        inf_val = priority_map.get(inf_name, 1)
        
        is_mismatch = row['Mismatch_Pred'] == 1
        prio_high = 1 if prio_val >= 2 else 0
        is_urgent_pred = 1 if is_mismatch ^ (prio_high == 1) else 0
        
        if is_mismatch:
            if is_urgent_pred == 1:
                mismatch_type = "Hidden Crisis"
                if inf_name not in ["High", "Critical"]:
                    inf_name = "High"
            else:
                mismatch_type = "False Alarm"
                if inf_name not in ["Low", "Medium"]:
                    inf_name = "Medium"
            inf_val = priority_map.get(inf_name, 1)
            df.at[idx, 'Inferred_Severity'] = inf_name
        else:
            mismatch_type = "Consistent"
            
        mismatch_types.append(mismatch_type)
        
        if is_mismatch:
            # Generate feature evidence list
            feature_evidence = []
            
            # 1. Keywords matched
            matched_keywords = []
            text_lower = row['Full Text'].lower()
            for w in meta['high_urgency_words']:
                if w in text_lower:
                    matched_keywords.append(w)
            for p in meta['escalation_phrases']:
                if p in text_lower:
                    matched_keywords.append(p)
                    
            matched_keywords = list(set(matched_keywords))
            
            if matched_keywords:
                kw_str = ", ".join(matched_keywords[:3]) # take top 3
                feature_evidence.append({
                    "signal": "keyword",
                    "value": kw_str,
                    "weight": f"{row['Signal_1']:.2f}"
                })
            else:
                feature_evidence.append({
                    "signal": "keyword",
                    "value": "None detected",
                    "weight": "0.00"
                })
                
            # 2. Resolution time
            res_val = f"{row['Pred_Resolution_Duration']:.1f} hours (predicted)"
            interpretation = "Expected resolution duration is high, indicating significant complexity."
            if row['Pred_Resolution_Duration'] < 10.0:
                interpretation = "Expected resolution duration is low, indicating a standard request."
                
            feature_evidence.append({
                "signal": "resolution_time",
                "value": res_val,
                "interpretation": interpretation
            })
            
            # Constraint analysis explanation
            delta_val = inf_val - prio_val
            severity_delta_str = f"{delta_val:+d}"
            
            if mismatch_type == "Hidden Crisis":
                explanation = f"The ticket was assigned '{prio_name}' priority, but the description indicates a critical issue regarding '{row['Cleaned Subject']}' for product '{row['Product Purchased']}'. The presence of urgent indicators like '{', '.join(matched_keywords[:2])}' and a predicted resolution time of {row['Pred_Resolution_Duration']:.1f} hours support a higher inferred severity of '{inf_name}', indicating a Hidden Crisis."
            else:
                explanation = f"The ticket was assigned '{prio_name}' priority, but the description indicates a standard or minor issue regarding '{row['Cleaned Subject']}' for product '{row['Product Purchased']}'. No high-urgency keywords were active, and the predicted resolution time is {row['Pred_Resolution_Duration']:.1f} hours, suggesting a lower inferred severity of '{inf_name}' and representing a False Alarm."
                
            dossier = {
                "ticket_id": str(row['Ticket ID']),
                "assigned_priority": prio_name,
                "inferred_severity": inf_name,
                "mismatch_type": mismatch_type,
                "severity_delta": severity_delta_str,
                "feature_evidence": feature_evidence,
                "constraint_analysis": explanation,
                "confidence": f"{row['Confidence']:.2f}"
            }
            dossiers.append(dossier)
            
    df['Mismatch_Type'] = mismatch_types
    
    # Save CSV predictions
    output_cols = ['Ticket ID', 'Ticket Priority', 'Inferred_Severity', 'Mismatch_Pred', 'Mismatch_Type', 'Confidence']
    df[output_cols].to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}.")
    
    # Save JSON dossiers
    with open(args.output_json, 'w') as f:
        json.dump(dossiers, f, indent=2)
    print(f"Evidence dossiers for {len(dossiers)} flagged tickets saved to {args.output_json}.")

if __name__ == "__main__":
    main()
