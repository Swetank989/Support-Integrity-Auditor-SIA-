import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(
    page_title="Support Integrity Auditor (SIA)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources with caching
@st.cache_resource
def load_models():
    if not os.path.exists("./sia_model") or not os.path.exists("pipeline_metadata.joblib"):
        return None, None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("./sia_model")
    classifier = AutoModelForSequenceClassification.from_pretrained("./sia_model")
    classifier.to(device)
    classifier.eval()
    
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    meta = joblib.load("pipeline_metadata.joblib")
    
    return tokenizer, classifier, model_st, meta

tokenizer, classifier, model_st, meta = load_models()

# Mapping configurations
priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
severity_levels = ["Low", "Medium", "High", "Critical"]

# Helper functions
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

def audit_single_ticket(row_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj = str(row_dict.get('Ticket Subject', ''))
    desc = str(row_dict.get('Ticket Description', ''))
    prod = str(row_dict.get('Product Purchased', ''))
    channel = str(row_dict.get('Ticket Channel', ''))
    prio_name = str(row_dict.get('Ticket Priority', ''))
    ticket_type = str(row_dict.get('Ticket Type', ''))
    
    desc_clean = desc.replace('{product_purchased}', prod)
    full_text = subj + " " + desc_clean
    
    # NLP Score (Signal 1)
    s1_raw = calculate_nlp_score(full_text, meta['high_urgency_words'], meta['escalation_phrases'])
    s1 = np.clip((s1_raw - meta['s1_min']) / (meta['s1_max'] - meta['s1_min'] + 1e-9), 0.0, 1.0)
    
    # Embedding Similarity + Cluster (Signal 2)
    emb = model_st.encode([full_text], show_progress_bar=False)
    cluster = meta['kmeans'].predict(emb)[0]
    s2_cluster = meta['cluster_rank_map'].get(cluster, 0.0)
    
    cos_sim_high = np.dot(emb, meta['ref_high_emb'].reshape(-1, 1))[0, 0] / (np.linalg.norm(emb) * np.linalg.norm(meta['ref_high_emb']) + 1e-9)
    cos_sim_low = np.dot(emb, meta['ref_low_emb'].reshape(-1, 1))[0, 0] / (np.linalg.norm(emb) * np.linalg.norm(meta['ref_low_emb']) + 1e-9)
    s2_sim = (cos_sim_high - cos_sim_low + 1.0) / 2.0
    s2 = np.clip(0.5 * s2_cluster + 0.5 * s2_sim, 0.0, 1.0)
    
    # Resolution Time Regression (Signal 3)
    pred_duration = meta['reg'].predict(emb)[0]
    s3 = np.clip((pred_duration - meta['s3_min']) / (meta['s3_max'] - meta['s3_min'] + 1e-9), 0.0, 1.0)
    
    # Fusion
    fusion_score = (s1 + s2 + s3) / 3.0
    
    # Inferred Severity
    if fusion_score < meta['q25']:
        inf_name = "Low"
    elif fusion_score < meta['q50']:
        inf_name = "Medium"
    elif fusion_score < meta['q75']:
        inf_name = "High"
    else:
        inf_name = "Critical"
        
    prio_val = priority_map.get(prio_name, 1)
    inf_val = priority_map.get(inf_name, 1)
    
    # Classifier Prediction
    input_text = f"Channel: {channel} | Product: {prod} | Subject: {subj} | Description: {desc_clean}"
    encodings = tokenizer([input_text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = classifier(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
    prio_high = 1 if prio_val >= 2 else 0
    prob_urgent_0 = probs[0]
    prob_urgent_1 = probs[1]
    
    # Mismatch = Urgent ^ Priority High
    if prio_high == 0:
        prob_mismatch_0 = prob_urgent_0
        prob_mismatch_1 = prob_urgent_1
    else:
        prob_mismatch_0 = prob_urgent_1
        prob_mismatch_1 = prob_urgent_0
        
    pred = 1 if prob_mismatch_1 > prob_mismatch_0 else 0
    confidence = prob_mismatch_1 if pred == 1 else prob_mismatch_0
        
    is_mismatch = (pred == 1)
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
    else:
        mismatch_type = "Consistent"
        
    # Generate Dossier
    feature_evidence = []
    matched_keywords = []
    text_lower = full_text.lower()
    for w in meta['high_urgency_words']:
        if w in text_lower:
            matched_keywords.append(w)
    for p in meta['escalation_phrases']:
        if p in text_lower:
            matched_keywords.append(p)
    matched_keywords = list(set(matched_keywords))
    
    if matched_keywords:
        feature_evidence.append({
            "signal": "keyword",
            "value": ", ".join(matched_keywords[:3]),
            "weight": f"{s1:.2f}"
        })
    else:
        feature_evidence.append({
            "signal": "keyword",
            "value": "None detected",
            "weight": "0.00"
        })
        
    interpretation = "Expected resolution duration is high, indicating significant complexity."
    if pred_duration < 10.0:
        interpretation = "Expected resolution duration is low, indicating a standard request."
        
    feature_evidence.append({
        "signal": "resolution_time",
        "value": f"{pred_duration:.1f} hours (predicted)",
        "interpretation": interpretation
    })
    
    delta_val = inf_val - prio_val
    severity_delta_str = f"{delta_val:+d}"
    
    if mismatch_type == "Hidden Crisis":
        explanation = f"The ticket was assigned '{prio_name}' priority, but the description indicates a critical issue regarding '{subj}' for product '{prod}'. The presence of urgent indicators like '{', '.join(matched_keywords[:2])}' and a predicted resolution time of {pred_duration:.1f} hours support a higher inferred severity of '{inf_name}', indicating a Hidden Crisis."
    else:
        explanation = f"The ticket was assigned '{prio_name}' priority, but the description indicates a standard or minor issue regarding '{subj}' for product '{prod}'. No high-urgency keywords were active, and the predicted resolution time is {pred_duration:.1f} hours, suggesting a lower inferred severity of '{inf_name}' and representing a False Alarm."
        
    dossier = {
        "ticket_id": str(row_dict.get('Ticket ID', 'N/A')),
        "assigned_priority": prio_name,
        "inferred_severity": inf_name,
        "mismatch_type": mismatch_type,
        "severity_delta": severity_delta_str,
        "feature_evidence": feature_evidence,
        "constraint_analysis": explanation,
        "confidence": f"{confidence:.2f}"
    }
    
    return is_mismatch, mismatch_type, confidence, inf_name, dossier, pred_duration

# UI Elements
st.title("🔍 Support Integrity Auditor (SIA)")
st.write("Detect priority mismatches, surface hidden crises, and flag false alarms using a self-supervised text auditor.")

if tokenizer is None:
    st.error("Error: Models and pipeline metadata are missing! Run `train_pipeline.py` first to train the classifier and save the pipeline metadata.")
    st.stop()

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Audit Mode", ["Single Ticket Auditor", "Batch Auditor & Heatmap", "Dashboard Insights"])

if app_mode == "Single Ticket Auditor":
    st.header("🎟️ Audit a Single Ticket")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ticket Information")
        ticket_id = st.text_input("Ticket ID", value="TK-9875")
        subject = st.text_input("Ticket Subject", value="System outage - cannot access database")
        description = st.text_area("Ticket Description", value="We are facing a major outage on our production database. The application is completely frozen and the team cannot access any customer records. This is critical and preventing all checkout operations. Please help ASAP.")
        
    with col2:
        st.subheader("Ticket Metadata")
        product = st.selectbox("Product Purchased", list(meta['cluster_rank_map'].keys()) if hasattr(meta, 'cluster_rank_map') else ["Microsoft Office", "Dell XPS", "HP Pavilion", "Oracle Database", "Autodesk AutoCAD"])
        # We can map name in the keys or just list them
        # Let's check products list
        products = ["Dell XPS", "HP Pavilion", "Nintendo Switch", "MacBook Pro", "iPhone", "Samsung Galaxy", "Microsoft Office", "Autodesk AutoCAD", "Adobe Photoshop"]
        product = st.selectbox("Select Product", products)
        
        priority = st.selectbox("Assigned Ticket Priority (Human-labeled)", ["Low", "Medium", "High", "Critical"])
        channel = st.selectbox("Intake Channel", ["Email", "Chat", "Phone", "Social media"])
        ticket_type = st.selectbox("Ticket Type", ["Technical issue", "Billing inquiry", "Cancellation request", "Product inquiry", "Refund request"])
        
    if st.button("Audit Ticket", type="primary"):
        row_dict = {
            'Ticket ID': ticket_id,
            'Ticket Subject': subject,
            'Ticket Description': description,
            'Product Purchased': product,
            'Ticket Priority': priority,
            'Ticket Channel': channel,
            'Ticket Type': ticket_type
        }
        
        with st.spinner("Auditing ticket contents and running checks..."):
            is_mismatch, mismatch_type, confidence, inf_name, dossier, pred_duration = audit_single_ticket(row_dict)
            
        st.subheader("Audit Findings")
        
        # Consistent vs Mismatch display
        if not is_mismatch or mismatch_type == "Consistent":
            st.success(f"✅ Consistent Priority: Assigned priority ({priority}) matches inferred severity ({inf_name}) with {confidence*100:.1f}% confidence.")
        else:
            if mismatch_type == "Hidden Crisis":
                st.error(f"🚨 Priority Mismatch Flagged: Hidden Crisis Detected! (Assigned: {priority} ➡️ Inferred: {inf_name})")
            else:
                st.warning(f"⚠️ Priority Mismatch Flagged: False Alarm Detected. (Assigned: {priority} ➡️ Inferred: {inf_name})")
                
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("Assigned Priority", priority)
            with col_d2:
                st.metric("Inferred Severity", inf_name, delta=dossier['severity_delta'])
            with col_d3:
                st.metric("Classification Confidence", f"{confidence*100:.1f}%")
                
            st.subheader("Evidence Dossier")
            st.json(dossier)

elif app_mode == "Batch Auditor & Heatmap":
    st.header("📊 Batch File Auditor & Heatmap")
    st.write("Upload a CSV file containing support tickets to run audit checks and output a dossier list.")
    
    uploaded_file = st.file_uploader("Upload Ticket CSV", type="csv")
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(raw_df)} rows. Preview:")
        st.dataframe(raw_df.head(5))
        
        if st.button("Run Batch Audit", type="primary"):
            progress_bar = st.progress(0)
            predictions = []
            mismatch_types = []
            confidences = []
            inferred_severities = []
            dossiers = []
            deltas = []
            
            total = len(raw_df)
            for idx, row in raw_df.iterrows():
                row_dict = row.to_dict()
                is_mismatch, mismatch_type, confidence, inf_name, dossier, pred_dur = audit_single_ticket(row_dict)
                predictions.append(1 if is_mismatch else 0)
                mismatch_types.append(mismatch_type)
                confidences.append(confidence)
                inferred_severities.append(inf_name)
                
                assigned_val = priority_map.get(row['Ticket Priority'], 1)
                inf_val = priority_map.get(inf_name, 1)
                deltas.append(inf_val - assigned_val)
                
                if is_mismatch:
                    dossiers.append(dossier)
                progress_bar.progress((idx + 1) / total)
                
            raw_df['Inferred_Severity'] = inferred_severities
            raw_df['Mismatch_Pred'] = predictions
            raw_df['Mismatch_Type'] = mismatch_types
            raw_df['Confidence'] = confidences
            raw_df['Severity_Delta'] = deltas
            
            st.subheader("Batch Audit Results")
            st.dataframe(raw_df[['Ticket ID', 'Ticket Priority', 'Inferred_Severity', 'Mismatch_Pred', 'Mismatch_Type', 'Confidence']].head(10))
            
            # Downloads
            csv_data = raw_df.to_csv(index=False)
            st.download_button("Download Predictions CSV", csv_data, file_name="audit_predictions.csv", mime="text/csv")
            
            json_dossiers = json.dumps(dossiers, indent=2)
            st.download_button("Download Grounded Dossiers JSON", json_dossiers, file_name="grounded_dossiers.json", mime="application/json")
            
            # Severity Delta Heatmap
            st.subheader("🔥 Severity Delta Heatmap")
            st.write("Shows average severity delta (Inferred - Assigned) across channels and ticket categories.")
            
            heatmap_data = raw_df.groupby(['Ticket Type', 'Ticket Channel'])['Severity_Delta'].mean().unstack().fillna(0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
            ax.set_title("Average Severity Delta by Type and Channel")
            plt.tight_layout()
            st.pyplot(fig)

elif app_mode == "Dashboard Insights":
    st.header("📈 Priority Mismatch Dashboard")
    st.write("Insights and distribution of flagged tickets and mismatch signals.")
    
    # Load processed data to show distribution
    if os.path.exists("eval_processed.csv"):
        df = pd.read_csv("eval_processed.csv")
        # Run predictions or use the mismatch column
        st.write("Displaying insights from the evaluation split dataset.")
        
        # Calculate mismatch metrics
        # Consistent vs Mismatched
        # Let's count them
        mismatched_count = df['Mismatch'].sum()
        total_count = len(df)
        consistent_count = total_count - mismatched_count
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Audited Tickets", total_count)
        with col_m2:
            st.metric("Consistent Priority", consistent_count, delta=f"{consistent_count/total_count*100:.1f}%")
        with col_m3:
            st.metric("Mismatches Flagged", mismatched_count, delta=f"{mismatched_count/total_count*100:.1f}%", delta_color="inverse")
            
        # Draw Mismatch distribution pie chart
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.pie([consistent_count, mismatched_count], labels=["Consistent", "Mismatched"], autopct='%1.1f%%', colors=["#4CAF50", "#FFC107"], startangle=90)
        ax1.axis('equal')
        ax1.set_title("Audit Conclusion Distribution")
        
        # Mismatch type distribution
        # In a real batch we can calculate from the priority comparisons
        # Let's mock a simple distribution from the priority and mismatch
        hidden_crisis = 0
        false_alarm = 0
        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        for idx, row in df.iterrows():
            if row['Mismatch'] == 1:
                prio_val = priority_map.get(row['Ticket Priority'], 1)
                inf_val = row['Inferred_Severity']
                if prio_val <= 1 and inf_val >= 2:
                    hidden_crisis += 1
                elif prio_val >= 2 and inf_val <= 1:
                    false_alarm += 1
                    
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=["Hidden Crisis", "False Alarm"], y=[hidden_crisis, false_alarm], palette="viridis", ax=ax2)
        ax2.set_ylabel("Count")
        ax2.set_title("Mismatch Category Distribution")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.pyplot(fig1)
        with col_c2:
            st.pyplot(fig2)
            
        # Top contributing signals or words
        st.subheader("🔑 Top Contributing Escalation Keywords")
        # We can extract the keywords from df['Full Text']
        keywords_freq = {}
        for text in df['Full Text'].astype(str):
            text_lower = text.lower()
            for w in meta['high_urgency_words'] + meta['escalation_phrases']:
                if w in text_lower:
                    keywords_freq[w] = keywords_freq.get(w, 0) + 1
                    
        sorted_kws = sorted(keywords_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=[x[0] for x in sorted_kws], y=[x[1] for x in sorted_kws], palette="magma", ax=ax3)
        plt.xticks(rotation=45)
        ax3.set_ylabel("Frequency")
        ax3.set_title("Top 10 Detected Urgency Keywords/Phrases")
        plt.tight_layout()
        st.pyplot(fig3)
        
    else:
        st.info("No audit logs found. Go to the Batch File Auditor tab to upload a CSV and generate predictions first.")
