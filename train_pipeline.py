import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df_path):
    print("Loading raw dataset...")
    df = pd.read_csv(df_path)
    
    # Preprocessing
    def preprocess_desc(row):
        desc = str(row['Ticket Description'])
        prod = str(row['Product Purchased'])
        return desc.replace('{product_purchased}', prod)

    df['Cleaned Description'] = df.apply(preprocess_desc, axis=1)
    df['Cleaned Subject'] = df['Ticket Subject'].fillna("").astype(str)
    df['Full Text'] = df['Cleaned Subject'] + " " + df['Cleaned Description']

    # Date calculations
    df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
    df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')
    df['Resolution Duration'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds().abs() / 3600.0
    return df

# Keywords and phrases for Signal 1
high_urgency_words = [
    "urgent", "immediate", "asap", "broken", "crashed", "outage", "blocked", 
    "prevent", "cannot access", "security", "compromised", "down", "critical", 
    "emergency", "lost data", "loss of data", "not working", "fail", "not charging",
    "not turning on", "no power", "unusable", "freeze", "crash", "disabled", "hacked"
]
escalation_phrases = [
    "contacted multiple times", "unresolved", "frustrated", "refund", "cancel my subscription",
    "legal", "warn", "disappointed", "poor customer service", "unacceptable", "terrible",
    "waiting for days", "stuck", "still not", "help me"
]

def get_nlp_score(text):
    text = text.lower()
    score = 0.0
    for word in high_urgency_words:
        if word in text:
            score += 2.0
    for phrase in escalation_phrases:
        if phrase in text:
            score += 1.5
    negations = ["not", "no", "never", "cannot", "doesn't", "don't", "wasn't"]
    words = text.split()
    for i, w in enumerate(words):
        if w in negations and i + 1 < len(words):
            next_w = words[i+1]
            if next_w in ["working", "charging", "turning", "responding", "connecting", "opening"]:
                score += 1.5
    return score

class TicketDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['InputText'].tolist()
        self.labels = df['Is_Urgent'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    df = preprocess_data("customer_support_tickets.csv")
    
    # 1. Signal 1: NLP
    print("Calculating Signal 1 (Rule-based NLP)...")
    df['Signal_1_Raw'] = df['Full Text'].apply(get_nlp_score)
    s1_min = df['Signal_1_Raw'].min()
    s1_max = df['Signal_1_Raw'].max()
    df['Signal_1'] = (df['Signal_1_Raw'] - s1_min) / (s1_max - s1_min + 1e-9)
    
    # 2. Signal 2: Embeddings
    print("Calculating Signal 2 (Embedding-based Clustering)...")
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_st.encode(df['Full Text'].tolist(), show_progress_bar=True)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(embeddings)
    
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['Priority_Numeric'] = df['Ticket Priority'].map(priority_map)
    
    cluster_priority = df.groupby('Cluster')['Priority_Numeric'].mean()
    cluster_rank = cluster_priority.rank(method='first') - 1
    cluster_rank_map = (cluster_rank / 4.0).to_dict()
    df['Signal_2_Cluster'] = df['Cluster'].map(cluster_rank_map)
    
    ref_high_emb = model_st.encode("emergency critical failure down crash blocked cannot access loss of data broken not working")
    ref_low_emb = model_st.encode("general inquiry information request how to change settings setup guidelines question")
    
    cos_sim_high = np.dot(embeddings, ref_high_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(ref_high_emb) + 1e-9)
    cos_sim_low = np.dot(embeddings, ref_low_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(ref_low_emb) + 1e-9)
    df['Signal_2_Sim'] = (cos_sim_high - cos_sim_low + 1.0) / 2.0
    df['Signal_2'] = 0.5 * df['Signal_2_Cluster'] + 0.5 * df['Signal_2_Sim']
    
    # 3. Signal 3: Resolution Time
    print("Calculating Signal 3 (Resolution Time Regression)...")
    closed_df = df[df['Ticket Status'] == 'Closed'].copy()
    closed_embeddings = embeddings[closed_df.index]
    
    reg = Ridge(alpha=1.0)
    reg.fit(closed_embeddings, closed_df['Resolution Duration'].fillna(8.0))
    
    df['Pred_Resolution_Duration'] = reg.predict(embeddings)
    s3_min = df['Pred_Resolution_Duration'].min()
    s3_max = df['Pred_Resolution_Duration'].max()
    df['Signal_3'] = (df['Pred_Resolution_Duration'] - s3_min) / (s3_max - s3_min + 1e-9)
    
    # Fusion
    print("Fusing signals and generating pseudo-labels...")
    df['Fusion_Score'] = (df['Signal_1'] + df['Signal_2'] + df['Signal_3']) / 3.0
    
    q25 = df['Fusion_Score'].quantile(0.25)
    q50 = df['Fusion_Score'].quantile(0.50)
    q75 = df['Fusion_Score'].quantile(0.75)
    
    def get_inferred_severity(score):
        if score < q25:
            return 0
        elif score < q50:
            return 1
        elif score < q75:
            return 2
        else:
            return 3
            
    df['Inferred_Severity'] = df['Fusion_Score'].apply(get_inferred_severity)
    
    def get_mismatch(row):
        inf = row['Inferred_Severity']
        prio = row['Priority_Numeric']
        if inf >= 2 and prio <= 1:
            return 1
        elif inf <= 1 and prio >= 2:
            return 1
        else:
            return 0
            
    df['Mismatch'] = df.apply(get_mismatch, axis=1)
    
    print("Mismatch value counts:")
    print(df['Mismatch'].value_counts(normalize=True))
    
    # Save pipeline metadata
    print("Saving pipeline metadata...")
    metadata = {
        's1_min': s1_min,
        's1_max': s1_max,
        's3_min': s3_min,
        's3_max': s3_max,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'cluster_rank_map': cluster_rank_map,
        'ref_high_emb': ref_high_emb,
        'ref_low_emb': ref_low_emb,
        'kmeans': kmeans,
        'reg': reg,
        'high_urgency_words': high_urgency_words,
        'escalation_phrases': escalation_phrases
    }
    joblib.dump(metadata, "pipeline_metadata.joblib")
    
    # 4. Training classifier
    print("Preparing classifier training...")
    df['Is_Urgent'] = (df['Inferred_Severity'] >= 2).astype(int)
    df['InputText'] = "Channel: " + df['Ticket Channel'] + " | Product: " + df['Product Purchased'] + " | Subject: " + df['Cleaned Subject'] + " | Description: " + df['Cleaned Description']
    
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_Urgent'])
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = TicketDataset(train_df, tokenizer)
    eval_dataset = TicketDataset(eval_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    
    # Handle class imbalance
    class_counts = train_df['Is_Urgent'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    best_mismatch_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        print(f"Loss: {total_loss/len(train_loader):.4f}")
        
        # Eval
        model.eval()
        all_urgent_preds = []
        all_urgent_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                all_urgent_preds.extend(preds)
                all_urgent_labels.extend(labels.cpu().numpy())
                
        # Calculate Mismatch Metrics using XOR
        eval_prio_high = (eval_df['Priority_Numeric'] >= 2).astype(int).values
        eval_mismatch_true = eval_df['Mismatch'].values
        
        # Mismatch Pred = Urgent Pred ^ Priority High
        eval_mismatch_pred = np.bitwise_xor(all_urgent_preds, eval_prio_high)
        
        acc = accuracy_score(eval_mismatch_true, eval_mismatch_pred)
        f1 = f1_score(eval_mismatch_true, eval_mismatch_pred, average='macro')
        rec_0 = recall_score(eval_mismatch_true, eval_mismatch_pred, pos_label=0)
        rec_1 = recall_score(eval_mismatch_true, eval_mismatch_pred, pos_label=1)
        
        print(f"Eval Mismatch Accuracy: {acc:.4f}")
        print(f"Eval Mismatch Macro F1: {f1:.4f}")
        print(f"Recall Class 0 (Consistent): {rec_0:.4f}")
        print(f"Recall Class 1 (Mismatched): {rec_1:.4f}")
        
        if f1 > best_mismatch_f1:
            best_mismatch_f1 = f1
            print("New best F1! Saving model to ./sia_model...")
            model.save_pretrained("./sia_model")
            tokenizer.save_pretrained("./sia_model")
            
    print("\nTraining complete! Best Mismatch Macro F1 achieved:", best_mismatch_f1)
    
if __name__ == "__main__":
    train()
