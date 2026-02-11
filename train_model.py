import pandas as pd
import os
import joblib
import re
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
MASTER_FILE = "data/training_master.xlsx"
MODEL_DIR = "models"

# ============================================================
# HELPER (EXACT COPY FROM ENGINE - CRITICAL)
# ============================================================

def normalize(text):
    """
    EXACT SAME normalization as Tiksom Engine.
    Required for 100% Memory Match.
    """
    if pd.isna(text): return ""
    text = str(text).lower()
    # 1. Remove legal terms
    text = re.sub(r'\b(inc|ltd|llc|corp|company|co|limited|gmbh|plc|pvt|private)\b', '', text)
    # 2. Remove Special Characters
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    # 3. Fix Spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def retrain_model():
    print("\n[START] Starting Auto-Training Process...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("[WARNING] Data folder created. Please add files.")
        return False, 0

    # 1. READ ALL FILES
    all_files = glob.glob(f"{DATA_DIR}/*.xlsx") + glob.glob(f"{DATA_DIR}/*.csv")
    
    if not all_files:
        print("[WARNING] No data files found to train.")
        return False, 0

    df_list = []
    
    for f in all_files:
        try:
            if f.endswith('.csv'):
                temp_df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
            else:
                temp_df = pd.read_excel(f)
            
            # Normalize columns
            temp_df.columns = [str(c).strip().lower() for c in temp_df.columns]
            
            col_text = next((c for c in temp_df.columns if 'text' in c or 'company' in c), None)
            col_label = next((c for c in temp_df.columns if 'industry' in c or 'label' in c), None)
            
            if col_text and col_label:
                temp_df = temp_df[[col_text, col_label]].rename(columns={col_text: 'text', col_label: 'industry'})
                df_list.append(temp_df)
        except Exception as e:
            print(f"[ERROR] Error reading {f}: {e}")

    if not df_list:
        return False, 0

    # 2. COMBINE & CLEAN
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['industry'] = full_df['industry'].astype(str).str.strip()
    full_df = full_df[full_df['industry'].str.lower() != 'nan']
    
    # CRITICAL: Apply Universal Normalization
    full_df['text_clean'] = full_df['text'].apply(normalize)
    
    # Remove junk
    full_df = full_df[full_df['text_clean'].str.len() > 3] 
    
    # 3. DEDUPLICATE
    full_df = full_df.drop_duplicates(subset=['text_clean'])
    after_len = len(full_df)
    
    print(f"[INFO] Total Memorized Patterns: {after_len} rows.")

    # 4. SAVE MASTER (For Memory Bank)
    try:
        full_df[['text', 'industry']].to_excel(MASTER_FILE, index=False)
        
        # Cleanup temp files
        for f in all_files:
            if os.path.abspath(f) != os.path.abspath(MASTER_FILE):
                try: os.remove(f)
                except: pass
    except Exception as e:
        print(f"[WARNING] Save failed: {e}")

    # 5. TRAIN MODEL
    if after_len < 5:
        return False, after_len

    print("[INFO] Training Brain...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X_vec = tfidf.fit_transform(full_df['text_clean'])
    y = full_df['industry'] 
    
    clf = LogisticRegression(n_jobs=-1, max_iter=2000)
    clf.fit(X_vec, y)
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(clf, f"{MODEL_DIR}/industry_ml_model.joblib")
    joblib.dump(tfidf, f"{MODEL_DIR}/industry_tfidf.joblib")
    
    print("[SUCCESS] Brain Updated!")
    return True, after_len

if __name__ == "__main__":
    retrain_model()