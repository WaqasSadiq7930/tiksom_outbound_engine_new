import streamlit as st
import pandas as pd
from groq import Groq
import io
import zipfile
import os
import joblib
import time
import re

# Import Logic
from tiksom_industry_engine import classify_dataframe, generate_campaign_text, normalize
import train_model

# ============================================================
# CONFIG & MEMORY LOADER
# ============================================================

st.set_page_config(page_title="Tiksom Auto-Learn", layout="wide")
st.title("🚀 Tiksom: Smart Learning Mode (With Memory 🧠)")

if 'api_status' not in st.session_state: st.session_state['api_status'] = "⚪ Ready"

@st.cache_resource
def load_models_and_memory():
    # 1. Load ML Model
    try:
        clf = joblib.load("models/industry_ml_model.joblib")
        tfidf = joblib.load("models/industry_tfidf.joblib")
    except:
        clf, tfidf = None, None

    # 2. Load Memory (Ratta System)
    memory_dict = {}
    try:
        if os.path.exists("data/training_master.xlsx"):
            master_df = pd.read_excel("data/training_master.xlsx")
            # Create a dictionary: { "clean_text": "Industry" }
            master_df['text_clean'] = master_df['text'].apply(normalize)
            memory_dict = pd.Series(master_df.industry.values, index=master_df.text_clean).to_dict()
    except Exception as e:
        print(f"Memory Load Error: {e}")
    
    return clf, tfidf, memory_dict

clf, tfidf, memory_dict = load_models_and_memory()

# ============================================================
# MAIN APP
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    use_ai = st.checkbox("Enable AI (API)", value=True)
    api_key = st.text_input("Groq API Key", type="password", disabled=not use_ai)
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    
    st.divider()
    
    # Stats Display
    mem_size = len(memory_dict) if memory_dict else 0
    st.info(f"🧠 Memory Bank: {mem_size} Patterns Memorized")
    
    if clf: st.success("✅ Neural Brain Active")
    else: st.warning("⚠️ Brain Empty (Learning...)")
    
    st.markdown("---")
    if st.button("🧠 Force Retrain Brain"):
        with st.spinner("Training Brain on Master Data..."):
            success, count = train_model.retrain_model()
            if success: 
                st.success(f"Trained on {count} rows!")
                st.cache_resource.clear()
            else: 
                st.error("Training Failed (Check Data).")

if uploaded_file and st.button("🚀 Start Processing & Learning"):
    
    # 1. LOAD
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
        
    # COLUMNS (Smart Detection)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Priority: Company > Organization > Name (To avoid picking Person Name)
    col_name = next((c for c in df.columns if 'company' in c), None)
    if not col_name: col_name = next((c for c in df.columns if 'organization' in c), None)
    if not col_name: col_name = next((c for c in df.columns if 'account' in c), None)
    if not col_name: col_name = next((c for c in df.columns if 'name' in c), None) # Last resort
    
    if not col_name: 
        st.error("❌ Could not find 'Company' column. Please rename a column to 'Company'.")
        st.stop()
    
    # Find Description & Industry Input columns
    col_desc = next((c for c in df.columns if 'desc' in c or 'about' in c or 'title' in c), None)
    col_ind_src = next((c for c in df.columns if 'industry' in c or 'sector' in c), None)
    col_head = next((c for c in df.columns if 'headcount' in c or 'employ' in c), 'headcount_def')
    
    if col_head == 'headcount_def': df['headcount_def'] = "1-10"

    # --- RICH TEXT CONSTRUCTION (The Fix) ---
    # Combine Company + Description + Input Industry into one text block
    df['synthetic_text'] = df[col_name].astype(str)
    
    if col_desc:
        df['synthetic_text'] += " " + df[col_desc].astype(str)
    
    if col_ind_src:
        df['synthetic_text'] += " " + df[col_ind_src].astype(str) # <--- CRITICAL FIX
        
    target_text_col = 'synthetic_text'

    # 2. PROCESS
    client = Groq(api_key=api_key) if (use_ai and api_key) else None
    
    st.markdown("### 📊 Live Processing Stats")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_total = st.empty()
    with c2: metric_local = st.empty()
    with c3: metric_ai = st.empty()
    with c4: metric_general = st.empty()
    
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    batch_size = 20
    all_results = []
    total_rows = len(df)
    processed_count = 0
    
    count_local = 0
    count_ai = 0
    count_general = 0
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Pass the RICH text column
        result_batch = classify_dataframe(batch_df, clf, tfidf, client, target_text_col, col_head, col_name, memory_dict)
        all_results.append(result_batch)
        
        # Update Counters
        local_in_batch = result_batch['source'].str.contains('ML|Keyword|Strong|Memory|Broad', case=False, na=False).sum()
        ai_in_batch = result_batch['source'].str.contains('AI', case=False, na=False).sum()
        general_in_batch = len(result_batch[result_batch['final_industry'] == 'General'])
        
        count_local += local_in_batch
        count_ai += ai_in_batch
        count_general += general_in_batch
        processed_count += len(batch_df)
        
        # UPDATE UI
        curr_stat = st.session_state.get('api_status', "⚪")
        
        metric_total.metric("Total Rows", f"{processed_count} / {total_rows}")
        metric_local.metric("⚡ Local (Fast)", int(count_local))
        metric_ai.metric("🤖 AI (Calls)", int(count_ai))
        metric_general.metric("⚠️ General", int(count_general))
        
        status_placeholder.info(f"API Status: {curr_stat}")
        progress_bar.progress(int((processed_count / total_rows) * 100))

    final_df = pd.concat(all_results, ignore_index=True)
    
    # 3. AUTO-LEARNING (MEMORY UPDATE) 🧠
    st.markdown("---")
    st.info("🧠 Memorizing new patterns...")
    
    # Only learn from High Confidence sources (AI or Strong Signal)
    new_knowledge = final_df[
        (final_df['final_industry'] != 'General') &
        (final_df['source'].str.contains('AI|Strong', case=False, na=False))
    ]
    
    if len(new_knowledge) > 0:
        # Save temp data
        train_ready = pd.DataFrame({
            'text': final_df.loc[new_knowledge.index, 'synthetic_text'], # Use Rich Text for training too
            'industry': new_knowledge['final_industry']
        })
        
        if not os.path.exists("data"): os.makedirs("data")
        train_ready.to_csv("data/temp_new_knowledge.csv", index=False)
        
        st.write(f"🤓 Memorized **{len(train_ready)}** new items. Updating Brain...")
        
        # Call Trainer
        success, total_rows_in_brain = train_model.retrain_model()
        
        if success:
            st.success(f"✅ Brain & Memory Updated! Total Knowledge: **{total_rows_in_brain}** rows.")
            st.cache_resource.clear()
        else:
            st.warning("Training skipped.")
    else:
        st.write("💤 No new knowledge found.")

    # 4. DOWNLOAD
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("full_report.csv", final_df.to_csv(index=False).encode('utf-8'))
        try:
            groups = final_df.groupby(['final_industry', 'headcount_bucket'])
            for (ind, hc), group in groups:
                folder_name = ind
                zf.writestr(f"{folder_name}/{hc}_leads.csv", group.to_csv(index=False).encode('utf-8'))
                camp_txt = generate_campaign_text(ind, hc)
                zf.writestr(f"{folder_name}/{hc}_campaign.txt", camp_txt.encode('utf-8'))
        except Exception as e:
            st.error(f"Download generation error: {e}")
            zf.writestr("error_log.txt", str(e))

    st.success("🚀 Done!")
    st.download_button("📦 Download Results", zip_buffer.getvalue(), "Tiksom_Results_Fixed.zip", mime="application/zip")