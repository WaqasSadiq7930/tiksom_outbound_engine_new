import pandas as pd
import numpy as np
import re
import time
import streamlit as st
from groq import Groq
from industry_keywords import INDUSTRY_MAP, STRONG_SIGNALS

# ============================================================
# 1. UNIVERSAL NORMALIZER (The Fix 🔧)
# ============================================================
def normalize(text):
    """
    Ye function Text ko 'Haddi' tak clean kar deta hai.
    Dono Engine aur Trainer ab yehi use karenge.
    'Tiksom, Inc. (Lahore)' -> 'tiksom lahore'
    """
    if pd.isna(text): return ""
    text = str(text).lower()
    # 1. Remove standard legal terms (Inc, Ltd, LLC etc)
    text = re.sub(r'\b(inc|ltd|llc|corp|company|co|limited|gmbh|plc|pvt|private)\b', '', text)
    # 2. Remove Special Characters (Sirf a-z aur 0-9 bachega)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    # 3. Fix Spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 2. 2026 CONTENT DB (THE SOUL: PAIN + SOL + VALUE) 💎
# ============================================================
CONTENT_DB = {
    "IT_Software": {
        "1-10": {
            "pain": "developer burnout & backlog",
            "sol": "AI-Augmented Remote Squads",
            "val": "ship features 3x faster",
            "subject": "Dev capacity at {Company}",
            "intro": "Are your developers stuck on maintenance instead of shipping new features?"
        },
        "11-50": {
            "pain": "scaling technical debt",
            "sol": "Autonomous DevOps Pipelines",
            "val": "cut deployment costs by 40%",
            "subject": "Scaling bottlenecks...",
            "intro": "As you scale to 50+, technical debt starts killing your momentum."
        }
    },
    "Healthcare": {
        "1-10": {
            "pain": "patient admin overload",
            "sol": "AI Voice Agents",
            "val": "automate 90% of scheduling",
            "subject": "Front desk chaos?",
            "intro": "Small clinics lose revenue when calls go unanswered or patients no-show."
        },
        "11-50": {
            "pain": "data silos & interoperability",
            "sol": "FHIR-Compliant Data Layers",
            "val": "unify patient data instantly",
            "subject": "Data silos at {Company}",
            "intro": "With multiple departments, patient data gets stuck in disconnected systems."
        }
    },
    "Fintech": {
        "1-10": {
            "pain": "compliance bottlenecks",
            "sol": "Automated KYC/AML Engines",
            "val": "onboard users in 3 seconds",
            "subject": "KYC slowing growth?",
            "intro": "Startups lose 40% of users during slow sign-ups due to manual checks."
        },
        "11-50": {
            "pain": "legacy infrastructure limits",
            "sol": "Modular Banking APIs",
            "val": "launch new products in weeks",
            "subject": "Infra scaling...",
            "intro": "Growing fintechs get stuck on legacy infra that breaks when you scale."
        }
    },
    "Real_Estate": {
        "1-10": {
            "pain": "missed tenant inquiries",
            "sol": "24/7 AI Leasing Agents",
            "val": "never miss a commission",
            "subject": "Leads slipping away?",
            "intro": "If a tenant calls at 2 AM, do you answer? Missed calls = Missed revenue."
        },
        "11-50": {
            "pain": "property maintenance chaos",
            "sol": "IoT-Predictive Maintenance",
            "val": "reduce repair costs drastically",
            "subject": "Maintenance costs...",
            "intro": "Managing 50+ units is chaotic. You are reacting to breaks instead of preventing them."
        }
    },
    "E-commerce": {
        "1-10": {
            "pain": "manual order processing",
            "sol": "Automated Fulfillment Bots",
            "val": "process orders 5x faster",
            "subject": "Fulfillment lag?",
            "intro": "Still printing labels manually? That kills your margins and speed."
        },
        "11-50": {
            "pain": "inventory sync issues",
            "sol": "Multi-Channel AI Sync",
            "val": "eliminate stockouts completely",
            "subject": "Stockouts at {Company}",
            "intro": "Selling on Amazon & Shopify? Overselling destroys brand reputation."
        }
    },
    "Logistics": {
        "1-10": {
            "pain": "rising fuel costs",
            "sol": "AI Route Optimization",
            "val": "cut fuel costs by 30%",
            "subject": "Fuel costs...",
            "intro": "Fuel prices are eating your margins. Are your drivers taking the best routes?"
        },
        "11-50": {
            "pain": "supply chain visibility",
            "sol": "Real-Time Fleet Tracking",
            "val": "predict delays instantly",
            "subject": "Delivery delays...",
            "intro": "Managing 50+ trucks without real-time data is a nightmare."
        }
    },
    "Education": {
        "1-10": {
            "pain": "student enrollment drop",
            "sol": "AI Admission Chatbots",
            "val": "boost admissions by 40%",
            "subject": "Enrollment numbers...",
            "intro": "Students expect instant replies. If you wait 24 hours, they go elsewhere."
        },
        "11-50": {
            "pain": "admin paperwork load",
            "sol": "Automated Student Records",
            "val": "save 1000+ admin hours",
            "subject": "Admin overload...",
            "intro": "Your staff is drowning in paperwork instead of focusing on education."
        }
    },
    "Manufacturing": {
        "1-10": {
            "pain": "unexpected downtime",
            "sol": "Predictive Maintenance Sensors",
            "val": "zero unplanned downtime",
            "subject": "Production stops?",
            "intro": "One broken machine can kill a whole week's production."
        },
        "11-50": {
            "pain": "supply chain blindspots",
            "sol": "End-to-End Inventory AI",
            "val": "optimize raw material flow",
            "subject": "Supply chain gaps...",
            "intro": "Do you know exactly when your raw materials will run out?"
        }
    },
    "Media": {
        "1-10": {
            "pain": "content scale limits",
            "sol": "Generative AI Tools",
            "val": "produce 10x more content",
            "subject": "Content bottleneck?",
            "intro": "Need to produce more content but can't hire more editors?"
        },
        "11-50": {
            "pain": "ad targeting waste",
            "sol": "AI Audience Segmentation",
            "val": "double ad ROAS",
            "subject": "Ad spend waste...",
            "intro": "You are wasting 50% of your ad budget on the wrong audience."
        }
    },
    "General": { 
        "1-10": {
            "pain": "repetitive manual work",
            "sol": "GenAI Process Automation",
            "val": "reclaim 20 hours/week",
            "subject": "Efficiency at {Company}",
            "intro": "Small teams shouldn't be bogged down by manual data entry."
        },
        "11-50": {
            "pain": "communication silos",
            "sol": "Unified Intelligence Dashboards",
            "val": "make real-time decisions",
            "subject": "Management silos...",
            "intro": "With 50+ people, it's hard to know what's happening across all teams."
        }
    }
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_industry_name(name):
    if not name: return "General"
    return name.strip().replace(" ", "_")

def clean_company_name(name):
    if pd.isna(name): return ""
    name = str(name)
    name = re.sub(r'\b(inc|ltd|llc|corp|company|co|limited|gmbh|plc|pvt|private)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip()

def bucket_headcount(val):
    if pd.isna(val) or str(val).strip() == "": 
        return "1-10", True 
    try:
        val_str = str(val).lower().replace(",", "")
        nums = re.findall(r'\d+', val_str)
        if not nums: return "1-10", True
        
        if len(nums) > 1:
            avg = (int(nums[0]) + int(nums[1])) / 2
            val_int = int(avg)
        else:
            val_int = int(nums[0])

        if val_int <= 10: return "1-10", False
        else: return "11-50", False
    except:
        return "1-10", True

# ============================================================
# 4. CAMPAIGN GENERATOR (THE FORMULA IS BACK) 🧪
# ============================================================

def generate_campaign_text(industry, headcount):
    lookup_ind = normalize_industry_name(industry)
    lookup_ind = lookup_ind if lookup_ind in CONTENT_DB else "General"
    
    bucket = headcount if headcount in ["1-10", "11-50"] else "11-50"
    
    # Get the Magic Components
    industry_data = CONTENT_DB.get(lookup_ind, CONTENT_DB["General"])
    c = industry_data.get(bucket, industry_data["1-10"]) 
    
    # Construct Emails using Pain + Sol + Value
    return f"""=== CAMPAIGN STRATEGY: {industry} | {bucket} Employees ===
PAIN POINT: {c['pain']}
SOLUTION: {c['sol']}
VALUE PROP: {c['val']}

-------------------------------------------------------
[EMAIL 1 - INITIAL OUTREACH]
Subject: {c['subject']}

Hi {{First Name}},

{c['intro']}

We use **{c['sol']}** to help companies {c['val']}.

See examples: https://www.tiksom.co.uk/portfolio
Are you open to a 10-min chat?

Best,
Waqas
-------------------------------------------------------

[EMAIL 2 - FOLLOW UP (3 Days Later)]
Subject: Re: {c['subject']}

Hi {{First Name}},

I mentioned {c['pain']} earlier because I see many teams struggling with it.
Our system can help you {c['val']} in just a few weeks.

Thoughts?
Waqas
-------------------------------------------------------

[EMAIL 3 - VALUE ADD (7 Days Later)]
Subject: Quick question

Hi {{First Name}},

Just one 15-min call could show you how to eliminate {c['pain']} forever.

Let me know if this is relevant.
Waqas
-------------------------------------------------------

[EMAIL 4 - BREAK UP (14 Days Later)]
Subject: Permission to close file?

Hi {{First Name}},

Permission to close this file? Best of luck!

Waqas
-------------------------------------------------------
"""

# ============================================================
# 5. CLASSIFICATION LOGIC (MEMORY + BROAD MATCH)
# ============================================================

def ai_classify(client, text):
    if not client: return "General"
    
    prompt = f"""Task: Classify this company description into ONE category: 
    Options: {', '.join(INDUSTRY_MAP.keys())}. 
    
    Rules:
    1. Do NOT return 'General' unless absolutely no info is present.
    2. If it sells physical goods -> 'E-commerce'.
    3. If it builds websites/apps/tech -> 'IT_Software'.
    4. If it transports goods -> 'Logistics'.
    5. Return ONLY the category name. No extra text.
    
    Text: {text[:400]}"""

    try:
        if 'api_status' in st.session_state:
            st.session_state['api_status'] = "🟢 AI Working"
            
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20
        )
        return normalize_industry_name(resp.choices[0].message.content.strip())
    except:
        if 'api_status' in st.session_state:
            st.session_state['api_status'] = "🔴 Skip (Local Fallback)"
        return "General"

def classify_row(text, clf, tfidf, client, memory_dict):
    clean_txt = normalize(text) # <--- UNIVERSAL CLEANER (100% Match)
    
    # STEP 0: EXACT MEMORY (RATTA SYSTEM) 🧠
    if clean_txt in memory_dict:
        return memory_dict[clean_txt], "Memory_Cache", 1.0

    # STEP 1: STRONG SIGNALS
    for ind, signals in STRONG_SIGNALS.items():
        for s in signals:
            if re.search(r"\b" + re.escape(s) + r"\b", clean_txt):
                return normalize_industry_name(ind), "Strong_Signal", 1.0

    # STEP 2: ML MODEL
    if clf and tfidf:
        try:
            X = tfidf.transform([clean_txt])
            probs = clf.predict_proba(X)[0]
            best_idx = np.argmax(probs)
            if probs[best_idx] >= 0.15: 
                return normalize_industry_name(clf.classes_[best_idx]), "ML_Relaxed", probs[best_idx]
        except: pass

    # STEP 3: EXACT KEYWORDS
    scores = {ind: 0 for ind in INDUSTRY_MAP}
    for ind, kws in INDUSTRY_MAP.items():
        for k in kws:
            if re.search(r"\b" + re.escape(k) + r"\b", clean_txt): scores[ind] += 1
            
    best_kw = max(scores, key=scores.get)
    if scores[best_kw] > 0: 
        return normalize_industry_name(best_kw), "Keyword", 0.6

    # STEP 4: BROAD MATCH
    for ind, kws in INDUSTRY_MAP.items():
        for k in kws:
            if k in clean_txt: 
                return normalize_industry_name(ind), "Broad_Match", 0.4

    # STEP 5: API
    if client:
        return ai_classify(client, clean_txt), "AI_Forced", 1.0
        
    return "General", "General_Fallback", 0.0

# ============================================================
# 6. PIPELINE
# ============================================================

def classify_dataframe(df, clf, tfidf, client, text_col, headcount_col, company_col, memory_dict):
    df = df.copy()
    
    # Pre-clean company names for better readability in output
    df['clean_company'] = df[company_col].apply(clean_company_name)
    
    try:
        headcount_results = df[headcount_col].apply(bucket_headcount)
        df['headcount_bucket'] = [x[0] for x in headcount_results]
        df['is_estimated_headcount'] = [x[1] for x in headcount_results]
    except:
        df['headcount_bucket'] = "1-10"

    results = []
    sources = []
    confs = []
    
    for row in df.itertuples():
        c_text = str(getattr(row, text_col, ""))
        c_comp = str(getattr(row, company_col, ""))
        
        # KEY POINT: Ye wo text hai jo MEMORY KEY banega
        full_text = f"{c_comp} {c_text}"
        
        ind, src, conf = classify_row(full_text, clf, tfidf, client, memory_dict)
        
        results.append(ind)
        sources.append(src)
        confs.append(conf)
        
    df['final_industry'] = results
    df['source'] = sources
    df['confidence'] = confs
    return df