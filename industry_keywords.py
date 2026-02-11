# industry_keywords.py

INDUSTRY_MAP = {
    "IT_Software": ["software", "saas", "cloud", "ai", "app", "tech", "data", "cyber", "platform", "devops", "system integrator", "network", "analytics", "api", "mobile", "web", "development", "blockchain", "llm", "chatbot", "erp", "crm", "it solutions", "digital", "it services", "it consulting", "technology", "information technology"],
    "Healthcare": ["health", "medical", "clinic", "hospital", "pharma", "patient", "care", "biotech", "dental", "nurse", "wellness", "fitness", "surgery", "telemedicine", "therapy", "diagnostics", "labs", "md", "dr"],
    "Fintech": ["bank", "finance", "invest", "crypto", "payment", "wallet", "insurance", "capital", "wealth", "trading", "tax", "audit", "lending", "credit", "defi", "neobank", "equity", "fund"],
    "Real_Estate": ["real estate", "property", "broker", "construction", "realtor", "housing", "architect", "development", "leasing", "rental", "proptech", "homes", "estates", "builders"],
    "E-commerce": ["ecommerce", "retail", "store", "shop", "fashion", "consumer", "marketplace", "shopify", "d2c", "amazon", "dropshipping", "subscription", "luxury", "goods", "mart", "boutique"],
    "Logistics": ["logistics", "transport", "freight", "shipping", "delivery", "supply chain", "fleet", "warehouse", "cargo", "3pl", "courier", "last mile", "trucking", "movers", "express"],
    "Education": ["education", "school", "university", "learning", "training", "student", "edtech", "academy", "college", "lms", "bootcamp", "course", "tutoring", "institute"],
    "Manufacturing": ["manufacturing", "factory", "industrial", "production", "machinery", "engineering", "oil", "gas", "energy", "automation", "cnc", "fabrication", "3d printing", "textile", "steel", "works"],
    "Media": ["media", "marketing", "agency", "advertising", "content", "design", "video", "entertainment", "pr", "digital marketing", "branding", "podcast", "studio", "creative"],
    "Legal": ["legal", "law", "attorney", "firm", "justice", "compliance", "lawyer", "litigation", "contract", "notary", "solicitors", "chambers"],
    "Travel": ["travel", "hotel", "tourism", "booking", "hospitality", "flight", "resort", "cruise", "ota", "vacation", "tours"],
    "Automotive": ["automotive", "car", "vehicle", "dealer", "repair", "motor", "ev", "aerospace", "mechanic", "auto", "parts", "garage"]
}

STRONG_SIGNALS = {
    "IT_Software": ["saas", "cloud", "api", "platform", "llm", "ai", "labs", "it services", "it consulting", "software house"],
    "Healthcare": ["clinic", "hospital", "patient", "pharma", "care"],
    "Real_Estate": ["realtor", "broker", "property", "estates"],
    "Fintech": ["bank", "wallet", "crypto", "payment", "capital"],
    "Automotive": ["motors", "auto", "garage"]
}