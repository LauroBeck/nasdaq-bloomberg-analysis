import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# 1. SYNTHETIC DATA GENERATION
# -----------------------------

np.random.seed(42)
n = 200

utilities = pd.DataFrame({
    "utility": [f"Utility_{i}" for i in range(n)],
    "region": np.random.choice(
        ["Brazil", "US", "Europe", "Asia"], size=n
    ),

    # Operational / market signals
    "outage_index": np.random.beta(2, 5, size=n),
    "der_growth": np.random.beta(2, 4, size=n),
    "capex_growth": np.random.beta(2, 3, size=n),

    # Digital behavior indicators
    "digital_investment": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
    "data_modernization": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
})

# -----------------------------
# 2. TARGET SIMULATION LOGIC
# -----------------------------
# Simulate realistic adoption behavior

pressure = (
    utilities["outage_index"] * 0.35 +
    utilities["der_growth"] * 0.25 +
    utilities["capex_growth"] * 0.20 +
    utilities["digital_investment"] * 0.10 +
    utilities["data_modernization"] * 0.10
)

# Higher pressure → higher probability of cloud expansion
probability = pressure + np.random.normal(0, 0.05, size=n)

utilities["cloud_expansion"] = (probability > 0.55).astype(int)

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

utilities["modernization_pressure"] = pressure

features = [
    "outage_index",
    "der_growth",
    "capex_growth",
    "digital_investment",
    "data_modernization",
    "modernization_pressure"
]

X = utilities[features]
y = utilities["cloud_expansion"]

# -----------------------------
# 4. TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 6. MODEL EVALUATION
# -----------------------------

predictions = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, predictions))

# -----------------------------
# 7. PROBABILITY SCORING
# -----------------------------

utilities["cloud_expansion_probability"] = model.predict_proba(X)[:, 1]

# -----------------------------
# 8. RANKING UTILITIES
# -----------------------------

ranked = utilities.sort_values(
    by="cloud_expansion_probability",
    ascending=False
)

print("\nTop Cloud / Digital Expansion Candidates:\n")
print(
    ranked[[
        "utility",
        "region",
        "cloud_expansion_probability",
        "modernization_pressure"
    ]].head(10)
)

# -----------------------------
# 9. BUSINESS INTERPRETATION
# -----------------------------

high_value = ranked.head(10)

print("\nInterpretation:\n")
print("Utilities most likely experiencing:")
print("- Grid stress or modernization pressure")
print("- DER / renewables integration growth")
print("- Active digital / data initiatives")
print("- Elevated probability of cloud expansion")
