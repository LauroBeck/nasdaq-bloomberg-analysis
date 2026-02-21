import pandas as pd
import numpy as np

# ------------------------------------
# 1. DATA INSPIRED BY CNN BRASIL SCREEN
# ------------------------------------

data = [
    {"utility": "Energisa",      "price": 52.17, "change_pct":  1.62},
    {"utility": "Eneva",         "price": 21.69, "change_pct":  1.88},
    {"utility": "Engie Brasil",  "price": 33.55, "change_pct":  0.84},
    {"utility": "Equatorial",    "price": 40.94, "change_pct":  0.22},
    {"utility": "CPFL Energia",  "price": 49.44, "change_pct": -0.26},
]

df = pd.DataFrame(data)

print("\nMarket Snapshot:\n")
print(df)

# ------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------
# Translate market behavior into signals

# Momentum proxy (very common in finance)
df["momentum_score"] = df["change_pct"] / 100

# Simulated operational / modernization signals
np.random.seed(10)

df["outage_index"] = np.random.uniform(0.2, 0.8, len(df))
df["der_growth"] = np.random.uniform(0.1, 0.7, len(df))
df["capex_growth"] = np.random.uniform(0.2, 0.9, len(df))

# Binary digital indicators (typical enterprise modeling)
df["digital_investment"] = np.random.choice([0, 1], len(df))
df["data_modernization"] = np.random.choice([0, 1], len(df))

# ------------------------------------
# 3. MODERNIZATION PRESSURE MODEL
# ------------------------------------

df["modernization_pressure"] = (
    df["outage_index"] * 0.35 +
    df["der_growth"] * 0.25 +
    df["capex_growth"] * 0.20 +
    df["digital_investment"] * 0.10 +
    df["data_modernization"] * 0.10
)

# Cloud expansion probability proxy
df["cloud_expansion_probability"] = (
    df["modernization_pressure"] * 0.7 +
    df["momentum_score"] * 0.3
)

# Clip to probability bounds
df["cloud_expansion_probability"] = df[
    "cloud_expansion_probability"
].clip(0, 1)

# ------------------------------------
# 4. RANK UTILITIES
# ------------------------------------

ranked = df.sort_values(
    by="cloud_expansion_probability",
    ascending=False
)

print("\nRanked Cloud / Digital Expansion Signals:\n")
print(
    ranked[[
        "utility",
        "price",
        "change_pct",
        "modernization_pressure",
        "cloud_expansion_probability"
    ]]
)

# ------------------------------------
# 5. INTERPRETATION LAYER
# ------------------------------------

print("\nInterpretation:\n")

for _, row in ranked.iterrows():
    print(f"{row['utility']} → Probability {row['cloud_expansion_probability']:.2f}")

print("\nHigher scores typically indicate:")
print("- Grid stress / reliability pressure")
print("- DER / renewables integration growth")
print("- Active digital & data initiatives")
print("- Elevated likelihood of cloud / analytics demand")
