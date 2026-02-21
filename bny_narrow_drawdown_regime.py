"""
BNY Mellon Narrow Drawdown Regime Engine
Institutional Tactical Overlay Model
"""

def classify_regime(
    pct_sp_positive,
    avg_positive_return,
    avg_negative_tech_return,
    tech_weight_negative
):

    print("\nBNY MELLON NARROW DRAWDOWN DIAGNOSTIC")
    print("--------------------------------------------------")

    print(f"% S&P Positive: {pct_sp_positive}%")
    print(f"Avg Positive Return: {avg_positive_return}%")
    print(f"Avg Negative Tech Return: {avg_negative_tech_return}%")
    print(f"Tech Weight Negative: {tech_weight_negative}%")

    # -----------------------------
    # LAYER 1: BREADTH
    # -----------------------------

    if pct_sp_positive > 60:
        breadth_regime = "BROAD RISK-ON"
    elif pct_sp_positive > 40:
        breadth_regime = "MIXED"
    else:
        breadth_regime = "WEAK"

    # -----------------------------
    # LAYER 2: CONCENTRATION STRESS
    # -----------------------------

    concentration_stress = (
        abs(avg_negative_tech_return) > 10
        and tech_weight_negative > 15
    )

    # -----------------------------
    # LAYER 3: DISPERSION
    # -----------------------------

    dispersion = avg_positive_return - abs(avg_negative_tech_return)

    # -----------------------------
    # FINAL REGIME LOGIC
    # -----------------------------

    if breadth_regime == "BROAD RISK-ON" and not concentration_stress:
        regime = "FULL RISK-ON"
        action = "OVERWEIGHT EQUITIES"

    elif breadth_regime == "BROAD RISK-ON" and concentration_stress:
        regime = "NARROW DRAWDOWN"
        action = "ROTATE / REDUCE TECH / ADD DEFENSIVES"

    elif breadth_regime == "MIXED":
        regime = "NEUTRAL"
        action = "FACTOR BALANCE"

    else:
        regime = "RISK-OFF"
        action = "DEFENSIVE / TREASURIES"

    print("\nREGIME CLASSIFICATION")
    print("--------------------------------------------------")
    print("Breadth:", breadth_regime)
    print("Concentration Stress:", concentration_stress)
    print("Dispersion:", round(dispersion, 2))
    print("Final Regime:", regime)
    print("Portfolio Action:", action)

    return regime, action


if __name__ == "__main__":

    # Chart values from BNY graphic
    classify_regime(
        pct_sp_positive=64.2,
        avg_positive_return=11.4,
        avg_negative_tech_return=-16.1,
        tech_weight_negative=20.4
    )
