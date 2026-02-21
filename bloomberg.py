import pandas as pd
import numpy as np
from xbbg import blp # Requires Bloomberg Terminal open
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------
# BLOOMBERG CONFIGURATION
# ----------------------------------------------------------
# Mapping the "Identifier" column from your image to live Bloomberg tickers
tickers = [
    "NVDA US Equity", "AAPL US Equity", "SNDK US Equity", "MSFT US Equity", 
    "AVGO US Equity", "CIEN US Equity", "ORCL US Equity", "PSTG US Equity", 
    "CRM US Equity", "NET US Equity", "NOW US Equity", "AMD US Equity"
]

start_date = "2021-01-01" # 3-year lookback matching the image title
end_date = "2024-01-01"

# ----------------------------------------------------------
# REAL-TIME DATA DOWNLOAD (DIRECT FROM BLOOMBERG)
# ----------------------------------------------------------
print("Connecting to Bloomberg Terminal...")

# Pulling 'PX_LAST' (Last Price) using Bloomberg's historical data request (BDH)
prices = blp.bdh(
    tickers=tickers, 
    flds="PX_LAST", 
    start_date=start_date, 
    end_date=end_date
)

# Flatten multi-index columns from Bloomberg output
prices.columns = [c[0] for c in prices.columns]
returns = prices.pct_change().dropna()

# ----------------------------------------------------------
# ATTRITION & CTR ANALYSIS (Replicating Bloomberg CTR Column)
# ----------------------------------------------------------
# The image shows weights (Avg % Wgt Port) - you can pull these live for the fund
# field "PORTFOLIO_WEIGHTS" can be used for managed funds via BDS
weights_data = blp.bdp(tickers, "CUR_MKT_CAP") # Example reference data
# For a full fund breakdown like the image, use the Bloomberg Portfolio (PRTU) tool

# [The rest of your optimization logic remains the same, 
# but now uses high-fidelity Bloomberg pricing data]
