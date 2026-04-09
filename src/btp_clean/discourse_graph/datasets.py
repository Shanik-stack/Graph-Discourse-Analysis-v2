import numpy as np
import pandas as pd
from datasets import load_dataset
from btp_clean.paths import DATA_DIR

def kialo_data():
    ds = load_dataset("timchen0618/Kialo")
    print(ds)
    
import pandas as pd

def Ethix_data(path=None):
    if path is None:
        path = DATA_DIR / "discourse_graph" / "Ethix_dataset.csv"

    # Read CSV
    ds = pd.read_csv(path)
    
    # Group by Debate and combine Arguments and Schemes into lists
    grouped_ds = ds.groupby("Debate").agg({
        "Argument": list,
        "Scheme": list
    }).reset_index()
    
    return grouped_ds

if __name__ == "__main__":
    # Example usage
    df_combined = Ethix_data()


    x = Ethix_data()
    print(x.iloc[0]["Debate"])
    print(x.iloc[0]["Argument"], len(x.iloc[0]["Argument"]))

    l  = 0
    I = 0
    for i in range(len(x)):
        m = len(x.iloc[i]["Scheme"])
        
        if(m>l): I = i
        l = max(l,m)
    print(l,I)
