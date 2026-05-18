import pandas as pd

df = pd.read_csv("../binospec_data_pkl/r_hl_table.txt", 
                 sep=r"\s+", header=None, names=["slit", "rhl"])

# 保留每个 slit 的最后一行
df = df.drop_duplicates(subset="slit", keep="last")

array = df.sort_values(by='slit').to_numpy()
