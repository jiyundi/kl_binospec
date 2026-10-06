import joblib
with open('008b_vary_thetaint_slitLPA_major/thetaint_slitLPA_0.0.pkl', "rb") as f:
    data_info = joblib.load(f)
print(data_info['galaxy']['beta'])