import pickle
with open("output/output/audio_quality_poor_method1.pkl", "rb") as f:
    loaded_series = pickle.load(f)
print(loaded_series)
