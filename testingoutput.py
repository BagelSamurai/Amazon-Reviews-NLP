import pickle

# Load specific result
with open('output/output/audio_quality_poor_method1.pkl', 'rb') as f:
    results = pickle.load(f)
    
# Load reviews
import pandas as pd
df = pd.read_pickle("reviews_segment.pkl")

# Look at some results
for review_id in results[:5]:
    review = df[df['review_id'] == review_id].iloc[0]
    print(f"\nReview ID: {review_id}")
    print(f"Rating: {review['customer_review_rating']}")
    print(f"Text: {review['review_text'][:200]}...")