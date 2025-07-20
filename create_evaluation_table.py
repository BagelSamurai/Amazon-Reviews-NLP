import pickle
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import os

def is_relevant(review_text, rating, aspect1, aspect2, opinion, review_title):
    """Determine if a review is relevant based on multiple criteria"""
    # Convert everything to lowercase
    review_text = review_text.lower()
    aspect1 = aspect1.lower()
    aspect2 = aspect2.lower()
    opinion = opinion.lower()
    
    # Split into sentences
    sentences = sent_tokenize(review_text)
    
    # Check if any sentence contains both aspect and opinion
    has_aspect_opinion = False
    for sentence in sentences:
        has_aspect = aspect1 in sentence and aspect2 in sentence
        has_opinion = opinion in sentence
        if has_aspect and has_opinion:
            has_aspect_opinion = True
            break
    
    if not has_aspect_opinion:
        return False
        
    # Define positive and negative opinions
    positive_opinions = {'strong', 'useful', 'sharp'}
    negative_opinions = {'poor', 'problem'}
    
    # Check if opinion matches rating
    is_positive_opinion = opinion in positive_opinions
    is_negative_opinion = opinion in negative_opinions
    
    if is_positive_opinion:
        return rating > 3 and has_aspect_opinion
    elif is_negative_opinion:
        return rating <= 3 and has_aspect_opinion
    
    return has_aspect_opinion

def evaluate_method(review_df, aspect1, aspect2, opinion, method):
    """Evaluate a single method's results"""
    filename = f"output/output/{aspect1}_{aspect2}_{opinion}_{method}.pkl"
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        # Number of retrieved documents
        num_retrieved = len(results)
        if num_retrieved == 0:
            return 0, 0, 0
            
        # Count relevant documents
        num_relevant = 0
        for review_id in results:
            if isinstance(review_id, str):
                if not review_id.startswith("'"):
                    review_id = f"'{review_id}'"
            matching_reviews = review_df[review_df['review_id'] == review_id]
            if not matching_reviews.empty:
                review = matching_reviews.iloc[0]
                if is_relevant(str(review['review_text']), 
                             int(review['customer_review_rating']),
                             aspect1, aspect2, opinion,
                             str(review['review_title'])):
                    num_relevant += 1
        
        # Calculate precision
        precision = num_relevant / num_retrieved if num_retrieved > 0 else 0
        
        return num_retrieved, num_relevant, round(precision, 3)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0, 0, 0

def create_evaluation_table():
    """Create the evaluation table with all metrics"""
    # Load review data
    review_df = pd.read_pickle("reviews_segment.pkl")
    
    # Define queries
    queries = [
        ('audio', 'quality', 'poor'),
        ('wifi', 'signal', 'strong'),
        ('mouse button', 'click', 'problem'),
        ('gps', 'map', 'useful'),
        ('image', 'quality', 'sharp')
    ]
    
    # Print header
    print("\nQuery | Baseline (Boolean) | Method 1 (M1) | Method 2 (M2)")
    print("      | #Ret. #Rel. Prec. | #Ret. #Rel. Prec. | #Ret. #Rel. Prec.")
    print("-" * 70)
    
    # Process each query
    for aspect1, aspect2, opinion in queries:
        query = f"{aspect1} {aspect2}:{opinion}"
        
        # Evaluate each method
        baseline = evaluate_method(review_df, aspect1, aspect2, opinion, "method3")
        m1 = evaluate_method(review_df, aspect1, aspect2, opinion, "method4")
        m2 = evaluate_method(review_df, aspect1, aspect2, opinion, "method5")
        
        # Print results
        print(f"{query:<20} | {baseline[0]:>5} {baseline[1]:>5} {baseline[2]:>5.3f} | "
              f"{m1[0]:>5} {m1[1]:>5} {m1[2]:>5.3f} | "
              f"{m2[0]:>5} {m2[1]:>5} {m2[2]:>5.3f}")

if __name__ == "__main__":
    # Download NLTK data if needed
    nltk.download('punkt', quiet=True)
    create_evaluation_table()