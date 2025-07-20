import pandas as pd
import os
import pickle
import numpy as np
from tabulate import tabulate
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

def load_review_texts(review_ids, review_df):
    """Load review texts and ratings for given IDs"""
    # Clean review IDs in DataFrame
    review_df['review_id'] = review_df['review_id'].str.strip("'").str.strip('"')
    
    # Get reviews that match the IDs
    matching_reviews = review_df[review_df['review_id'].isin(review_ids)]
    return matching_reviews

def evaluate_results(aspect1, aspect2, opinion, review_df):
    """Evaluate results using alternative metrics"""
    results = {}
    base_filename = f"{aspect1}_{aspect2}_{opinion.replace(' ', '_')}"
    
    # Combine search terms for relevance checking
    search_terms = f"{aspect1} {aspect2} {opinion}".lower().split()
    
    for method in range(1, 7):
        filename = f"{base_filename}_method{method}.pkl"
        filepath = os.path.join('output', 'output', filename)
        
        try:
            # Load results
            with open(filepath, 'rb') as f:
                results_series = pickle.load(f)
                
            # Get actual reviews
            matching_reviews = load_review_texts(results_series, review_df)
            
            if len(matching_reviews) > 0:
                # 1. Calculate term frequency in results
                term_presence = []
                for text in matching_reviews['review_text']:
                    text = str(text).lower()
                    presence = sum(1 for term in search_terms if term in text)
                    term_presence.append(presence / len(search_terms))
                
                avg_term_presence = np.mean(term_presence) if term_presence else 0
                
                # 2. Calculate rating consistency
                ratings = matching_reviews['customer_review_rating'].astype(float)
                rating_std = np.std(ratings) if len(ratings) > 1 else 0
                rating_consistency = 1 / (1 + rating_std)  # Normalize to 0-1
                
                # 3. Calculate cohesion score if we have enough documents
                if len(matching_reviews) > 2:
                    # Create TF-IDF vectors
                    vectorizer = TfidfVectorizer(max_features=1000)
                    tfidf_matrix = vectorizer.fit_transform(
                        matching_reviews['review_text'].astype(str)
                    )
                    try:
                        cohesion_score = silhouette_score(
                            tfidf_matrix.toarray(),
                            np.ones(len(matching_reviews))  # All docs in one cluster
                        )
                        # Convert from [-1,1] to [0,1]
                        cohesion_score = (cohesion_score + 1) / 2
                    except:
                        cohesion_score = 0
                else:
                    cohesion_score = 0
                
                # Combine metrics into final score
                final_score = (avg_term_presence + rating_consistency + cohesion_score) / 3
                
            else:
                avg_term_presence = 0
                rating_consistency = 0
                cohesion_score = 0
                final_score = 0
            
            results[f'method{method}'] = {
                '#Ret': len(matching_reviews),
                'TermScore': round(avg_term_presence, 3),
                'RatingCons': round(rating_consistency, 3),
                'Cohesion': round(cohesion_score, 3),
                'FinalScore': round(final_score, 3)
            }
            
        except FileNotFoundError:
            results[f'method{method}'] = {
                '#Ret': 0,
                'TermScore': 0.000,
                'RatingCons': 0.000,
                'Cohesion': 0.000,
                'FinalScore': 0.000
            }
    
    return results

def create_results_table(review_df):
    """Create and display results table for all queries"""
    queries = [
        ('audio', 'quality', 'poor'),
        ('wifi', 'signal', 'strong'),
        ('mouse', 'button', 'click problem'),
        ('gps', 'map', 'useful'),
        ('image', 'quality', 'sharp')
    ]
    
    # Store results for all queries
    all_results = []
    
    for aspect1, aspect2, opinion in queries:
        query_results = evaluate_results(aspect1, aspect2, opinion, review_df)
        query_name = f"{aspect1} {aspect2}:{opinion}"
        row_data = {'Query': query_name}
        
        for method in range(1, 7):
            method_key = f'method{method}'
            row_data.update({
                f'M{method}_Ret': query_results[method_key]['#Ret'],
                f'M{method}_Score': query_results[method_key]['FinalScore']
            })
        
        all_results.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save detailed results
    df.to_csv('output/output/evaluation_results.csv', index=False)
    
    # Print formatted table
    print("\nResults Table:")
    for method in range(1, 7):
        print(f"\nMethod {method}:")
        method_df = df[['Query', f'M{method}_Ret', f'M{method}_Score']]
        method_df.columns = ['Query', '#Ret', 'Score']
        print(tabulate(method_df, headers='keys', tablefmt='pipe', floatfmt=".3f"))
    
    return df

if __name__ == "__main__":
    # Load the review DataFrame
    review_df = pd.read_pickle("reviews_segment.pkl")
    
    # Create output directory if it doesn't exist
    os.makedirs('output/output', exist_ok=True)
    
    # Generate results table
    results_df = create_results_table(review_df)
    
    print("\nResults have been saved to 'output/output/evaluation_results.csv'")