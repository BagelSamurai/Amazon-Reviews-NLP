
import pickle
import pandas as pd
from tabulate import tabulate

def load_and_analyze_results():
    """Analyze and compare results from all methods"""
    # Required queries
    queries = [
        ('audio', 'quality', 'poor'),
        ('wifi', 'signal', 'strong'),
        ('gps', 'map', 'useful'),
        ('image', 'quality', 'sharp')
    ]
    
    # Load review data
    review_df = pd.read_pickle("reviews_segment.pkl")
    
    # Results table
    results_table = []
    
    for aspect1, aspect2, opinion in queries:
        query_name = f"{aspect1}_{aspect2}_{opinion}"
        print(f"\nAnalyzing query: {aspect1} {aspect2}:{opinion}")
        
        for method in ['method1', 'method2', 'method3', 'method4', 'method5']:
            filename = f"output/output/{aspect1}_{aspect2}_{opinion}_{method}.pkl"
            try:
                with open(filename, 'rb') as f:
                    results = pickle.load(f)
                num_results = len(results)
                
                # Get sample reviews
                sample_reviews = []
                for review_id in results[:3]:  # Look at top 3 results
                    review = review_df[review_df['review_id'] == review_id].iloc[0]
                    rating = review['customer_review_rating']
                    text = review['review_text'][:100] + "..."  # First 100 chars
                    sample_reviews.append(f"Rating:{rating}, Text:{text}")
                
                results_table.append([
                    query_name,
                    method,
                    num_results,
                    "\n".join(sample_reviews)
                ])
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Print comparison table
    print("\nResults Comparison:")
    headers = ["Query", "Method", "# Results", "Sample Reviews"]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))
    
    # Save table to file
    with open('output/output/results_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(tabulate(results_table, headers=headers, tablefmt="grid"))

def run_all_queries():
    """Run all required queries with all methods"""
    queries = [
        ('audio', 'quality', 'poor'),
        ('wifi', 'signal', 'strong'),
        ('gps', 'map', 'useful'),
        ('image', 'quality', 'sharp')
    ]
    
    import subprocess
    
    print("Running all queries...")
    for aspect1, aspect2, opinion in queries:
        for method in ['method1', 'method2', 'method3', 'method4', 'method5']:
            cmd = [
                'python', 'boolean_search_help.py',
                '--aspect1', aspect1,
                '--aspect2', aspect2,
                '--opinion', opinion,
                '--method', method
            ]
            print(f"\nExecuting: {' '.join(cmd)}")
            subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test and evaluate boolean search results")
    parser.add_argument("--run", action="store_true", help="Run all queries")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    
    args = parser.parse_args()
    
    if args.run:
        run_all_queries()
    if args.analyze:
        load_and_analyze_results()
    if not (args.run or args.analyze):
        print("Please specify --run to run queries or --analyze to analyze results")
