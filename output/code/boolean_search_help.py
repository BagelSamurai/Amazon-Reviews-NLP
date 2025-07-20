import argparse
import pickle
import json
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import string
import pandas as pd
from tqdm import tqdm
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

class BooleanSearcher:
    def __init__(self, review_df):
        self.word_to_int_mapping = {}
        self.int_to_review_mapping = {}
        self.word_lemmatizer = WordNetLemmatizer()
        self.review_df = review_df
        
        # Create necessary directories
        os.makedirs('output/code', exist_ok=True)
        os.makedirs('output/output', exist_ok=True)
        
        # Download required NLTK data
        print("Initializing NLTK resources...")
        for nltk_resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{nltk_resource}' if nltk_resource == 'punkt' 
                              else f'corpora/{nltk_resource}')
            except LookupError:
                nltk.download(nltk_resource, quiet=True)
                
        self.excluded_stop_words = set(stopwords.words('english'))
        self.excluded_punctuation_marks = set(string.punctuation)
        
        if not self.load_mappings():
            print("Building new mappings...")
            self.build_mappings()
            
        self.save_code()
        
    def save_code(self):
        """Save a copy of the current script to output/code directory"""
        try:
            current_file = os.path.abspath(__file__)
            shutil.copy2(current_file, 'output/code/boolean_search_help.py')
        except Exception as e:
            print(f"Warning: Could not save code copy: {e}")
   
    def preprocess_review_text(self, review_content):
        """Tokenize, lemmatize and preprocess text using NLTK"""
        if pd.isna(review_content):
            return []
        raw_word_tokens = word_tokenize(str(review_content).lower())
        processed_word_tokens = [self.word_lemmatizer.lemmatize(current_token)
                for current_token in raw_word_tokens 
                if current_token not in self.excluded_punctuation_marks 
                and current_token not in self.excluded_stop_words
                and current_token.isalnum()]
        return processed_word_tokens
        
    def load_mappings(self):
        """Try to load existing mappings"""
        try:
            if not os.path.exists('output/output/posting_list.pkl') or \
               not os.path.exists('output/output/int_to_review_mapping.pkl'):
                return False
                
            with open('output/output/posting_list.pkl', 'rb') as f:
                self.word_to_int_mapping = pickle.load(f)
            with open('output/output/int_to_review_mapping.pkl', 'rb') as f:
                self.int_to_review_mapping = pickle.load(f)
                
            print("Successfully loaded existing mappings")
            print(f"Vocabulary size: {len(self.word_to_int_mapping)} words")
            return True
            
        except Exception as e:
            print(f"Could not load mappings: {e}")
            return False
            
    def build_mappings(self):
        """Build posting list and review ID mapping"""
        print("Building mappings...")
        
        # First build integer to review ID mapping using actual review IDs
        self.int_to_review_mapping = {}
        for i, row in tqdm(enumerate(self.review_df.itertuples()), desc="Building review ID mapping", total=len(self.review_df)):
            self.int_to_review_mapping[i] = row.review_id.strip("'").strip('"')
        
        # Then build posting list
        print("Building posting list...")
        for idx, row in tqdm(enumerate(self.review_df.itertuples()), desc="Processing reviews", total=len(self.review_df)):
            processed_tokens = self.preprocess_review_text(row.review_text)
            for unique_token in set(processed_tokens):
                if unique_token not in self.word_to_int_mapping:
                    self.word_to_int_mapping[unique_token] = set()
                self.word_to_int_mapping[unique_token].add(idx)
        
        print("Saving mappings...")
        
        with open('output/output/posting_list.pkl', 'wb') as f:
            pickle.dump(self.word_to_int_mapping, f)
            
        with open('output/output/int_to_review_mapping.pkl', 'wb') as f:
            pickle.dump(self.int_to_review_mapping, f)
            
        print(f"Vocabulary size: {len(self.word_to_int_mapping)} words")

    def convert_ints_to_review_ids(self, int_indices):
        """Convert integer indices to actual review IDs."""
        return [self.int_to_review_mapping[idx] for idx in sorted(int_indices)]


    def method1(self, aspect1, aspect2, opinion):
        """OR operation on everything: aspect1 OR aspect2 OR opinion"""
        aspect1 = self.word_lemmatizer.lemmatize(aspect1.lower())
        aspect2 = self.word_lemmatizer.lemmatize(aspect2.lower())
        opinion_words = [self.word_lemmatizer.lemmatize(word.lower()) for word in opinion.split()]
        
        result_set = set()
        for word in [aspect1, aspect2] + opinion_words:
            if word in self.word_to_int_mapping:
                result_set.update(self.word_to_int_mapping[word])
                
        int_results = sorted(list(result_set))
        return self.convert_ints_to_review_ids(int_results)

    def method2(self, aspect1, aspect2, opinion):
        """AND operation on everything: aspect1 AND aspect2 AND opinion"""
        aspect1 = self.word_lemmatizer.lemmatize(aspect1.lower())
        aspect2 = self.word_lemmatizer.lemmatize(aspect2.lower())
        opinion_words = [self.word_lemmatizer.lemmatize(word.lower()) for word in opinion.split()]
        
        if aspect1 not in self.word_to_int_mapping:
            return []
            
        result_set = self.word_to_int_mapping[aspect1].copy()
        for word in [aspect2] + opinion_words:
            if word not in self.word_to_int_mapping:
                return []
            result_set.intersection_update(self.word_to_int_mapping[word])
            
        int_results = sorted(list(result_set))
        return self.convert_ints_to_review_ids(int_results)

    def method3(self, aspect1, aspect2, opinion):
        """(aspect1 OR aspect2) AND opinion"""
        aspect1 = self.word_lemmatizer.lemmatize(aspect1.lower())
        aspect2 = self.word_lemmatizer.lemmatize(aspect2.lower())
        opinion_words = [self.word_lemmatizer.lemmatize(word.lower()) for word in opinion.split()]
        
        aspect_docs = set()
        for aspect in [aspect1, aspect2]:
            if aspect in self.word_to_int_mapping:
                aspect_docs.update(self.word_to_int_mapping[aspect])
        
        if not aspect_docs:
            return []
            
        result_set = aspect_docs
        for word in opinion_words:
            if word not in self.word_to_int_mapping:
                return []
            result_set.intersection_update(self.word_to_int_mapping[word])
            
        int_results = sorted(list(result_set))
        return self.convert_ints_to_review_ids(int_results)

    def is_opinion_positive(self, opinion):
        """Determine if opinion is positive"""
        positive_opinions = {'useful', 'strong', 'sharp', 'good', 'great'}
        negative_opinions = {'poor', 'weak', 'bad', 'problem'}
        
        opinion_words = opinion.lower().split()
        for word in opinion_words:
            if word in positive_opinions:
                return True
            if word in negative_opinions:
                return False
        return True

    def method4(self, aspect1, aspect2, opinion):
        """Window-based aspect-opinion scoring"""
        base_results = self.method3(aspect1, aspect2, opinion)
        scored_reviews = []
        window_size = 5  # words before/after aspect term
        
        opinion_words = set(opinion.lower().split())
        is_positive_opinion = self.is_opinion_positive(opinion)
        
        for review_id in tqdm(base_results, desc="Processing reviews"):
            # Add quotes back for DataFrame lookup
            lookup_id = f"'{review_id}'"
            rows = self.review_df[self.review_df['review_id'] == lookup_id]
            if rows.empty:
                continue
            row = rows.iloc[0]
            review_text = str(row['review_text']).lower()
            rating = int(row['customer_review_rating'])
            
            sentences = review_text.split('.')
            review_score = 0
            
            for sentence in sentences:
                words = sentence.split()
                
                aspect_positions = []
                for i, word in enumerate(words):
                    if word in [aspect1.lower(), aspect2.lower()]:
                        aspect_positions.append(i)
                
                for pos in aspect_positions:
                    start = max(0, pos - window_size)
                    end = min(len(words), pos + window_size)
                    window = words[start:end]
                    
                    if any(op_word in window for op_word in opinion_words):
                        score = 1.0
                        
                        if (is_positive_opinion and rating > 3) or \
                        (not is_positive_opinion and rating <= 3):
                            score += 0.5
                        
                        try:
                            opinion_pos = next(i for i, w in enumerate(window) 
                                        if any(op_word in w for op_word in opinion_words))
                            distance = abs(opinion_pos - (pos - start))
                            proximity_score = (window_size - distance) / window_size
                            score += proximity_score
                        except:
                            pass
                        
                        review_score = max(review_score, score)
            
            if review_score > 0:
                scored_reviews.append((review_id, review_score))
        
        scored_reviews.sort(key=lambda x: x[1], reverse=True)
        return [rid for rid, _ in scored_reviews]

    def method5(self, aspect1, aspect2, opinion, min_freq=15):
        """N-gram based filtering method"""
        # Find frequent bigrams
        frequent_bigrams = self.get_frequent_bigrams(min_freq)

        # Enhance the query by adding relevant bigrams
        expanded_query = [aspect1, aspect2] + opinion.split()
        for bigram in tqdm(frequent_bigrams, desc="Expanding query with bigrams"):
            if aspect1 in bigram or aspect2 in bigram or any(opinion_word in bigram for opinion_word in opinion.split()):
                expanded_query.extend(bigram)

        # Perform the Boolean search with the expanded query
        print("Performing n-gram enhanced Boolean search...")
        result_set = set()
        for word in set(expanded_query):
            if word in self.word_to_int_mapping:
                result_set.update(self.word_to_int_mapping[word])

        int_results = sorted(list(result_set))
        return self.convert_ints_to_review_ids(int_results)

    def get_frequent_bigrams(self, min_freq=15):
        """
        Find frequent bigrams from the review text data.
        :param min_freq: Minimum frequency for a bigram to be considered frequent.
        :return: List of frequent bigrams.
        """
        print("Finding frequent bigrams...")
        bigram_finder = BigramCollocationFinder.from_words(
            [word for text in tqdm(self.review_df['review_text'], desc="Tokenizing review text") for word in word_tokenize(text.lower())]
        )
        bigram_finder.apply_freq_filter(min_freq)
        frequent_bigrams = bigram_finder.nbest(BigramAssocMeasures.raw_freq, 1000)  # Get top 1000 bigrams
        print(f"Found {len(frequent_bigrams)} frequent bigrams.")
        return frequent_bigrams
    def method6(self, aspect1, aspect2, opinion):
        """LSA-based semantic search method"""
        print("\nStarting LSA search...")
        
        # Create our search query
        query = f"{aspect1} {aspect2} {opinion}"
        print(f"Searching for: {query}")
        
        # Get all review texts ready
        review_texts = self.review_df['review_text'].fillna('').astype(str)
        
        # Set up LSA pipeline
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('lsa', TruncatedSVD(n_components=100))
        ])
        
        # Transform all reviews
        print("Analyzing reviews...")
        review_vectors = text_pipeline.fit_transform(review_texts)
        
        # Transform query
        query_vector = text_pipeline.transform([query])
        
        # Find similar reviews
        print("Finding matches...")
        similarities = cosine_similarity(query_vector, review_vectors).flatten()
        
        # Get matching reviews (similarity > 0.3)
        matching_indices = similarities >= 0.3
        matching_reviews = self.review_df[matching_indices]
        
        # Sort by similarity
        similar_reviews = list(zip(matching_reviews['review_id'], 
                                 similarities[matching_indices],
                                 matching_reviews['customer_review_rating']))
        similar_reviews.sort(key=lambda x: x[1], reverse=True)
        
        # Save visualization data
        self.plot_search_results(similar_reviews, aspect1, aspect2, opinion)
        
        # Return review IDs only
        results = [rid.strip("'").strip('"') for rid, _, _ in similar_reviews]
        
        print(f"Found {len(results)} matching reviews!")
        return results

    def plot_search_results(self, similar_reviews, aspect1, aspect2, opinion):
        """Create visualizations for search results"""
        # Extract data
        similarities = [score for _, score, _ in similar_reviews]
        ratings = [rating for _, _, rating in similar_reviews]
        
        # Create figure with subplots
        plt.figure(figsize=(15, 5))
        
        # 1. Similarity Score Distribution
        plt.subplot(131)
        sns.histplot(similarities, bins=20)
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        
        # 2. Ratings vs Similarity
        plt.subplot(132)
        plt.scatter(similarities, ratings)
        plt.title('Review Ratings vs Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Rating')
        
        # 3. Rating Distribution of Matched Reviews
        plt.subplot(133)
        sns.histplot(ratings, bins=5)
        plt.title('Rating Distribution of Matched Reviews')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        # Save plot
        plot_filename = f"lsa_results_{aspect1}_{aspect2}_{opinion.replace(' ', '_')}.png"
        plt.savefig(os.path.join('output', 'output', plot_filename))
        plt.close()
        print(f"Visualizations saved as {plot_filename}")
    

def save_as_series(cleaned_ids, output_path):
    """Save cleaned review IDs as a pandas Series to a pickle file."""
    df = pd.DataFrame(cleaned_ids, columns=["review_id"])
    series = df["review_id"]
    with open(output_path, 'wb') as f:
        pickle.dump(series, f)
    print(f"Results saved as Series to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform the boolean search.")
    
    parser.add_argument("--aspect1", type=str, required=True, help="First word of the aspect")
    parser.add_argument("--aspect2", type=str, required=True, help="Second word of the aspect")
    parser.add_argument("--opinion", type=str, required=True, help="Opinion word(s)")
    parser.add_argument("--method", type=str, required=True, 
                       help="Method (method1/method2/method3/method4/method5/method6)")

    args = parser.parse_args()
    review_df = pd.read_pickle("reviews_segment.pkl")
    searcher = BooleanSearcher(review_df)

    method_map = {
        'method1': (searcher.method1, "OR operation"),
        'method2': (searcher.method2, "AND operation"),
        'method3': (searcher.method3, "(aspect1 OR aspect2) AND opinion"),
        'method4': (searcher.method4, "Window-based aspect-opinion scoring"),
        'method5': (searcher.method5, "N-gram based filtering"),
        'method6': (searcher.method6, "LSA semantic search")
    }

    if args.method not in method_map:
        print(f"Error: Invalid method. Choose from {list(method_map.keys())}")
        return

    method_func, method_desc = method_map[args.method]
    result = method_func(args.aspect1, args.aspect2, args.opinion)

    print(f"\n{args.method} ({method_desc}) found {len(result)} matching documents")
    print("Sample results:", result[:5])
    
    # Save results as a Pandas Series
    output_filename = f"{args.aspect1}_{args.aspect2}_{args.opinion.replace(' ', '_')}_{args.method}.pkl"
    output_path = os.path.join('output', 'output', output_filename)

    # Clean the review IDs before saving
    cleaned_result = [rid.strip("'").strip('"') for rid in result]
    save_as_series(cleaned_result, output_path)

if __name__ == "__main__":
    main()