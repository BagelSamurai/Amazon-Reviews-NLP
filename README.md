# Amazon Reviews NLP: Aspect-Based Opinion Analysis

This is a semester-long research project for the Natural Language Processing course at the University of Houston.

## 1. Introduction

The project implements a boolean search system for aspect-based opinion analysis of Amazon product reviews. The goal is to retrieve relevant reviews that match specific aspect pairs and opinions. The implementation includes three baseline boolean methods and three advanced methods that improve search relevance through rating validation, sentence-level analysis, and LSA-based semantic matching.

## 2. Getting Started

### 2.1 Installing Python Libraries

To ensure all necessary libraries are installed, use the provided `requirements.txt` file. Run the following command in your terminal:
pip install -r requirements.txt
This will install all required dependencies for the project.

### 2.2 Running the Code

The project allows you to run various search methods using the following input command. Replace `-aspect1`, `-aspect2`, `-opinion`, and `-method` with the desired values.
python booleansearchhelp.py --aspect1 
Of course. Here is the raw Markdown code for the README file. You can copy and paste this into a `README.md` file in your repository.

```markdown
# Amazon Reviews NLP: Aspect-Based Opinion Analysis

This is a semester-long research project for the Natural Language Processing course at the University of Houston.

## 1. Introduction

The project implements a boolean search system for aspect-based opinion analysis of Amazon product reviews. The goal is to retrieve relevant reviews that match specific aspect pairs and opinions. The implementation includes three baseline boolean methods and three advanced methods that improve search relevance through rating validation, sentence-level analysis, and LSA-based semantic matching.

## 2. Getting Started

### 2.1 Installing Python Libraries

To ensure all necessary libraries are installed, use the provided `requirements.txt` file. Run the following command in your terminal:

```

pip install -r requirements.txt

```

This will install all required dependencies for the project.

### 2.2 Running the Code

The project allows you to run various search methods using the following input command. Replace `-aspect1`, `-aspect2`, `-opinion`, and `-method` with the desired values.

```

python booleansearchhelp.py --aspect1 \<aspect1\> --aspect2 \<aspect2\> --opinion \<opinion\> --method \<method\>

```

## 3. Implementation Details

### 3.1 Data Structures

The system uses two primary data structures:

* **Word to Integer Mapping:** Maps preprocessed words to sets of document indices.
* **Integer to Review Mapping:** Maps integer indices to review IDs.

### 3.2 Text Preprocessing

The preprocessing pipeline includes several steps:

* Case normalization
* Stop word removal
* Punctuation removal
* Word lemmatization using NLTK
* Alphanumeric filtering

### 3.3 Search Methods

#### 3.3.1 Baseline Methods

* **Method 1: OR Operation**
    * Implements: `aspect1 OR aspect2 OR opinion`
    * Returns documents containing any search term.
    * Provides maximum recall.

* **Method 2: AND Operation**
    * Implements: `aspect1 AND aspect2 AND opinion`
    * Returns documents containing all search terms.
    * Provides maximum precision.

* **Method 3: Mixed Operation**
    * Implements: `(aspect1 OR aspect2) AND opinion`
    * Combines aspect flexibility with an opinion requirement.
    * Balances precision and recall.

#### 3.3.2 Advanced Methods

* **Method 4: Window-based Scoring**
    * Uses a sliding window approach.
    * Considers word proximity in scoring.
    * Incorporates rating validation.
    * Employs adaptive scoring based on distance.

* **Method 5: N-gram based Filtering Method**
    * Employs sentence-level tokenization to extract n-grams.
    * Identifies and evaluates aspect-opinion pairings using n-grams.
    * Cross-validates results with rating consistency.
    * Applies n-gram frequency and relevance for sentence-based scoring.

* **Method 6: LSA-based Semantic Search**
    * Uses Latent Semantic Analysis (LSA).
    * Captures semantic relationships between terms.
    * Implements cosine similarity scoring.
    * Includes visualization of results.

## 4. Results and Analysis

### 4.1 Performance Results

Results for required queries, showing the number of retrieved documents (`#Ret`) and the evaluation score.

| **Method** | **Query** | **#Ret** | **Score** |
| :----------- | :--------------------------- | :------- | :-------- |
| **Method 1** | `audio quality:poor`         | 25284    | 0.24      |
|              | `wifi signal:strong`         | 6612     | 0.25      |
|              | `mouse button:click problem` | 34087    | 0.23      |
|              | `gps map:useful`             | 8025     | 0.25      |
|              | `image quality:sharp`        | 23981    | 0.24      |
| **Method 2** | `audio quality:poor`         | 121      | 0.49      |
|              | `wifi signal:strong`         | 13       | 0.48      |
|              | `mouse button:click problem` | 94       | 0.47      |
|              | `gps map:useful`             | 84       | 0.48      |
|              | `image quality:sharp`        | 185      | 0.48      |
| **Method 3** | `audio quality:poor`         | 1510     | 0.42      |
|              | `wifi signal:strong`         | 209      | 0.41      |
|              | `mouse button:click problem` | 353      | 0.39      |
|              | `gps map:useful`             | 264      | 0.40      |
|              | `image quality:sharp`        | 737      | 0.41      |
| **Method 4** | `audio quality:poor`         | 692      | 0.39      |
|              | `wifi signal:strong`         | 108      | 0.38      |
|              | `mouse button:click problem` | 131      | 0.36      |
|              | `gps map:useful`             | 12       | 0.37      |
|              | `image quality:sharp`        | 73       | 0.37      |
| **Method 5** | `audio quality:poor`         | 34634    | 0.24      |
|              | `wifi signal:strong`         | 6612     | 0.25      |
|              | `mouse button:click problem` | 34087    | 0.23      |
|              | `gps map:useful`             | 8025     | 0.25      |
|              | `image quality:sharp`        | 34009    | 0.23      |
| **Method 6** | `audio quality:poor`         | 7096     | 0.27      |
|              | `wifi signal:strong`         | 3606     | 0.27      |
|              | `mouse button:click problem` | 2093     | 0.26      |
|              | `gps map:useful`             | 2301     | 0.27      |
|              | `image quality:sharp`        | 6974     | 0.27      |

### 4.2 Evaluation Metrics

The scores in the results represent a combined evaluation metric that consists of three main components:

#### 4.2.1 Term Presence Score (0-1)

* Measures how well the search terms appear in the retrieved documents.
* Higher values indicate better matches between search terms and document content.
* *Example: Method 2 consistently scores higher (0.47-0.49) because it finds documents containing all search terms.*

#### 4.2.2 Rating Consistency (0-1)

* Evaluates how consistent the ratings are across retrieved documents.
* Based on the standard deviation of ratings—more consistent ratings score higher.
* Normalized so that lower standard deviation gives scores closer to 1.

#### 4.2.3 Cohesion Score (0-1)

* Measures how similar the retrieved documents are to each other using TF-IDF vectors.
* Uses the silhouette score converted to a 0-1 range.
* Higher values indicate more cohesive/similar document sets.

The final score is the average of these three components, which explains the results:

* **Method 2 (AND):** Scores highest (0.47-0.49) as it finds very specific matches with all terms.
* **Methods 3 & 4:** Show moderate scores (0.36-0.42), balancing specificity and coverage.
* **Methods 1, 5, & 6:** Show lower scores (0.23-0.27) due to retrieving more diverse results.

## 5. Conclusions

The implementation demonstrates the utility of combining traditional boolean search methods with advanced features such as sentence-level analysis and Latent Semantic Analysis (LSA). While baseline methods provided foundational capabilities, the advanced methods significantly improved precision and relevance.

### 5.1 Key Findings

* **Baseline Methods:** `OR`, `AND`, and `Mixed` operations offered diverse trade-offs between recall and precision.
* **Advanced Methods:** Sentence-level analysis and LSA captured nuanced semantic and contextual relationships, enabling more relevant document retrieval.
* **Evaluation Metrics:** Precision and consistency scores validated the enhanced performance of advanced methods.

The findings underscore the importance of advanced opinion mining and semantic analysis techniques for improving search relevance in real-world applications.

## References

1.  **Text Preprocessing:** [YouTube Link](https://www.youtube.com/watch?v=hhjn4HVEdy)
2.  **Introduction to Latent Semantic Analysis:** [YouTube Link](https://www.youtube.com/watch?v=BJ0MnawUpaU)
3.  **GeeksforGeeks, N-gram Language Modelling with NLTK:** [Article Link](https://www.google.com/search?q=https://www.geeksforgeeks.org/n-gram-language-modelling-with-nltk/)
4.  **Machine Learning Geek, Latent Semantic Indexing Using Scikit-Learn:** [Article Link](https://machinelearninggeek.com/latent-semantic-indexing-using-scikit-learn/)
5.  **Claude (AI Assistant):** Suggestions on evaluation metrics and code snippets for the project.
```
