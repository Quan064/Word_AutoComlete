# TOPIC-AWARE AUTOCOMPLETE

# I. Introduction

### Problem Background and Objective

In modern text editing systems, the autocomplete feature plays a crucial role in:

* Increasing typing speed
* Reducing spelling errors
* Helping users express ideas more effectively

However, traditional autocomplete methods are typically based only on:

* Prefix matching
* Word frequency

As a result, they suffer from significant limitations:

* They do not understand the context of the sentence being written
* They cannot distinguish between words that share the same prefix but have different meanings



### Problem Statement

Consider the following example:

```
machine learning is used for image p...

A traditional system might suggest:
- people
- police
- party

While the contextually correct words are:
- processing
- prediction
```

This illustrates the need for a system that can:

* understand semantic context
* combine it with fast prefix-based search



### Objectives of the Study

This project aims to develop an autocomplete system for English text that is capable of:

* Suggesting words based on the prefix entered by the user
* Ensuring that the suggestions are consistent with the topic of the text

This is achieved by combining:

* **Trie** → Fast prefix-based search
* **LDA & CGS** → Topic modeling for the text

These algorithms are selected due to their low time complexity, simplicity, and ease of implementation.



# II. Algorithm Overview

## 1. Trie Data Structure

### Definition

A **Trie** (Prefix Tree) is a tree-based data structure used to store and query a collection of strings. Each node in a Trie represents a character.

### Implementation

For example, with the following words:

```
cat, car, dog
```

The Trie would have the following structure:


```
(root)
 ├── c
 │    ├── a
 │         ├── t
 │         └── r
 └── d
      └── o
           └── g
```


### Prefix Search

When the user enters: `ca`

The Trie will:

* Traverse along the branch `c` → `a`
* Explore all words in the subtree below

→ return:

```"
cat, car
```


### Ranking Results by Frequency (Heap Sort)

#### Problem

A Trie only returns a list of words that match the given prefix, but it **does not rank them by popularity**. In practice, users expect the **most frequently used words** to appear first in the suggestions.

#### Solution: Heap Sort by Frequency

**Idea:**

* Each word node in the Trie additionally stores its **frequency of occurrence**
* During search, after traversing the Trie branch, use a **Heap (Min-Heap)** to rank words by descending frequency
* Return the top-K results with the highest frequency


#### Data Structure of the  Trie node

```
Node {
  char: character,
  children: Map<char, Node>,
  frequency: int,        // ← Thêm tần suất
  is_end_of_word: bool
}
```

#### Example

For the keyword `ca`:

```
Trie returns: cat, car, can, call, case

Frequencies:
- cat:  100 (common)
- car:  450 (very common)
- can:  320 (fairly common)
- call: 150 (less common)
- case:  80 (rare)

After Heap Sort (top-3):
1. car  (450)
2. can  (320)
3. cat  (100)
```

### Advantages and Limitations

* Advantages:

  * Fast prefix-based search
  * Well-suited for autocomplete

* Limitations:

  * Does not understand the meaning of suggested words
  * Does not consider context



## 2. LDA Model (Latent Dirichlet Allocation)

### Main Idea

LDA is a probabilistic model used to discover “hidden topics” within a collection of documents.

### Assumptions of LDA

LDA assumes that:

* Each document is a mixture of multiple topics
* Each topic is a probability distribution over words



### Intuitive Example

A document may consist of:

* `technology` : 70%
* `education` : 30%

The `technology` topic may include:

* `code`, `AI`, `data`, `algorithm`, $\dots$

### Probabilistic Representation

Based on these assumptions, LDA defines two probability distributions:

1. Word distribution given a topic: $P(\text{word} \mid \text{topic})$

2. Topic distribution given a document: $P(\text{topic} \mid \text{document})$



### Application in the Problem

In the autocomplete system:

* The previously typed text (context) is treated as a “document”
* LDA uses an algorithm called **CGS** to infer $P(\text{topic} \mid \text{context})$, thereby identifying the current semantic context

## 3. CGS Algorithm (Collapsed Gibbs Sampling)

### Idea

CGS is an inference method in LDA based on sequential sampling, where topic assignments for each word in a document are iteratively updated according to the number of topics.

CGS is based on two assumptions:

* A document typically has a dominant topic
* A word typically has a dominant meaning



### CGS Procedure

Given a corpus of documents, the number of topics $(K)$, and hyperparameters $\alpha$ and $\beta$ (sometimes denoted as $\eta$):

1. Randomly assign a topic to each word in all documents.
   (Identical words may be assigned different topics)

Consider a specific document $d$:

1. Count how many times each topic $k$ appears in document $d$.
2. Count how many times each word appears in each topic across the entire dataset.
3. In document $d$, remove the current topic assignment of a word $w_{d,n}$.
4. Assign a new topic to $w_{d,n}$ based on:

   * Which topics are prevalent in document $d$ (from step 2)
   * The frequency of $w_{d,n}$ in each topic (from step 3)

   Formula:

  $$
  \frac{n_{d,k}+\alpha}{\sum^K_in_{d,i}+\alpha K}\times\frac{m_{w,k}+\beta}{\sum^V_im_{i,k}+\beta V}
  $$

   Where:

   * $n_{d,k}$: Number of words assigned to topic $k$ in document $d$
   * $\sum^K_i n_{d,i}$: Total number of words in document $d$
   * $m_{w,k}$: Number of times $w_{d,n}$ appears in topic $k$
   * $\sum^V_i m_{i,k}$: Total number of word occurrences in topic $k$
   * $\alpha \in [0, 1]$: Controls topic distribution across documents
   * $\beta \in [0, 1]$: Controls word distribution within topics

   The result of this formula is the probability of assigning topic $k$ to $w_{d,n}$ (sampling proportionally).

↻ Repeat steps 2–5 until convergence or a stopping condition is met.



### Results

After many iterations, CGS yields:

* Word distribution over topics — $\phi$, where $\phi_{k,w} = P(\text{word} = w \mid \text{topic} = k)$
* Topic distribution over documents — $\theta$, where $\theta_{d,k} = P(\text{topic} = k \mid \text{document} = d)$

### Advantages and Limitations

#### Advantages

* Simple and easy to implement
* Interpretable: it is possible to understand how a topic is assigned to a word

#### Limitations

* Computationally slow due to many iterations (not directly suitable for real-time use without optimization)
* Treats text as a bag of words, thus ignoring positional relationships between words



## 4. Combining Trie and LDA

### 4.1. Data Preparation (Offline)

#### Step 1 - Data Collection and Preprocessing

* **LDA** and **Trie**

  * **Lowercase**: convert all text to lowercase
  * **Tokenization**: transform text into discrete units (usually words) called tokens
  * **Non-alphabetic tokens filtering**: remove tokens that are not alphabetic
  * **Punctuation removal**: remove punctuation marks
  * **Space removal**: remove extra whitespace
  * **Stopword removal**: remove stopwords (words that contribute little semantic meaning)
  * **POS filtering**: retain only informative parts of speech (nouns, verbs, adjectives)
  * **Lemmatization**: reduce different forms of a word to a single canonical form

The steps in LDA and Trie(Stopword removal, POS filtering, and Lemmatization) help reduce computational cost by eliminating high-frequency but semantically insignificant words, which may otherwise introduce noise into the model.


#### Step 2 - Training LDA

Use the corpus to train the LDA model and obtain:

1. **Topic → Word distribution**

$$
P(\text{word} \mid \text{topic})
$$

Example:

| Topic                | WordId_0: machine | WordId_1: learning | WordId_2: cat | WordId_3: fish |
| -------------------- | ----------------- | ------------------ | ------------- | -------------- |
| Topic_1 (Technology) | 0.45              | 0.45               | 0.05          | 0.05           |
| Topic_2 (Animal)     | 0.02              | 0.03               | 0.60          | 0.35           |

2. **Word → Topic distribution**

$$
P(\text{topic} \mid \text{word})
$$

**Transformation**

| Word                | Topic Distribution [Technology, Animal] |
| ------------------- | :-------------------------------------: |
| WordId_0 (machine)  |               [0.45, 0.02]              |
| WordId_1 (learning) |               [0.45, 0.03]              |
| WordId_2 (cat)      |               [0.05, 0.60]              |
| WordId_3 (fish)     |               [0.05, 0.35]              |

**Purpose**: determine the proportion of each topic associated with a given word.


#### Step 3 - Building the Trie

From the entire vocabulary, construct a **Trie index**.

Each word in the Trie additionally stores:

* frequency
* topic distribution

Example:

```
machine
- frequency: 3200
- topic:
  + Technology: 0.45
  + Animal: 0.02
```


**Purpose**

The Trie not only stores words but also retains **topic-related information**.

---

### 4.2. Suggestion Process During User Input (Online)

Suppose the user is typing:

```
machine learning is very po
```

We split it into two parts:

```
context = "machine learning is very"
prefix  = "po"
```


---

### 4.3. Context Analysis with LDA

#### Step 1 — Tokenize the Input Context (Stopword Removal, POS Tagging, and Lemmatization)

(~~"is"~~, ~~"very"~~)

```
["machine", "learning"]
```

#### Step 2 — Infer Topic Distribution

We estimate the topic distribution of the context by averaging the topic distributions of the tokens (Linearized LDA):

$$\begin{bmatrix} 0.45 \\ 0.02 \end{bmatrix} + \begin{bmatrix} 0.45 \\ 0.03 \end{bmatrix} = \begin{bmatrix} 0.90 \\ 0.05 \end{bmatrix} \xrightarrow{Normalize} \begin{bmatrix} 0.947 \\ 0.053 \end{bmatrix}$$

Resulting context distribution:

```
Technology: 0.947
Animal: 0.053
```

This averaging approach is significantly faster than running full CGS convergence, but it comes at the cost of reduced accuracy.


---

### 4.4. Retrieving Candidate Words from Trie

Find words that start with the given prefix.

```
prefix = "po"
```

The Trie returns:

```
power
point
policy
politics
polynomial
```


---

### 4.5. Topic-Aware Suggestion Scoring

Each word has a topic distribution:

```
power:
- freq: 6800
- topic:
  + AI: 0.40
  + Physics: 0.45
  + ...

policy:
- freq: 5000
- topic:
  + Politics: 0.80
  + ...
```

We compute the **relevance to the context** of a word using a heuristic function ($\alpha$ is a hyperparameter):

$$\text{score}(w) = \frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)} \times \left(1 + \alpha \cdot \text{cosineˍsim}(\text{topic}(w), \text{topic}(\text{context}))^2\right)$$

**Interpretation**: Words with higher scores are prioritized in the suggestions.


---

### 4.6. Ranking Results

Example autocomplete output:

| word       | score |
| ---------- | ----- |
| power      | 0.62  |
| polynomial | 0.55  |
| point      | 0.32  |
| policy     | 0.05  |


---

### 4.7. Time Complexity Analysis (Online)

#### Overall Process

The online suggestion pipeline consists of the following main steps:

1. **Tokenize & preprocess the context**
2. **Infer the topic distribution of the context (Linearized LDA)**
3. **Retrieve prefix-matching words from the Trie**
4. **Compute scores for each candidate word**
5. **Select the top-K results**

#### Step-by-Step Analysis

| Step                                 | Time Complexity         | Description                                             |
| ------------------------------------ | ----------------------- | ------------------------------------------------------- |
| **1. Tokenize & preprocess context** | $O(C)$                  | $C$ = length of the context (characters)                |
| **2. POS tagging & Lemmatization**   | $O(n)$ or $O(n \log V)$ | $n$ = number of tokens (~ $C/5$); $V$ = vocabulary size |
| **3. Compute topic distribution**    | $O(n \cdot K)$          | $n$ = number of tokens; $K$ = number of topics (~15–30) |
| **4. Normalize topic vector**        | $O(K)$                  | Normalize by dividing by total sum                      |
| **5. Traverse Trie prefix branch**   | $O(L)$                  | $L$ = prefix length (~3–5 characters)                   |
| **6. Traverse all matching words**   | $O(M)$                  | $M$ = total nodes/characters in the subtree             |
| **7. Heap sort top-K candidates**    | $O(N_m \log K)$         | $N_m$ = number of matched words (~10–100)               |
| **8. Scoring computation**           | $O(N_m \cdot K)$        | Cosine similarity per word                              |
| **9. Final top-K selection**         | $O(N_m \log K)$         | Using QuickSelect or HeapSort                           |


#### Overall Time Complexity

$$
T(n) = O(C + n \cdot K + L + N_m \cdot K + N_m \log K)
$$

**Simplified** (LDA inference and scoring dominate the runtime):

$$
T(n) \approx O(n \cdot K + N_m \cdot K)
$$

#### Practical Case Analysis

With typical system values:

* $C$ ~ 100–500 characters (context)
* $n$ ~ 10–30 tokens (after tokenization and stopword removal)
* $K$ ~ 15–30 topics (from LDA)
* $L$ ~ 3–5 characters (prefix)
* $N_m$ ~ 10–100 matching words (depending on the prefix)

| Case             | $n$ | $N_m$ | $K$ | Time (ms) | Notes                                    |
| ---------------- | --- | ----- | --- | --------- | ---------------------------------------- |
| **Best case**    | 10  | 10    | 15  | ~2–3      | Specific prefix, short context           |
| **Average case** | 20  | 50    | 20  | ~10–15    | Common prefix, moderate context          |
| **Worst case**   | 30  | 100   | 30  | ~30–50    | Generic prefix (e.g., "a"), long context |

#### Conclusion

**Dominant complexity:** $O((n + N_m) \cdot K)$

* Typically under 50 ms (suitable for real-time systems)
* Main bottleneck: scoring step with complexity $O(N_m \cdot K)$


---

### 4.7. Overall Strcuture 

```
                 Dataset
                    │
        ┌───────────┴───────────┐
        │                       │
   Train LDA                Build Trie
        │                       │
Topic-word matrix         Prefix index
        │                       │
        └──────────┬────────────┘
                   │
             User typing
                   │
        split(context, prefix)
                   │
        ┌──────────┴──────────┐
        │                     │
  LDA inference           Trie search
  P(topic|context)        prefix list
        │                     │
        └──────────┬──────────┘
                   │
           Topic-weighted ranking
                   │
              Autocomplete
```


# III. Experimental Dataset

## 1. Dataset Size

The dataset consists of **90,000 documents** (sourced from the *CNN News Corpus*):

* **Training set:** 54,000 documents (60%)
* **Validation set:** 18,000 documents (20%)
* **Test set:** 18,000 documents (20%)

## 2. Implementation Procedure

### 2.1. Hyperparameter Selection

Before training, define the model configurations to compare:

* Number of topics: $K = 15, 20, 25, 30$
* Dirichlet parameters: $\alpha$, $\beta$

Each configuration corresponds to a different LDA model.

### 2.2. Training on the Training Set

For each configuration:

1. Use **54,000 documents** from the training set
2. Run **Gibbs Sampling** for multiple iterations
3. After convergence, obtain:

   * Topic–word distributions: $\phi_k$
   * Document–topic distributions: $\theta_d$

These parameters remain fixed after training.

### 2.3. Topic Inference on the Validation Set

Apply the trained model to **18,000 validation documents**:

* Initialize topic assignments randomly

During this process:

* **Do not update** the topic–word distribution $\phi_k$
* Only perform **Linearized LDA** to infer $\theta_d$ for the validation set

### 2.4. Model Evaluation on the Validation Set

After inference:

> Evaluate configurations based on human subjective judgment. This approach is chosen instead of relying on metrics (e.g., Coherence), as there is ongoing debate regarding their reliability, since they may not align well with the actual quality of topic models.

→ Compare configurations to select the best model.

Then, use the best configuration to design a heuristic function (combining Trie and LDA) that achieves the highest **Hit@K** score (the proportion of queries where the correct result appears in the top K) on the validation set.

### 2.5. Final Evaluation

Use **18,000 test documents** (fully independent):

* Compare:

  * Combined model: **Trie Freq + LDA** (best configuration + selected heuristic)
  * Model using **Trie Freq**
  * Model using only **Trie**

* Metric:

  * **Hit@K** (the proportion of queries where the correct result appears in the top K)


## 3. Significance of Data Splitting

This data split helps to:

* Reduce **overfitting**
* Ensure objective evaluation on unseen data
* Accurately reflect the model’s generalization performance

# IV. Evaluation

## 1. Analysis of the LDA Training Set

The training dataset for LDA includes:

* **Before preprocessing:**

  * Number of documents: 53,928
  * Total number of tokens: 34,707,979
  * Vocabulary size: 189,878
  * Average document length: 643.60
  * Standard deviation of document length: 337.28

* **After preprocessing:**

  * Number of documents: 53,926
  * Total number of tokens: 13,435,668
  * Vocabulary size: 25,185
  * Average document length: 249.15
  * Standard deviation of document length: 131.98

**Analysis Table**


<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Histogram%20of%20Document%20Lengths.png?raw=true" alt="Not found">
  <figcaption>Distribution of document lengths in the LDA training set after preprocessing</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Total%20Tokens.png?raw=true" alt="Not found">
  <figcaption>Vocabulary size before and after preprocessing</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Vocabulary%20Size.png?raw=true" alt="Not found">
  <figcaption>Total number of tokens before and after preprocessing</figcaption>
</figure>

**Observations**

The preprocessing is highly effective:

* Vocabulary reduced from 189,878 → 25,185 (~87% reduction)
* Total tokens reduced from 34.7M → 13.4M (~61% reduction)
* Not only noise is removed, but most documents are preserved (53,928 → 53,926): almost no data loss

→ The training set is thoroughly preprocessed, with sufficient data and a well-balanced distribution → providing strong conditions for training LDA to achieve high-quality results


## 2. Hyperparameter Selection and Heuristic Function

### 2.1. Choosing the Number of Topics (K) for LDA

#### Experiments with Different Values of K

We experiment with LDA models using different values of $K$ (number of topics):
$K \in {5, 10, 15, 20, 25, 30, \dots, 400}$

#### Evaluation Methods

1. **Qualitative evaluation based on human judgment**
   Select the value of $K$ that produces the most reasonable topic distributions.

2. **Topic Coherence** ($C_v$: measures how frequently words within the same topic co-occur)
   Although previously noted that this metric may not always be reliable, we still consider selecting a value of $K$ based on this method in case it yields better **Hit@K** performance for the heuristic function.


#### Results

1. **Evaluation based on human judgment**: $K = 20$

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/LDA_CGS/Topic_Distribution_K20.png?raw=true" alt="Not found">
  <figcaption>Topic distribution when the number of topics is 20</figcaption>
</figure>

2. **Topic Coherence** ($C_v$): $K = 170$

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/LDA_CGS/Coherence_K170.png?raw=true" alt="Not found">
  <figcaption>Coherence score on the validation set</figcaption>
</figure>

### 2.2. Heuristic Function for Combining Trie and LDA

#### Selected Heuristic Function

$$\text{score}(w) = \frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)} \times \left(1 + \alpha \cdot \text{cosineˍsim}(\text{topic}(w), \text{topic}(\text{context}))^2\right)$$


Where:

* $w$: suggested word
* $\text{freq}(w)$: frequency of word $w$ in the training set
* $\text{maxˍfreq}$: maximum frequency in the vocabulary
* $\text{topic}(w)$: topic distribution of word $w$ (K-dimensional vector)
* $\text{topic}(\text{context})$: topic distribution of the current context
* $\alpha$: hyperparameter controlling the weight of topic relevance (typically $\alpha \in [0.5, 2.0]$)


#### Component Analysis

**1. Frequency Component: $\frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)}$**


| Property           | Explanation                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Why logarithm?** | Raw frequencies follow a power-law distribution (a few words are extremely frequent) → log helps normalize |
| **Why +1?**        | Avoid $\log(0)$ when freq = 0                                                                              |
| **Value range**    | $[0, 1]$ → easy to combine with other components                                                           |
| **Meaning**        | Prevent overly frequent words from dominating more semantically relevant ones                              |

**2. Topic Component: $1 + \alpha \cdot \text{cosineˍsim}(...)^2$**

| Property                    | Explanation                                                              |
| --------------------------- | ------------------------------------------------------------------------ |
| **Cosine similarity**       | Measures the angle between two topic vectors → range [0, 1]              |
| **Squared term ($^2$)**     | Amplifies differences → clearer topic relevance                          |
| **Leading $1 +$**           | Ensures that completely unrelated words (sim = 0) still have a score > 0 |
| **Hyperparameter $\alpha$** | Controls the level of topic sensitivity                                  |


#### Why Multiply the Two Components?

$$\text{score}(w) = \underbrace{\frac{\log(\text{freq})}{\log(\text{maxˍfreq})}}_\text{phần A: popularity} \times \underbrace{(1 + \alpha \cdot \text{sim}^2)}_\text{phần B: relevance}$$

**Intuition:**

* A contextually relevant but rare word → moderate score
* A frequent but irrelevant word → low score
* A word that is both frequent and contextually relevant → high score

#### Note

The heuristic function and hyperparameter $\alpha$ are tuned using the validation set.

### 2.3 Conclusion

* Best $K$: 20
* Best $\alpha$ values corresponding to prefix lengths from 1 to 6:
  $[0.1, 0.5, 1.0, 1.5, 2.0, 2.5]$


## 3. Overall Evaluation Results

The overall evaluation is conducted on the test set (18,000 documents), where 10 random tokens are selected from each document as queries.

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Analysis/Model_comp.png?raw=true" alt="Not found">
  <figcaption>Overall comparison of autocomplete models on the test set (queries for each prefix length ~1.8 million)</figcaption>
</figure>

### Observations

The comparison shows that the Trie + LDA model does not demonstrate a clearly superior performance compared to Trie Freq. Moreover, if the test set contains contexts with higher topic dispersion, the performance of Trie + LDA and Trie Freq is likely to converge.

Below are some possible reasons for this outcome:

1. **Words in documents are often not strongly semantic or topic-specific**:
   Word frequency (Trie Freq) is already sufficient for generating good suggestions, and topic information from LDA does not provide significant additional benefit.

2. **Information loss in Linearized LDA**:
   Using simple averaging instead of full Collapsed Gibbs Sampling reduces inference accuracy.

3. **Suboptimal heuristic function**:
   The frequency component may dominate the scoring, overshadowing the topic component. The hyperparameter $\alpha$ may require further tuning.

4. **CGS is limited in capturing semantic relationships**:
   CGS relies on co-occurrence statistics and does not truly capture semantic similarity between words. For example, words like "neural" and "deep" may not be considered semantically close if they appear in different contexts within the corpus. In contrast, word embeddings (e.g., Word2Vec, FastText, BERT) can capture deeper semantic relationships and potentially improve suggestion quality.

# V. The Application

## 1. The Application Interface and User Interaction

After receiving the Trie + LDA model, we build the auto-suggestion application by using Streamlit - an open-source Python framework that allows developers to create and share custom web apps for machine learning and data science. 

Our application works as follow:

* When the user accesses the application, it will load all needed components (including the pre-trained Trie + LDA model, the widgets (user input section, application settings (theme mode), the text deletion button, and the model's information (the size of the dictionary, the model's status))).
* The user then writes texts into the input section, and the application will bring out suggested words (5 words at most due to the application's settings).
* The user then can choose one of the suggested words by clicking the word's button, and the application will fill the chosen word into the input section.
* When receiving the chosen word, the user can choose to continue writing and by doing so, receiving suggested words, or deletting the whole text input by clicking the all-text deletion button. The user can also self-delete the texts by using the backspace in the keyboard.

## 2. The Strengths of The Application

The Streamlit application has several strengths, including:

* **Quick user interface and model loading:** The application loads all needed components quickly and efficiently.
* **Quick word suggestion:** The application suggests words based on the current prefix in the input section with low latency.
* **Quick word-filling into the input section:** The application fills the chosen word into the input section in a short amount of time.
* **Having a huge word dictionary thanks to the Trie model:** The application receives the dictionary of 61 815 distinct vocabularies.
* **Saving user's input history (when the user clicks the deletion button):** The application can save previous texts when users clicked the text-deletion button.
* **Personal customization (changing theme: Light/Dark/System mode):**

## 3. The Drawback of The Application

Despite the strong points of the application, there is a weakness that we have not been able to deal with:

* **The lost of the mouse cursor's current position when receiving chosen words into the input section:** We noticed that when filling the chosen word into a long input paragraph, the application failed to keep track of the last position of the paragraph (the chosen word's position). Unfortunately, we have not yet found a proper way to solve this irritating issue, partly due to the lack of time.

# VI. Conclusion

Although the results between Trie with Frequency and Trie with Frequency and LDA Context are not varied much from each other, it is still suitable to build a realtime auto-suggestion application due to its efficient word-suggestion speed. If possible, we wish we were able to improve the model's performance more and tackle the application's weakness.

## Member's Job Division

**Thong Minh Quan, Leader:** 

* Suggesting the project, the trie, and LDA model.

* Assigning jobs to every team members.

* Supervising other members' jobs.

* Writing teamwork's rules.

* Implementing the trie, the model.

* Writing the readme report.

* Creating videos.

* Talking about the project in the videos.

**Hoang Nguyen The Hien, Team Member:**

* Suggesting Streamlit to build the application.

* Finding information about the evaluation method.

* Supervising other members' jobs.

* Doing the data-preprocessing task.

* Implementing and improving the trie, the LDA model, and the Hit@K comparison between the three methods.

* Implementing the Streamlit application.

* Writing the readme report.

* Talking about the project in the videos.

**Vo Le Nam Khanh Ann, Team Member:**

* Suggesting evaluation method.

* Finding information about the evaluation method.

* Supervising other members' jobs.

* Doing the data-preprocessing task.

* Writing the readme report.

* Creating videos.

*Talking about the project in the videos.

*Project Dataset: 
