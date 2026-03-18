import pickle
import tomotopy as tp
import re
import spacy
from Trie.trie import Trie, TrieNode


def load_models():
    """Load pre-trained LDA model and Trie"""
    # Load LDA model
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")
    
    # Build word to id mapping
    word_to_id = {}
    for idx in range(lda_model.num_vocabs):
        word = lda_model.used_vocabs[idx]
        word_to_id[word] = idx
    
    # Load Trie
    with open("Trie/Trie.pkl", 'rb') as f:
        trie = pickle.load(f)
    
    # Load spaCy for tokenization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    topic_word_matrix = [
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ]
    
    return lda_model, trie, word_to_id, nlp, topic_word_matrix


def tokenize(text, nlp):
    """
    Tokenize text with lemmatization (matching LDA training preprocessing)
    
    Xử lý:
    1. Lemmatization: "running" -> "run", "learns" -> "learn"
    2. Loại bỏ stopwords: "is", "the", "a"
    3. Chỉ giữ NOUN, VERB, ADJ
    4. Loại bỏ punctuation và khoảng trắng
    """
    doc = nlp(text.lower())
    lemmatized_tokens = [
        t.lemma_ for t in doc if
        t.is_alpha and
        not t.is_punct and
        not t.is_space and
        not t.is_stop and
        t.pos_ in ['NOUN', 'VERB', 'ADJ']
    ]
    return lemmatized_tokens


def infer_topic_distribution(lda_model, context_tokens):
    """
    Infer topic distribution P(topic | context) using LDA
    Returns array where index is topic_id and value is probability
    """
    if not context_tokens:
        # Return uniform distribution if no context
        return [0.0] * lda_model.k
    
    # Create a temporary document with context tokens
    doc = lda_model.make_doc(context_tokens)
    
    # Infer topic distribution - returns (topic_dist_array, log_likelihood)
    topic_dist_array, _ = lda_model.infer(doc)
    
    return topic_dist_array


def normalize_word(word, nlp):
    """Normalize a word the same way training data was normalized."""
    doc = nlp(word.lower())
    if len(doc) == 0:
        return word.lower()

    token = doc[0]
    if not token.is_alpha or token.is_punct or token.is_space:
        return word.lower()

    return token.lemma_


def get_word_topic_distribution(lda_model, word, word_to_id, nlp, topic_word_matrix):
    """
    Get P(word | topic) for each topic
    Returns array where index is topic_id and value is P(word | topic)
    """
    # Normalize word to match LDA vocabulary (lemmatization + lower-case)
    norm_word = normalize_word(word, nlp)

    # Get word ID
    word_id = word_to_id.get(norm_word)
    if word_id is None:
        # Word not in vocabulary, return uniform distribution
        return [0.0] * lda_model.k
    
    return [topic_word_matrix[k][word_id] for k in range(len(topic_word_matrix))]


def calculate_suggestion_score(context_topic_dist, word_topic_dist):
    """
    Calculate suggestion score based on conditional probability
    score(word) = Σ P(topic | context) * P(word | topic)
    Both inputs are arrays/lists where index represents topic_id
    
    Ý nghĩa: từ nào thuộc topic giống với context sẽ được ưu tiên.
    """
    score = 0.0
    for topic_id in range(len(context_topic_dist)):
        score += context_topic_dist[topic_id] * word_topic_dist[topic_id]
    
    return score


def suggest_words(lda_model, trie, word_to_id, nlp, topic_word_matrix, user_input, num_suggestions=5, verbose=False):
    """
    Main suggestion pipeline
    
    Quy trình:
    2. Tách input thành context và prefix
    3. Phân tích ngữ cảnh bằng LDA: P(topic | context)
    4. Lấy candidate từ Trie (bắt đầu bằng prefix)
    5. Tính điểm gợi ý: score(word) = Σ P(topic | context) * P(word | topic)
    6. Xếp hạng kết quả
    7. Trả kết quả gợi ý
    
    Args:
        lda_model: Trained LDA model
        trie: Trie data structure with words
        word_to_id: Dictionary mapping words to IDs in LDA model
        nlp: spaCy language model for tokenization
        user_input: User's input text (e.g., "machine learning is very po")
        num_suggestions: Number of suggestions to return
        verbose: Print debug information
    
    Returns:
        List of suggested words ranked by score: [(word, score), ...]
    """
    # Step 1: Split input into context and prefix
    words = user_input.split()
    if not words:
        return []
    
    prefix = words[-1]
    context = " ".join(words[:-1]) if len(words) > 1 else ""
    
    if verbose:
        print(f"  Context: '{context}'")
        print(f"  Prefix: '{prefix}'")
    
    # Step 2: Tokenize context with lemmatization
    context_tokens = tokenize(context, nlp)
    
    if verbose:
        print(f"  Context tokens (after lemmatization): {context_tokens}")
    
    # Step 3: Infer topic distribution from context
    context_topic_dist = infer_topic_distribution(lda_model, context_tokens)
    
    # Find top topics
    if verbose:
        top_topics = sorted(
            [(i, prob) for i, prob in enumerate(context_topic_dist)],
            key=lambda x: -x[1]
        )[:3]
    
        print(f"  Top topics from context: {top_topics}")
    
    # Step 4: Get candidate words from Trie
    candidates = trie.topK(prefix, num_suggestions * 3)  # Get more candidates to filter
    
    if not candidates:
        if verbose:
            print(f"  No candidates found for prefix '{prefix}'")
        return []
    
    if verbose:
        print(f"  Candidates from Trie: {[word for word, _ in candidates[:5]]}")
    
    # Step 5: Calculate scores for each candidate
    scored_candidates = []
    for word, freq in candidates:
        word_topic_dist = get_word_topic_distribution(lda_model, word, word_to_id, nlp, topic_word_matrix)
        score = calculate_suggestion_score(context_topic_dist, word_topic_dist)
        scored_candidates.append((word, score, freq))
    
    # Step 6: Sort by score (descending)
    # alpha = 0.8
    # scored_candidates.sort(
    #     key=lambda x: alpha * x[1] + (1 - alpha) * x[2],
    #     reverse=True
    # )
    scored_candidates.sort(
        key=lambda x: x[1:],
        reverse=True
    )
    
    # Return top suggestions
    result = [(word, score) for word, score, freq in scored_candidates[:num_suggestions]]
    
    if verbose:
        print(f"  Top suggestions: {[word for word, _ in result]}")
    
    return result


def interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix):
    """Interactive mode for real-time suggestions"""
    print("\n" + "="*60)
    print("INTERACTIVE AUTOCOMPLETE MODE")
    print("="*60)
    print("Nhập văn bản để nhận gợi ý từ (gõ 'quit' để thoát)")
    print()
    
    while True:
        user_input = input("Nhập văn bản > ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            print("Vui lòng nhập văn bản")
            continue
        
        print()
        suggestions = suggest_words(
            lda_model, trie, word_to_id, nlp, topic_word_matrix, user_input,
            num_suggestions=10, verbose=True
        )
        
        print("\nGợi ý kết quả:")
        if suggestions:
            for i, (word, score) in enumerate(suggestions, 1):
                print(f"  {i}. {word:20s} (score: {score:.6f})")
        else:
            print("  Không tìm thấy gợi ý")
        
        print()


def main():
    """Main autocomplete function"""
    print("="*60)
    print("WORD AUTOCOMPLETE - LDA + TRIE PIPELINE")
    print("="*60)
    print("\nLoading models...")
    lda_model, trie, word_to_id, nlp, topic_word_matrix = load_models()
    print(f"✓ LDA Model loaded: {lda_model.k} topics")
    print(f"✓ Trie loaded successfully")
    print(f"✓ Vocabulary size: {lda_model.num_vocabs}")
    print(f"✓ spaCy model loaded for tokenization")
    print()
    
    interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix)


if __name__ == "__main__":
    main()
