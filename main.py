import pickle
import tomotopy as tp
import spacy
import numpy as np
from Trie.trie import Trie, TrieNode


# ================== LOAD ==================

def load_models():
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")

    word_to_id = {
        lda_model.used_vocabs[i]: i
        for i in range(lda_model.num_vocabs)
    }

    with open("Trie/Trie.pkl", 'rb') as f:
        trie = pickle.load(f)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    topic_word_matrix = np.array([
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ])

    return lda_model, trie, word_to_id, nlp, topic_word_matrix


# ================== ANALYZE CONTEXT ==================

def analyze_context(text, nlp):
    doc = nlp(text.lower())

    token_count = 0
    non_token_count = 0
    tokens = []

    for t in doc:
        if t.is_alpha and not t.is_stop and t.pos_ in ['NOUN', 'VERB', 'ADJ']:
            tokens.append(t.lemma_)
            token_count += 1
        else:
            non_token_count += 1

    return tokens, token_count, non_token_count


# ================== LDA ==================

def infer_topic_distribution(lda_model, context_tokens):
    if not context_tokens:
        return np.zeros(lda_model.k)

    doc = lda_model.make_doc(context_tokens)
    topic_dist, _ = lda_model.infer(doc)
    return np.array(topic_dist)


# ================== ENTROPY ==================

def topic_entropy(dist):
    dist = dist + 1e-9
    return -np.sum(dist * np.log(dist))


# ================== NORMALIZE (CACHE) ==================

normalize_cache = {}

def normalize_word_cached(word, nlp):
    if word in normalize_cache:
        return normalize_cache[word]

    doc = nlp(word.lower())
    if len(doc) == 0:
        normalize_cache[word] = word.lower()
        return normalize_cache[word]

    token = doc[0]
    lemma = token.lemma_ if token.is_alpha else word.lower()

    normalize_cache[word] = lemma
    return lemma


# ================== MAIN ==================

def suggest_words(
    lda_model,
    trie,
    word_to_id,
    nlp,
    topic_word_matrix,
    user_input,
    num_suggestions=5,
    verbose=False
):
    words_input = user_input.split()
    if not words_input:
        return []

    prefix = words_input[-1]
    context = " ".join(words_input[:-1]) if len(words_input) > 1 else ""

    if verbose:
        print(f"Context: '{context}'")
        print(f"Prefix: '{prefix}'")

    # ===== ANALYZE CONTEXT =====
    context_tokens, token_count, non_token_count = analyze_context(context, nlp)
    context_topic_dist = infer_topic_distribution(lda_model, context_tokens)

    if context_topic_dist.sum() == 0:
        return []

    # ===== TRIE CANDIDATES =====
    candidates = trie.topK(prefix, num_suggestions * 20)
    if not candidates:
        return []

    # ===== GROUP BY LEMMA =====
    lemma_groups = {}
    lemma_freq = {}

    for word, freq in candidates:
        lemma = normalize_word_cached(word, nlp)

        if lemma not in word_to_id:
            continue

        if lemma not in lemma_groups:
            lemma_groups[lemma] = []

        lemma_groups[lemma].append((word, freq))
        lemma_freq[lemma] = max(lemma_freq.get(lemma, 0), freq)

    if not lemma_groups:
        return []

    # ===== VECTORIZE =====
    lemmas = list(lemma_groups.keys())
    word_ids = np.array([word_to_id[l] for l in lemmas])

    word_topic_matrix = topic_word_matrix[:, word_ids]

    # ===== LDA SCORE =====
    top_k = 3
    top_idx = np.argsort(context_topic_dist)[-top_k:]

    context_top = context_topic_dist[top_idx]
    word_top = word_topic_matrix[top_idx, :]

    score_lda = context_top @ word_top

    # ===== FREQ SCORE =====
    score_trie = np.array([lemma_freq[l] for l in lemmas])

    # ===== CHỌN WORD ĐẠI DIỆN =====
    final_words = []
    for lemma in lemmas:
        best_word = max(lemma_groups[lemma], key=lambda x: x[1])[0]
        final_words.append(best_word)

    # ===== TÍNH SLOT (CORE IDEA) =====
    semantic_ratio = token_count / (token_count + non_token_count + 1e-9)

    entropy = topic_entropy(context_topic_dist)
    entropy_norm = entropy / np.log(len(context_topic_dist))

    lda_weight = semantic_ratio * (1 - entropy_norm)
    lda_weight = np.clip(lda_weight, 0.1, 0.9)

    num_lda = int(num_suggestions * lda_weight)
    num_freq = num_suggestions - num_lda

    if verbose:
        print(f"semantic_ratio={semantic_ratio:.3f}, entropy={entropy_norm:.3f}")
        print(f"→ LDA={num_lda}, FREQ={num_freq}")

    # ===== RANK LDA =====
    lda_idx = np.argsort(score_lda)[::-1]
    lda_top = [final_words[i] for i in lda_idx[:num_lda]]

    # ===== RANK FREQ =====
    freq_idx = np.argsort(score_trie)[::-1]

    freq_top = []
    for i in freq_idx:
        w = final_words[i]
        if w not in lda_top:
            freq_top.append(w)
        if len(freq_top) >= num_freq:
            break

    # ===== MERGE (INTERLEAVE) =====
    result_words = []
    for i in range(max(len(lda_top), len(freq_top))):
        if i < len(lda_top):
            result_words.append(lda_top[i])
        if i < len(freq_top):
            result_words.append(freq_top[i])

    result_words = result_words[:num_suggestions]

    # ===== BUILD RESULT =====
    result = [(w, 0.0) for w in result_words]  # score không còn ý nghĩa chính

    if verbose:
        print("Top suggestions:", result)

    return result


# ================== INTERACTIVE ==================

def interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix):
    print("\n=== AUTOCOMPLETE MODE ===\n")

    while True:
        user_input = input("Nhập > ").strip()

        if user_input.lower() == "quit":
            break

        suggestions = suggest_words(
            lda_model,
            trie,
            word_to_id,
            nlp,
            topic_word_matrix,
            user_input,
            num_suggestions=10,
            verbose=True
        )

        print("\nGợi ý:")
        for i, (w, _) in enumerate(suggestions, 1):
            print(f"{i}. {w}")

        print()


# ================== MAIN ==================

def main():
    print("Loading...")
    lda_model, trie, word_to_id, nlp, topic_word_matrix = load_models()

    print(f"LDA topics: {lda_model.k}")
    print(f"Vocab size: {lda_model.num_vocabs}")

    interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix)


if __name__ == "__main__":
    main()