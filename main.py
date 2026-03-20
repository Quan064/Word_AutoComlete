import pickle
from pathlib import Path
import tomotopy as tp
import spacy
import numpy as np
from Trie.trie import Trie, TrieNode


# ================== LOAD ==================

LEMMA_CACHE_PATH = Path("Trie/word_lemma.pkl")
context_analysis_cache = {}
trie_candidates_cache = {}


def build_word_lemma_map(trie, nlp):
    words = []
    stack = [(trie.root, "")]

    while stack:
        node, prefix = stack.pop()
        if node.is_end:
            words.append(prefix)

        for char, child in node.child.items():
            stack.append((child, prefix + char))

    word_lemma_map = {}
    for doc, word in zip(nlp.pipe(words, batch_size=512), words):
        if len(doc) == 0:
            word_lemma_map[word] = word.lower()
            continue

        token = doc[0]
        word_lemma_map[word] = token.lemma_ if token.is_alpha else word.lower()

    with open("Trie/word_lemma.pkl", "wb") as f:
        pickle.dump(word_lemma_map, f)

    return word_lemma_map

def load_models():
    lda_model = tp.LDAModel.load("LDA_CGS/lda_cgs.bin")

    word_to_id = {
        lda_model.used_vocabs[i]: i
        for i in range(lda_model.num_vocabs)
    }

    with open("Trie/Trie.pkl", 'rb') as f:
        trie = pickle.load(f)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    if LEMMA_CACHE_PATH.exists():
        with open(LEMMA_CACHE_PATH, "rb") as f:
            word_lemma_map = pickle.load(f)
    else:
        word_lemma_map = build_word_lemma_map(trie, nlp)

    topic_word_matrix = np.array([
        lda_model.get_topic_word_dist(k)
        for k in range(lda_model.k)
    ])

    return lda_model, trie, word_to_id, nlp, topic_word_matrix, word_lemma_map


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


def get_context_analysis_cached(context, nlp, lda_model):
    if context in context_analysis_cache:
        return context_analysis_cache[context]

    context_tokens, token_count, non_token_count = analyze_context(context, nlp)
    context_topic_dist = infer_topic_distribution(lda_model, context_tokens)

    cached_value = (
        context_tokens,
        token_count,
        non_token_count,
        context_topic_dist,
    )
    context_analysis_cache[context] = cached_value
    return cached_value


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


def get_trie_candidates_cached(trie, prefix, limit):
    cache_key = (prefix, limit)
    if cache_key in trie_candidates_cache:
        return trie_candidates_cache[cache_key]

    candidates = trie.topK(prefix, limit)
    trie_candidates_cache[cache_key] = candidates
    return candidates


def clear_runtime_caches():
    context_analysis_cache.clear()
    trie_candidates_cache.clear()


# ================== MAIN ==================

def suggest_words(
    lda_model,
    trie,
    word_to_id,
    nlp,
    topic_word_matrix,
    word_lemma_map,
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
    (
        context_tokens,
        token_count,
        non_token_count,
        context_topic_dist,
    ) = get_context_analysis_cached(context, nlp, lda_model)

    # ===== TRIE CANDIDATES =====
    candidates = get_trie_candidates_cached(trie, prefix, num_suggestions * 20)
    if not candidates:
        return []

    # ===== PREPARE CANDIDATES =====
    # Keep raw trie candidates for freq-based ranking.
    freq_candidates = candidates  # (word, freq)

    # LDA candidate subset requires vocab mapping via lemma.
    lda_candidates = []  # (word, freq, word_id)
    for word, freq in candidates:
        lemma = word_lemma_map.get(word, word.lower())
        if lemma not in word_to_id:
            continue
        word_id = word_to_id[lemma]
        lda_candidates.append((word, freq, word_id))

    if not lda_candidates:
        # no LDA candidates possible, fallback to top freq from trie
        return [(w, 0.0) for w, _ in freq_candidates[:num_suggestions]]

    # ===== VECTORIZE =====
    final_words = [word for word, _, _ in lda_candidates]
    word_ids = np.array([word_id for _, _, word_id in lda_candidates])

    word_topic_matrix = topic_word_matrix[:, word_ids]

    # ===== LDA SCORE =====
    top_k = 3
    top_idx = np.argsort(context_topic_dist)[-top_k:]

    context_top = context_topic_dist[top_idx]
    word_top = word_topic_matrix[top_idx, :]

    score_lda = context_top @ word_top

    # ===== FREQ SCORE =====
    score_trie = np.array([freq for _, freq in freq_candidates])

    # ===== TÍNH SLOT (CORE IDEA) =====
    semantic_ratio = token_count / (token_count + non_token_count + 1e-9)

    entropy = topic_entropy(context_topic_dist)
    entropy_norm = entropy / np.log(len(context_topic_dist))

    lda_weight = semantic_ratio * (1 - entropy_norm)
    lda_weight = np.clip(lda_weight, 0.0, 1.0)

    num_lda = int(num_suggestions * lda_weight)
    num_freq = num_suggestions - num_lda

    # ===== RANK LDA =====
    lda_idx = np.argsort(score_lda)[::-1]
    lda_top = [final_words[i] for i in lda_idx[:num_lda]]
    num_lda = min(len(lda_top), num_lda)
    num_freq = num_suggestions - num_lda

    if verbose:
        print(f"semantic_ratio={semantic_ratio:.3f}, entropy={entropy_norm:.3f}")
        print(f"→ LDA={num_lda}, FREQ={num_freq}")

    # ===== RANK FREQ =====
    # Take directly from original trie candidates, excluding LDA picks.
    freq_idx = np.argsort(score_trie)[::-1]

    freq_top = []
    for i in freq_idx:
        w = freq_candidates[i][0]
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

def interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix, word_lemma_map):
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
            word_lemma_map,
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
    print("\n" + "="*50)
    print("  WORD AUTOCOMPLETE SYSTEM")
    print("="*50 + "\n")
    
    print("⏳ Loading models...")
    lda_model, trie, word_to_id, nlp, topic_word_matrix, word_lemma_map = load_models()
    print("✓ Models loaded successfully\n")

    print(f"📊 LDA Topics: {lda_model.k}")
    print(f"📚 Vocabulary Size: {lda_model.num_vocabs}\n")

    interactive_mode(lda_model, trie, word_to_id, nlp, topic_word_matrix, word_lemma_map)
    
    print("\n" + "="*50)
    print("  Goodbye!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
