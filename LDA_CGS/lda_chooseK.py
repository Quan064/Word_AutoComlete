import pickle
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel


def load_data(address):
    with open(address, 'rb') as f:
        tokenized_articles = pickle.load(f)

    return tokenized_articles

def preprocess(tokenized_articles):
    vocab = corpora.Dictionary(tokenized_articles)
    vocab.filter_extremes(no_below=5, no_above=0.5)
    corpus_bow = [vocab.doc2bow(article) for article in training_tokenized_articles]
    
    return vocab, corpus_bow


if __name__ == "__main__":
    training_tokenized_articles = load_data("Dataset/training_data_for_LDA.pkl")
    validation_tokenized_articles = load_data("Dataset/validation_data_for_LDA.pkl")

    training_vocab, training_corpus_bow = preprocess(training_tokenized_articles)
    validation_vocab, validation_corpus_bow = preprocess(validation_tokenized_articles)

    for K_i in (5, 10, 15, 20, 25, 30, 35, 40):
        lda_model = models.ldamodel.LdaModel(corpus=training_corpus_bow,
                                             id2word=training_vocab,
                                             num_topics=K_i,
                                             passes=10,
                                             alpha='auto',
                                             eta='auto',
                                             random_state=1)

        # Nạp thêm validation_set

        # Tính điểm Coherence
        coherence_model_lda = CoherenceModel(model=lda_model,
                                             texts=training_tokenized_articles,
                                             dictionary=training_vocab,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        print((K_i, coherence_lda))