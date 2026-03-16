import pickle
import tomotopy as tp
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def load_data(address):
    with open(address, 'rb') as f:
        tokenized_articles = pickle.load(f)
    return tokenized_articles


if __name__ == "__main__":

    training_tokenized_articles = load_data("Dataset/training_data_for_LDA.pkl")
    validation_tokenized_articles = load_data("Dataset/validation_data_for_LDA.pkl")

    # dictionary cho coherence
    dictionary = Dictionary(training_tokenized_articles)

    K_values = [40, 80, 120, 140, 150, 160, 170, 180, 200, 240, 280, 320, 360, 400]
    coherence_values = []

    for K_i in K_values:

        # Train LDA bằng CGS
        model = tp.LDAModel(
            k=K_i,
            alpha=0.1,
            eta=0.01,
            seed=1
        )

        # thêm tài liệu train
        for doc in training_tokenized_articles:
            model.add_doc(doc)

        # train
        for i in range(0, 200, 10):
            model.train(10)

        # lấy top words cho từng topic
        topics = []
        for k in range(model.k):
            topic_words = [w for w, _ in model.get_topic_words(k, top_n=10)]
            topics.append(topic_words)

        # tính coherence trên validation
        coherence_model = CoherenceModel(
            topics=topics,
            texts=validation_tokenized_articles,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence = coherence_model.get_coherence()
        coherence_values.append(coherence)

        print((K_i, coherence))

    # Vẽ đồ thị
    plt.figure(figsize=(8,5))
    plt.plot(K_values, coherence_values, marker='o')
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Coherence Score")
    plt.title("Topic Coherence vs Number of Topics")
    plt.grid(True)
    plt.show()