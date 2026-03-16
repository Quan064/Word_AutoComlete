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

    K = 170
    eta = 0.01

    model = tp.LDAModel(
        k=K,
        eta=eta,
        seed=1
    )

    # bật tự tối ưu alpha
    model.optim_interval = 10

    for doc in training_tokenized_articles:
        model.add_doc(doc)

    model.burn_in = 200

    for i in range(0, 1000, 50):
        model.train(50)
        print(f"Iteration {i+50}, alpha={model.alpha}")

    model.save("LDA_CGS/lda_cgs.bin")

    # lấy topic words
    topics = []
    for k in range(model.k):
        topic_words = [w for w, _ in model.get_topic_words(k, top_n=10)]
        topics.append(topic_words)

    coherence_model = CoherenceModel(
        topics=topics,
        texts=validation_tokenized_articles,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence = coherence_model.get_coherence()

    print("Final alpha:", model.alpha)
    print("Coherence:", coherence)

    # Coherence: 0.6161459815152538
    # Final alpha: [0.02709087 0.02101906 0.01855711 0.01904447 0.01226971 0.02022971
    #               0.02489826 0.01689207 0.0155392  0.02601165 0.02180631 0.01776157
    #               0.02384997 0.02211962 0.01641406 0.01931294 0.04131604 0.01359172
    #               0.03069537 0.01273382 0.01533076 0.1199773  0.00923671 0.01056572
    #               0.0200469  0.02154734 0.13670921 0.04147037 0.1574055  0.05216312
    #               0.01035323 0.0192283  0.01945508 0.0776346  0.02339698 0.0094066
    #               0.01241123 0.03597363 0.02823899 0.01159855 0.01350672 0.05814337
    #               0.01228543 0.02402422 0.01773305 0.01413022 0.01157957 0.01648378
    #               0.01739959 0.00827228 0.00912959 0.00928082 0.02350069 0.03511993
    #               0.01036462 0.07028669 0.01337986 0.0126803  0.01543318 0.01719098
    #               0.02134889 0.0065855  0.00942363 0.02517121 0.1047029  0.02408835
    #               0.01129103 0.0317177  0.02829964 0.01036895 0.01042371 0.02502959
    #               0.02458922 0.00926188 0.00618787 0.00840643 0.01075259 0.01173094
    #               0.02435048 0.01496991 0.02418457 0.00968084 0.03869837 0.03133348
    #               0.02700372 0.02409397 0.02963332 0.02625977 0.01368212 0.03156866
    #               0.01034628 0.01734024 0.01844684 0.01446735 0.03183814 0.0703919
    #               0.01248415 0.01441345 0.06079924 0.01548031 0.01623757 0.0156048
    #               0.07362241 0.01997841 0.01253803 0.01162936 0.01518646 0.04567403
    #               0.02048637 0.03001245 0.01930867 0.0063408  0.01646221 0.01690326
    #               0.05075462 0.02710568 0.00793526 0.01482303 0.01247226 0.02576797
    #               0.03535765 0.02229957 0.00732054 0.02418048 0.00925715 0.03849156
    #               0.02374175 0.01936023 0.037684   0.0875274  0.12724839 0.04116107
    #               0.00518303 0.01698706 0.01274324 0.0342939  0.01340108 0.0301996
    #               0.0117425  0.01732211 0.00729903 0.0118022  0.08475175 0.08249705
    #               0.02625887 0.0095536  0.01330285 0.02372324 0.01031109 0.03583189
    #               0.14314745 0.00854384 0.01544333 0.05281809 0.00835016 0.03424716
    #               0.01367676 0.01837215 0.0331282  0.00977272 0.08869661 0.0283436
    #               0.03194427 0.02126612 0.01619058 0.03383759 0.01344125 0.02227864
    #               0.01020642 0.01354525]