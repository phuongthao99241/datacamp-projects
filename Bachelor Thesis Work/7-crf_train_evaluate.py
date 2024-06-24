# Training process of this thesis is conceptualized from the paper Thomas Green, Diana Maynard, and Chenghua Lin. 2022. Development of a Benchmark Corpus to Support Entity Recognition in Job Descriptions. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 1201–1208, Marseille, France. European Language Resources Association.

import csv

from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report, f1_score
from itertools import product

# Organize the input data into a structured format suitable for feature extraction.
def prepare_data(data):
    sentences = []
    for sentence_id, sentence_data in data.groupby('sentence_id'):
        words = sentence_data['word'].tolist()
        tags = sentence_data['tag'].tolist()
        pos_tags = sentence_data['pos'].tolist()
        sentence = [{"word": word, "tag": tag, "pos": pos} for word, tag, pos in zip(words, tags, pos_tags)]
        sentences.append(sentence)
    return sentences


# Extract features from a word in a sentence for the CRF model.
def word2features(sent, i):
    word = str(sent[i]["word"])
    pos = sent[i]["pos"]

    features = {
        'word.lower()': word.lower(),
        'pos_tag': pos,
    }

    # Contextual feature for previous word
    if i > 0:
        prev_tag = sent[i - 1]["tag"]
        features['-1:tag'] = prev_tag
    else:
        features['BOS'] = True

    # Contextual feature for next word
    if i < len(sent) - 1:
        next_tag = sent[i + 1]["tag"]
        features['+1:tag'] = next_tag
        features['EOS'] = False
    else:
        features['+1:tag'] = 'EOS'
        features['EOS'] = True

    return features

# Generate features for all words in a sentence
def sentence2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

# Train the CRF model with given parameters.
def sentence2labels(sentence):
    return [token["tag"] for token in sentence]

# Train a model
def train_crf_model(X_train, y_train, c1, c2):
    crf_model = CRF(algorithm='lbfgs', c1=c1, c2=c2, max_iterations=100, all_possible_transitions=True)
    crf_model.fit(X_train, y_train)
    return crf_model

# Fine-tune the CRF model by testing various parameter combinations.
def fine_tune_crf(X_train, y_train, X_dev, y_dev, param_space):
    best_f1_score = 0
    best_params = None

    for params in product(*param_space.values()):
        c1, c2 = params
        print(f"Training with c1={c1}, c2={c2}")
        crf_model = train_crf_model(X_train, y_train, c1, c2)
        y_pred_dev = crf_model.predict(X_dev)
        weighted_f1 = f1_score(y_dev, y_pred_dev)

        print(f"F1 Score = {weighted_f1}")
        if weighted_f1 > best_f1_score:
            best_f1_score = weighted_f1
            best_params = params

    return best_params

# evaluate the CRF model 
def evaluate_crf_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    return report




