# Training process of this thesis is conceptualized from the paper Thomas Green, Diana Maynard, and Chenghua Lin. 2022. Development of a Benchmark Corpus to Support Entity Recognition in Job Descriptions. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 1201–1208, Marseille, France. European Language Resources Association.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib.util


spec = importlib.util.spec_from_file_location("crf_train_evaluate", "7-crf_train_evaluate.py")
crf_train_evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crf_train_evaluate)


# Load data
temp_data = pd.read_csv('processed_df_answers.csv', encoding='utf-8')
test_data = pd.read_csv('processed_df_testset.csv', encoding='utf-8')

# Splitting dataset
sentence_ids = temp_data['sentence_id'].unique()
train_sentence_ids, dev_sentence_ids = train_test_split(sentence_ids, test_size=len(test_data['sentence_id'].unique()), random_state=42)

train_data = temp_data[temp_data['sentence_id'].isin(train_sentence_ids)]
dev_data = temp_data[temp_data['sentence_id'].isin(dev_sentence_ids)]

train_sentences = crf_train_evaluate.prepare_data(train_data)
dev_sentences = crf_train_evaluate.prepare_data(dev_data)
test_sentences = crf_train_evaluate.prepare_data(test_data)

# Convert to features for each set. 
X_train = [crf_train_evaluate.sentence2features(sentence) for sentence in train_sentences]
y_train = [crf_train_evaluate.sentence2labels(sentence) for sentence in train_sentences]

X_test = [crf_train_evaluate.sentence2features(sentence) for sentence in test_sentences]
y_test = [crf_train_evaluate.sentence2labels(sentence) for sentence in test_sentences]

X_dev = [crf_train_evaluate.sentence2features(sentence) for sentence in dev_sentences]
y_dev = [crf_train_evaluate.sentence2labels(sentence) for sentence in dev_sentences]

# Define hyperparameter space
params_space = {
    'c1': np.arange(0, 1.0, 0.1),  # Values from 0 to 1 in steps of 0.1
    'c2': np.arange(0, 1.0, 0.1)  # Values from 0 to 1 in steps of 0.1
}

# Fine-tune the CRF model
best_params = crf_train_evaluate.fine_tune_crf(X_train, y_train, X_dev, y_dev, params_space)
print("Best Hyperparameters:", best_params)


# Train the final CRF model with best hyperparameters
final_crf_model = crf_train_evaluate.train_crf_model(X_train, y_train, best_params[0], best_params[1])

# Evaluate the final model
print("Evaluation on Development Set:")
report_dev = crf_train_evaluate.evaluate_crf_model(final_crf_model, X_dev, y_dev)
print(report_dev)

print("Evaluation on Test Set:")
report_test = crf_train_evaluate.evaluate_crf_model(final_crf_model, X_test, y_test)
print(report_test)


