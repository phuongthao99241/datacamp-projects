import pandas as pd


# Function to convert from csv into dataframe
def read_csv_to_dataframe(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


# Resetting the sentence_id to have a continuous sequence after manual removal of non-Vietnamese sentences
def reindexing_for_vi(df):
    sentence_id_map = {id: new_id for new_id, id in enumerate(df['sentence_id'].unique())}
    df['sentence_id'] = df['sentence_id'].map(sentence_id_map)
    return df


# Reclassification of labels (only B-Skill, I-Skill and O left)
def reclassification_for_english(df):
    labels_not_convert = ['B-Skill', 'I-Skill']
    new_label = 'O'

    df['tag'] = df['tag'].apply(lambda tag: new_label if tag not in labels_not_convert else tag)

    # Re-indexing, therefore sentence start from index 0
    df['sentence_id'] = df['sentence_id'] - 1

    return df


# Adjust to have a similar format (column names, sentence_id format) for all datasets
def process_dataset(df):
    data_dict = {}
    for _, row in df.iterrows():
        sentence_id = f"sentence_{row['sentence_id']}"
        if sentence_id not in data_dict:
            data_dict[sentence_id] = {'word': [], 'pos': [], 'tag': []}

        data_dict[sentence_id]['word'].append(row['word'])
        data_dict[sentence_id]['pos'].append(row['pos'])
        data_dict[sentence_id]['tag'].append(row['tag'])

    return data_dict


# Save processed files into csv
def save_to_csv(data_dict, dataset_name):
    sentences = []
    for sentence_id, sentence_data in data_dict.items():
        for word, pos, tag in zip(sentence_data['word'], sentence_data['pos'], sentence_data['tag']):
            sentences.append([sentence_id, word, pos, tag])

    df = pd.DataFrame(sentences, columns=['sentence_id', 'word', 'pos', 'tag'])
    df.to_csv(f'{dataset_name}.csv', index=False, encoding='utf-8')


answers_ENprocessed = reclassification_for_english(read_csv_to_dataframe("df_answers.csv"))
processed_answers_set_en = process_dataset(answers_ENprocessed)
save_to_csv(processed_answers_set_en, 'processed_df_answers')

test_ENprocessed = reclassification_for_english(read_csv_to_dataframe("df_testset.csv"))
processed_test_set_en = process_dataset(test_ENprocessed)
save_to_csv(processed_test_set_en, 'processed_df_testset')

kaggle_data_reindexing = reindexing_for_vi(read_csv_to_dataframe("labelled_kaggle_dataset.csv"))
process_kaggle = process_dataset(kaggle_data_reindexing)
save_to_csv(process_kaggle, 'processed_kaggle_dataset')

website_data_reindexing = reindexing_for_vi(read_csv_to_dataframe("labelled_website_dataset.csv"))
process_website = process_dataset(website_data_reindexing)
save_to_csv(process_website, 'processed_website_dataset')
