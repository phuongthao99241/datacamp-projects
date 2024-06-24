import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter


# Analysis on dataset to have annotated corpus statistics (For Table 3.1)
def analyze_dataset(dataset_name):
    data = pd.read_csv(f"{dataset_name}.csv", encoding='utf-8')

    # Calculate overall dataset statistics
    total_sentences = data['sentence_id'].nunique()
    total_tokens = len(data)
    avg_tokens = total_tokens / total_sentences

    # Count the number of job postings (sentences) that contain a skill
    sent_with_skill = data[data['tag'].isin(['B-Skill', 'I-Skill'])]
    job_sent_with_skills_count = len(sent_with_skill['sentence_id'].unique())
    job_sent_with_skills_freq = job_sent_with_skills_count / total_sentences

    # Create a structured table using pandas
    analysis_df = pd.DataFrame({
        "Criteria": ["Total Sentences", "Total Tokens", "Average Tokens per Sentence",
                     "Frequency of Job Sentences containing Skills"],
        "Value": [total_sentences, total_tokens, f"{avg_tokens:.2f}", f"{job_sent_with_skills_freq:.2%}"]
    })

    # Calculate label distribution (Table 3.2)
    label_distribution_all = data['tag'].value_counts(normalize=True)
    label_df = pd.DataFrame(label_distribution_all.items(), columns=["Label", "Proportion"])
    label_df["Proportion"] = label_df["Proportion"].apply(lambda x: f"{x * 100:.2f}%")

    # Display all analyzed data
    print(f"Data Analysis on Dataset {dataset_name}")
    print(analysis_df)
    print(f"\nLabel Distribution of Dataset {dataset_name}:")
    print(label_df)
    print("\n")


# Top 5 most frequent POS tags aligned with skil-related tokens. (Figure 3.1, 3.2)
# Tokens with label B-Skill and I-Skill are considered as skill-related tokens.
def pos_tag_distribution(dataset_name, graph_name):
    data = pd.read_csv(f"{dataset_name}.csv", encoding='utf-8')
    skill_words = data[data['tag'] != 'O']
    skill_pos_counts = skill_words['pos'].value_counts()
    pos_proportions = skill_pos_counts / skill_pos_counts.sum()
    top5_pos = pos_proportions.head(5)
    pos_comparison = pd.DataFrame({'Skill Words': top5_pos})
    pos_comparison.plot(kind='bar', figsize=(6, 3))
    plt.title(f'Top 5 POS Tags for Skill Words in {graph_name}')
    plt.xlabel('POS Tags')
    plt.ylabel('Proportion')
    plt.show()


# Top 10 most common skills occuring in datasets (Table 3.3, 3.4)
"""Logic: 
- If a word is tagged as 'B-Skill', it starts a new skill phrase.
- If a word is tagged as 'I-Skill', it is added to the current skill phrase.
- If a word is neither 'B-Skill' nor 'I-Skill', it marks the end of the current skill phrase"""
def most_common_skills(dataset_name):
    data = pd.read_csv(f"{dataset_name}.csv", encoding='utf-8')
    skill_phrases = []
    current_phrase = []

    for _, row in data.iterrows():
        word, tag = row['word'], row['tag']
        if tag == 'B-Skill':
            if current_phrase:
                skill_phrases.append(' '.join(current_phrase))
            current_phrase = [word]
        elif tag == 'I-Skill':
            current_phrase.append(word)
        else:
            if current_phrase:
                skill_phrases.append(' '.join(current_phrase))
                current_phrase = []

    if current_phrase:
        skill_phrases.append(' '.join(current_phrase))

    phrase_counts = Counter(skill_phrases)

    most_common_phrases = phrase_counts.most_common(10)
    return most_common_phrases


# Analyzing 'processed_kaggle_dataset'
print("Analyzing 'processed_kaggle_dataset': ")
analyze_dataset('processed_kaggle_dataset')
print("Displaying POS tag distribution for 'processed_kaggle_dataset'")
pos_tag_distribution('processed_kaggle_dataset', 'Kaggle Dataset')
print("Identifying most common skills in 'processed_kaggle_dataset'")
most_common_skills('processed_kaggle_dataset')

# Analyzing 'processed_website_dataset'
print("\nAnalyzing 'processed_website_dataset': ")
analyze_dataset('processed_website_dataset')
print("Displaying POS tag distribution for 'processed_website_dataset'")
pos_tag_distribution('processed_website_dataset', 'Website Dataset')
print("Identifying most common skills in 'processed_website_dataset'")
most_common_skills('processed_website_dataset')

# Analyzing 'processed_df_answers'
print("\nAnalyzing 'processed_df_answers'")
analyze_dataset('processed_df_answers')
print("Identifying most common skills in 'processed_df_answers'")
most_common_skills('processed_df_answers')

# Analyzing 'processed_df_testset'
print("\nAnalyzing 'processed_df_testset'")
analyze_dataset('processed_df_testset')
