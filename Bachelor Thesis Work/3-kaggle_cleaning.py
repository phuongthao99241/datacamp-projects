import pandas as pd
from bs4 import BeautifulSoup
import re
import random
from langdetect import detect
from underthesea import word_tokenize, pos_tag

# Read data from csv file
file_path = 'vietnamese-job-posting.csv'
df = pd.read_csv(file_path)
job_sentences = []

# Clean job requirements because they are still in HTML formatting
for html in df['job_requirements']:
    # Check if the value is a string
    if isinstance(html, str):
        soup = BeautifulSoup(html, 'html.parser')
        # Find the specific <p> elements containing the desired content
        desired_content = soup.find_all('p')
        # Extract the text content from the selected <p> elements
        if desired_content is not None:
            extracted_content = [p.get_text() for p in desired_content]
            # Join the extracted content into a single string
            result = "\n".join(extracted_content)

            for text in result.split('\n'):
                # Use regular expression to split the text into a list of sentences
                sentences = re.split(r'(?<!\d)-', text)

                # Remove empty strings and leading/trailing whitespace
                sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                for sentence in sentences:
                    if len(sentence) != 0:
                        job_sentences.append(sentence)

# Sample randomly 1200 job sentences from job requirements
random.seed(42)
random_sentences = random.sample(job_sentences, 1200)

def is_vietnamese(text):
    try:
        return detect(text) == 'vi'
    except:
        return False

# Cleaning
for sentence in random_sentences:
    if not is_vietnamese(sentence):
        random_sentences.remove(sentence)

indexing = dict()
for index, requirement in enumerate(random_sentences):
    tokens = word_tokenize(requirement)
    pos_tags = pos_tag(requirement)  #
    indexing[index] = [(token, tag) for token, tag in pos_tags]


data_list = [(key, value[0], value[1]) for key, values in indexing.items() for value in values]
df = pd.DataFrame(data_list, columns=['sentence_id', 'word', 'pos'])
# Save the DataFrame to a CSV file
df.to_csv("kaggle_dataset.csv", index=False)
