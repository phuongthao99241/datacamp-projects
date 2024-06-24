import pandas as pd
import re
from langdetect import detect
from underthesea import word_tokenize, pos_tag

file_path = 'jobs-website.csv'
df = pd.read_csv(file_path)

# Function to detect Vietnamese content
def is_vietnamese(text):
    try:
        return detect(text) == 'vi'
    except:
        return False

# Filter only posts in Vietnamese
df = df[df['Job Details'].astype(str).apply(is_vietnamese)]
df = df[df['Job Requirements'].astype(str).apply(is_vietnamese)]

# Filter only posts with available requirements
df = df[~(
    (df['Job Details'].astype(str).isin(['[]', 'nan', '']) & df['Job Requirements'].astype(str).isin(
        ['[]', 'nan', ''])))]

# Remove special fields (missing values)
df['Job Details'] = df['Job Details'].astype(str).apply(lambda x: re.sub(r'[\[\]]', '', x))
df['Job Requirements'] = df['Job Requirements'].astype(str).apply(lambda x: re.sub(r'[\[\]]', '', x))

# Save all requirements into a list
job_requirement = []

for job in df['Job Requirements']:
    all_requirements = re.findall(r"'(.*?)'", job)
    for requirement in all_requirements:
        job_requirement.append(requirement)

indexing = dict()
for index, requirement in enumerate(job_requirement):
    tokens = word_tokenize(requirement)
    pos_tags = pos_tag(requirement)
    indexing[index] = [(token, tag) for token, tag in pos_tags]

data_list = [(key, value[0], value[1]) for key, values in indexing.items() for value in values]
df = pd.DataFrame(data_list, columns=['sentence_ID', 'word', 'pos'])

# Save the DataFrame to a CSV file
df.to_csv("website_dataset.csv", index=False)

