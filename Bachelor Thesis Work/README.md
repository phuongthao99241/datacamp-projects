# BSc-Arbeit, Thi Phuong Thao Nguyen, 12107220



## Topic: Development of a Comprehensive Vietnamese Job Posting Corpus for Skill Extraction
This thesis focuses on the task skill extraction. To make it easy for readers to understand the structure of programming code, here's a detailed information about how data was obtained
### Original file: 
- Vietnamese dataset: 
  - vietnamese-job-posting.csv: original file from Kaggle
  - jobs-website.csv: crawled data from website
- English dataset: 
  - df_answers.csv
  - df_testset.csv
### Data Cleaning: 
- 2-website_cleaning.py => website_dataset.csv
- 3-kaggle_cleaning.py => kaggle_dataset.csv

These datasets will be then brought to Google Sheet for manual annotation step. 

### Data Annotation
After being manually annotated, the labeled datasets are saved into 2 csv-files, which are: 
- labelled_kaggle_dataset.csv
- labelled_website_dataset.csv

### Data Processing after Annotation 
The labeled data will be processed to have consistent format before training. 
- processed_kaggle_dataset.csv
- labelled_website_dataset.csv

Besides, English datasets are also reclassified. 
- processed_df_answers.csv
- processed_df_testset.csv

These processed data will be applied for experiments with different models in next steps. 