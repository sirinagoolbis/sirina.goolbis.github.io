# Behavioral NLP with BERT
The project fine tunes a BERT model for behavioral NLP classification. It utilized data from job applicants with respective skills, majors, and study focusesf further separating deiscplines based on textual data like summary and skills.

## Features
- preprocessing, cleaning and data loading from CSV file
- model training and evaluation via Hugging Face Trainer API
- tokenization via Hugging Face's BERT tokenizer
- Predictive analysis for CSV input and next text uploads
- accuracy metrics analyzed and further edited for evaluation

## Requirements
- Python 3.8+
- transformers
- pandas
- scikit-learn
- numpy
- dataset
- torch

## Usage
1. prepare dataset CSV ('candidates_dataset.csv') with columns: Summary, Skills, Discipline
2. Run training script

## inference
3. For futher preedictions on sample text, edit the 'test_samples' in the script or use the 'predict_from_file' function

## Output
- trained model is saved under './saved_model'
- predictions stoed in 'predictions_ouput.csv' when completing prediction on CSV
