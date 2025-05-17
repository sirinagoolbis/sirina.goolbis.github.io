# Candidate-Role Matching Project

This project uses a BERT model to organize candidate's resumes into their respective roles or disciplines given their provided skill set and summary. This models natural language processing (NLP) for role matching which is helpful when recruiting and in HR analytics.

## Project Structure
- 'candidate_dataset.csv': dataset presenting candidate summaries, their skills and respective roles
- 'train_candidate_role_classifier.py': main script that is used to pre-process data, train and evaluate the model.
- 'saved_model/': folder that contains trained model and tokenizer
- 'preditions_output.csv' - output file that showcases model predictions of new data (generated after running prediction)

## Dataset
Data is further divided by candidate information into columns that contain:
- **Summary** - professional summary of candidate
- **Skills** - pertinent skills obtained by candidate
- **Disciplines** - roll or a job category (label)

## Requirements

requires Python 3.8+ with following packages:
- pandas
- torch
- transformers
- numpy
- datasets
- scikit-learn

can be installed using:

'''bash
pip install -r requirements.txt
