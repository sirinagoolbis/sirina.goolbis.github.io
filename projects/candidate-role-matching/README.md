# Candidate-Role Matching Project

This project fine-tunes a BERT model to classify candidate resumes into relevant job roles based on professional summaries and skills. This approach utilizes natural language processing (NLP) to improve recruiter efficiency and support HR analytics.

## Project Structure
- ‘candidate_dataset.csv’: Data said that contains candidate summaries, major, skills, and job roles
- ‘train_candidate_role_classifier.py’: Script that pre-processes data, trains, and evaluates the model
- ‘saved.model/’: Directory that contains trained model and tokenizer
- 'predictions_output.csv': Output file that contains predictions on résumé data


## Dataset Overview
Source: AI-generaterd resumes across different fields
Total samples: 222 resumes
Disciplines (13 majors):
  - Engineering, Biology, Medicine, Psychology, English, Education, Psychology, Computer Science, Business, Design, Marketing, Communication and Mathematics


Each resume contains:
- **Summary**: Candidates professional summary
- **Skills**: Technical and soft skills
- **Discipline**: Target job rule or category (further utilized as label)

## Model Details
- Base model: bert-base-uncased
- Frameworks: HuggingFace Transformers, Pytorch
- Fine-tuning:
  - Train/Test split: 80/20
  - Token classification head added
  - Trained 5 epochs with early stopping
 
## Final Metrics
- Accuracy: 93.33%
- F1 Score (macro): 90.34%

## Features in Progress
- Interactive User Interface with confidence scores across disciplines
- INtegration with Streamlit for real-time uploads and classification


## Requirements

requires Python 3.8+ with following libraries:
- pandas
- torch
- transformers
- numpy
- datasets
- scikit-learn

You can install them after running:

'''bash
# Clone Repository
git clone https://github.com/sirinagoolbis/sirinagoolbis.github.io
cd projects/candidate-role-matching

# Install Dependencies
pip install -r requirements.txt

# Train the Model
python train_candidate_role_classifier.py

# Run predictions
python predict.py --input resume.pdf
