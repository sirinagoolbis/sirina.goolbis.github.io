# Behavioral NLP with BERT

## Overview:
This project fine-tunes a BERT model to classify job applicants into disciplines through natural language processing. It utilized structured and unstructured resume data (i.e. summary, skills, majors) and outputs predictions to better align positions with applicants.

## Projext Structure
- ‘scripts/train.py’: BERT fine-tuning code using HuggingFace Transformers
‘scripts/predict.py’: script for inferences on resumes
‘models/’: saved model checkpoints
‘app.py’: (in progress) streamline interface for real-time classification

## Data set overview
- **Source**: AI-generated data set with varied disciplines, summaries, skills and majors
- **Size**: 222 labeled resumes spanning multiple disciplines such as:
- 'Healthcare'
- 'Marketing'
- 'Cybersecurity'
- 'Product Management'
- 'Engineering'
- and more

Each resume included:
- Summary:brief overview of applicant
- Skills: technical and soft skills
- Discpline: target role or domain (additionally used as a label)

## Model Details
- **Base Model**: 'bert-base-uncased'
- **frameworks**: Pytorch, HuggingFace Transformers
- **Fine-tuning Strategy**:
    - 80/20 train-tes split
    - token classfication head used
    - trained 4 epochs with early stopping
- **evaluation**:
    - Accuracy: '87.4%'
    - F1-score (macro average): '0.86'
    - confusion matric located in 'notebooks/evaluation.ipynb'
 
## Web interface (in works)

Integration with **Streamlit** to allow:
- real-time discpline prediction and classification
- drag-and-drop resume uploads
- visual confidence bars for each category

## How to run 
'''bash
git clone https://github.com/sirinagoolbis/sirinagoolbis.github.io
cd projects/bert-np
pip install -r requirements.txt

# train model
python scripts/train.py

# predict
python scripts/predict.py --input resume.pdf
