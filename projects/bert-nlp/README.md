# Behavioral NLP with BERT

## Overview:
This project fine-tunes a BERT model to classify job applicants into disciplines through natural language processing. It utilized structured and unstructured resume data (i.e. summary, skills, majors) and outputs predictions to better align positions with applicants.

## Problem Statement
Recruiters face many obstacles when categorizing applicant’s resumes efficiently thus leading to potential mismatches, increased hiring time and extraneous back-end work. Through automation of this process, it can effectively reduce processing time further allowing work efforts and energy to be reallocated elsewhere. But most importantly, streamlining the recruitment process.

## Features
- Processes and cleans data for optimal performance
- Utilizes Hugging Face Transformers library for fine-tuning of the model
- Uses metrics like F1-score and confusion matrix to evaluate accuracy

## Installation
git clone https://github.com/sirinagoolbis/behavioral-nlp-bert.git
cd behavioral-nlp-bert
pip install -r requirements.txt

## Usage
python classify_resume.py --input resume.txt

## Results
- F1-score of 0.85 on validation set
- Reduction of manual classification time by ~60%

## Future Integrations
- Expand model to accommodate multilingual applications 
- Integrate web interface for real-time classification
