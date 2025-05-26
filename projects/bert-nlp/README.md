# Behavioral NLP with BERT - Resume Role Classifier

## Overview:
This project fine-tunes a BERT model to classify job applicants into professional disciplines using natural language processing. It leverages structured and unstructured resume fields-such as summaries, skills, and educational backgrounds-to further identify target roles that are ideal matches for job categories.

## Project Structure
- ‘scripts/train.py’: BERT fine-tuning code using HuggingFace Transformers
- ‘scripts/predict.py’: script for inferences on resumes
- ‘models/’: saved model checkpoints
- ‘app.py’: (in progress) streamlit interface for real-time classification

## Dataset overview
- **Source**: AI-generated dataset stimulating real-world resumes
- **Size**: 222 labeled resumes spanning multiple disciplines such as:
      - 'Healthcare'
      - 'Marketing'
      - 'Cybersecurity'
      - 'Product Management'
      - 'Engineering'
      - and more

**Each resume included**:
- 'Summary': brief overview of applicant
- 'Skills': technical and soft skills
- 'Discipline': target role or domain (used as a label)

## Model Details
- **Base Model**: 'bert-base-uncased'
- **Libraries**: Pytorch, Hugging Face Transformers
- **Fine-tuning Strategy**:
    - 80/20 train-test split
    - Tokenization via 'AutoTokenizer'
    - Classification head for multi-class prediction
    - Trained 4 epochs with early stopping
### Evaluation**:
    - **Accuracy**: '87.4%'
    - **F1-score (Macro)**: '0.86'
    - confusion matrix and error analysis located in 'notebooks/evaluation.ipynb'
 
## Web interface (in works)

Streamlit app for:
- real-time role classification with confidence scores
- drag-and-drop resume uploads
- visual explanation of label probabilities

## Tools & Technologies
- Python, Jupyter Notebooks
- PyTorch, Hugging Face Transformers
- Streamlit
- pandas, numpy, scikit-learn
- matplotlib, seaborn
  
## How to run 
"'''bash"
git clone https://github.com/sirinagoolbis/sirinagoolbis.github.io
cd projects/bert-nlp
pip install -r requirements.txt

# Train model
python scripts/train.py

# Predict
python scripts/predict.py --input resume.pdf

Future Improvements
- Incorporate resume parsing from raw PDF/DOCX files
- Extend label set with sub-domains (i.e. Data Science vs Data Engineering)
- Deploy the Streamlit app on Hugging Face Spaces or Streamlit Cloud
- Incorporate explainability tools (i.e. SHAP or LIME)
