# LLM Powered Student Engagement Summarizer

This project fine-tunes a **RoBERTa** transformer model to classify and summarize student engagement from peer-to-peer classroom conversations. It enables automatic tagging of **cognitive** and **social behaviors** (i.e. attentive, contributing, distracted), making it useful for:

- Learning analytics
- Intelligent tutoring and monitoring systems
- Real-time feedback to educators

## Problem Statement: 

Post pandemic environments for academia has seen a drastic decline in student contribution and engagement in classrooms due to remote and hybrid learning. This project aims to build a scalable NLP-based tool to:

- **Support Instructors** in tailoring lecture modalities
- **Detect engagement types** from raw conversation transcripts
- **Drive adaptive education** through LLM-powered insights

## Solution:

By fine-tuning a **RoBERTa-base model** on labeled conversation data, the system learns to classify dialogue into engagement types. Output summaries influences real-time feedback, course structure optimization, and addition academic interventions

## Project Structure
llm-student-engagement/
├── student_conversation_CSV/ # Raw classroom conversation transcripts
├── classroom_conversations_labeled.csv # Labeled training dataset
├── lllm_finetuning_script.py # Fine-tuning script
├── models/saved_model/ # Trained model checkpoints
├── requirements.txt # Dependencies
└── README.md # Project documentation

## Dataset

- **Source**: AI generated transcript of spoken classroom conversations between students during a university lecture
- **Size**: ~56,625 sample transcripts
- **Label**:
    - Attentive (11,470): Students are focused and attentive to lecture.
    - Contributing (11,412): Students are asking or answering questions to add meaningful discussion to dialogue.
    - Neutral (11,095): Students model passive participation; no definitive sign of engagement or lack thereof.
    - Collaborating (11,248): Students are engaged in lectures with classmates through multi-person dicussion and group activity.
    - Non-attentive (11,400): Students are distracted, not engaged or asking unrelated questions.

The goal is to train a model to automatically determine whether discussion belongs to any of the five categories.

## Model & Method

- **Architecture**: 'RoBERTa-base' (via Hugging Face Transformers)
- **Task**: Multi-class sequence classification
- **Training Setup**:
  - Devices: MPS (Apple Silicon GPU acceleration)
  - Libraries: 'transformers', 'datasets', 'torch', 'pandas'
  - Strategy:
    - 80/20 train-validation split
    - Stratified sampling to fix class imbalance
    - Early stopping & learning rate tuning

## Results

- **Final Training Loss**: '0.0000'
- **Validation Accuracy**: '1.0000'
- **F1 Precision (Macro)**: '1.0000'

  Note: These perfect scores are likely due to the regularity of the dataset. Initial accuracy was ~34.7%, indicating the complexity of the task and importance of correct preprocessing and adjusting parameters.
 
## Limitations and Future Work

- Dataset is **synthetic**, which lacks real-world ambiguity and non-verbal cues (gestures, tone, personalities)
- **Generalization** still needs testing across unseen lectures and universities

### Planned Enhancements

- Generative summarization of entire lecture using LLMs
- Integration of visual attention signals (gaze tracking, facial cues)
- Real-world multi-institutional dataset integration

## Requirements

- Python 3.8+
- Libraries:
    - 'pandas'
    - 'datasets'
    - 'torch'
    - 'transformers'

Install with:

'''bash
pip install -r requirements.txt

# How to run

git clone https://github.com/your-repo/llm-student-engagement.git
cd llm-student-engagement
python3 lllm_finetuning_script.py

## Real World Applications
- AI co-pilot integration on online learning platform
- Classroom behavior dashboards
- Adaptive learning modules specific to engagement styles


