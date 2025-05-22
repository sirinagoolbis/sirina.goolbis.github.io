# LLM Powered Student Engagement Summarizer

This project fine-tunes a RoBERTa model to showcase engagement-focused summaries from student to student classroom conversations in a college lecture. The model enables automatic tagging of social and cognitive behaviors, with applications for real-time feedback, learning analytics, and intelligent tutoring systems.

Problem Statement: 
Educators and ed-tech platforms post COVID pandemic are struggling to keep students engaged resulting in attention span and class participationg to decrease significantly. This project aims to measure student engagement across differe lecture modalities (in-person, asynchronous, etc.) futher providing a scalable measyre to identify engagement types (i.e. attentive, contributing, distracted) given raw dialogue transcripts using LLMs.

Solution:
By fine-tuning a transformer-based LLM (RoBERTa) on labeled conversational data, the system learns to classify and summarize student behavior, generating outputs that can aid professors in changing lecture format. Additionally supporting learning analytics, tutoring systems and classroom monitoring tools.

## Project Structure
- 'student_conversation_CSV': raw student-professor conversation data taken from lectures
- 'classroom_conversations_labeled.csv': labeled summaries used for training
- 'lllm_finetuning_script.py' - script for finetuning LLM (i.e. GPT)
- 'README.md' - project overview & instructions for recreation
- 'models/saved_model/: fine-tuned model checkpoint

## Dataset
- **source**: AI generated transcript of spoken classroom conversations between students during a lecture
- **size**: ~56,625 dialogues
- **label**:
    - Attentive (11,470): Student focused and attentive to lecture.
    - Contributing (11,412): Student is asking or answering questions to add meaningful discussion to dialogue.
    - Neutral (11,095): Student models passive participation; no definitive sign of engagement or lack thereof.
    - Collaborating (11,248): Student is engaged in lecture with classmates through multi-person dicussion and group activity.
    - Non-attentive (11,400): Student is distracted, not engaged or asking unrelated questions.

The goal is to train a model to automatically determine whether discussion belong to any of the five categories.

## Model & Method
- Architecture: RoBERTa-base (via Hugging Face Transformers)
- Training Script: llm_finetuning_script.py
- Libraries: transformers, datasets, torch, and pandas
- Task: text summarization using sequence-to-sequence learning
- Training Device: MPS (Apple Silicon GPU acceleration)

## Training Outcomes
- Train Loss (final): 0.0000
- Eval Accuracy: 1.0000
- F1 Precision: 1.0000

  Note: These perfect scores are likely due to the regularity of the dataset. Initial accuracy was ~34.7%, indicating the complexity of the task and importance of correct preprocessing and adjusting paramaters. By the end of the 5+ iterations of the model, performance improved through:
  - Correct label formatting
  - Resolving class imbalance through stratified sampling
  - Early stopping and monitoring overfitting
  - Adjusting the batch size, schedular and learning rate
 
## Limitations and Future Work
- Current evaluation was completed through a structured validation split from the data (additional cross, validation and testing on unseen lectures is further needed to assess generalization)
- Data set is AI generated, therefore may not capture ambiguity, real-world noises, and multimodal behavior (tone, gestures)
- Future extensions include:
      - Expansion through cross-insitutional dataset
      - Integration of visual attention signals
      - Summary of lecture generation

## Requirements

requires Python 3.8+ with libraries that include:
- pandas
- datasets
- torch
- transformers

Install with:
pip install -r requirements.txt

# How to run
git clone https://github.com/your-repo/llm-candidate-summarizer.git
cd llm-candidate-summarizer

python3 lllm_finetuning_script.py


