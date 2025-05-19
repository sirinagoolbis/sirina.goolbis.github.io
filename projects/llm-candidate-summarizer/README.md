# LLM Candidate Summarizer

This project fine-tunes a large language model (LLM) to showcase engagement-focused summaries from student to student classroom conversations in a college lecture. It applies NLP to model expected educational behaviors, creating automated feedback, behavioral analysis, and AI-assisted lecture summarization.

Problem Statement: 
Educators and ed-tech platforms post COVID pandemic are struggling to keep students engaged and attention spans/ class participation is dwindling. This project aims to measure student engagement across differe lecture modalities (in-person, asynchronous, etc.) futher providing a scalable measyre to identify engagement types (i.e. attentive, contributing, distracted) given raw dialogue transcripts using LLMs.

Solution:
By fine-tuning an LLM on labeled conversational data, the system learns to classify and summarize student behavior, generating outputs that can aid professors in changing lecture format. Additionally supporting learning analytics, tutoring systems and classroom monitoring tools.

## Project Structure
- 'student_conversation – CSV': raw student-professor conversation data taken from lectures
- 'classroom_conversations_labeled.csv': labeled summaries used for training
- 'lllm_finetuning_script.py' - script for finetuning LLM (i.e. GPT)
- 'README.md' - project overview & instructions for recreation

## Dataset
- **source**: AI generated transcript of spoken classroom conversations between students during a lecture
- **input**: raw transcripts of classroom conversations
- **label**:
    - Attentive: Student focused and attentive to lecture.
    - Contributing: Student is asking or answering questions to add meaningful discussion to dialogue.
    - Neutral: Student models passive participation; no definitive sign of engagement or lack thereof.
    - Collaborating: Student is engaged in lecture with classmates through multi-person dicussion and group activity.
    - Non-attentive: Student is distracted, not engaged or asking unrelated questions.

The goal is to train a model to automatically determine whether discussion belong to any of the five categories.

## Model & Method
- Model: GPT-based architecture (via HuggingFace Transformers)
- Training Script: llm_finetuning_script.py
- Libraries: transformers, datasets, torch, and pandas
- Task: text summarization using sequence-to-sequence learning
- Approach:
    - Preprocess conversation transcripts and categorize with labeled summaries
    - Fine-tune using supervised learning
    - Evaluate with ROUGE/L or summary-based metrics (in progress)

## Requirements

requires Python 3.8+ with libraries that include:
- pandas
- datasets
- torch
- transformers

Install with:
pip install -r requirements.txt

# How to run
git clone https://github.


