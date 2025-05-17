# LLM Candidate Summarizer

## project structure
- 'student_conversation – CSV': raw student-professor conversation data taken from lectures
- 'classroom_conversations_labeled.csv': labeled summaries used for training
- 'lllm_finetuning_script.py' - script for finetuning LLM (i.e. GPT)
- 'README.md' - project overview & instructions for recreation

## Dataset
- **input**: transcript of classroom conversations
- **label**: human-defined summary of conversations

## Requirements

requires Python 3.8+ with:

'''txt
- pandas
- datasets
- torch
- transformers

Install with:
pip install -r requirements.txt
