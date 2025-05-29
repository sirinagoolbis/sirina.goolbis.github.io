# LLM Powered Microsoft Support Ticket Classifier

## Project Overview

This project builds a machine learning-powered system to classify and respond to Microsoft support tickets. It integreates NLP principles and OpenAI's GPT models to automatically categorize issues submitted by users and further develop an empathetic, context-aware support response.

## Features

- **Automated ticket classification:** Classifies inciming support tickets into categories trhough a machine learning model trained on ticket data.
- **Response Generation:** Generates empathetic, situation specific response using OpenAI's GPT API.
- **Text preprocessing pipeline:** Includes robust data cleaning and TF-IDF vectorization for feature extraction.
- **Model persistence:** Continuous saving and loading of trained models and vectorizers for accuracy in deployment.
- **Streamlit web app:** Interactive UI for ticket input, classificaiton and response generation.
- **Error handling & environment management:** Secure API key management with '.env' and error handling for smooth usage.

## Problem Statement

Microsoft receives numerous support tickets across different products and services, requiring timely categroization and personable responses. Manual classification is timely and can result in human error. This project aims to produce an automated ticket classifier and response generator to further:
- Classify tickets into categories such as Security Concern, Performance Issue, Bug Report, and more
- Improve customer satisfaction with quicker response time
- Help support staff prioritize problems and respond accordingly
- Produce empathetic, personalized responses powered by OpenAI LLMs

## Technologies & Tools

- Python (pandas, scikit-learn, joblib)
- OpenAI API (GPT-3 text-davinci-003)
- Streamlit (for web interface)
- Regex and text preprocessing techniques
- Version control with GitHub

### Prerequisites

 - Python 3.8 or higher
 - OpenAI API key ([OpenAI](https://platform.openai.com))
 - Required Python pakcages (install with 'pip install -r requirements.txt')

# Usage

- Enter a support ticket descripting in the app
- Classifer with predict the ticket classification.
- System will generate appropriate, empathetic response based on ticket classification.

# Future Improvements

- Integrate real-time ticket iingestion from support platofrms (i.e. Zendesk, Microsoft Dyanmics)
- Incorporate multi-language support for global ticket handling
- 

