# LLM Powered Microsoft Support Ticket Classifier

This project builds a machine learning-powered system to classify and respond to Microsoft support tickets. It integreates NLP principles and OpenAI's GPT models to automatically categorize issues submitted by users and further develop an empathetic, context-aware support response.

It is designed for:
- Automated triaging of customer support tickets
- Immediate, relevant assistance to users
- Enhancing support team efficiency
- Streamlining customer-support at large scale

## Problem Statement

Microsoft receives numerous support tickets across different products and services, requiring timely categroization and personable responses. Manual classification is timely and can result in human error. This project aims to produce an automated ticket classifier and response generator to further:
- Classify tickets into categories such as Security Concern, Performance Issue, Bug Report, and more
- Improve customer satisfaction with quicker response time
- Help support staff prioritize problems and respond accordingly
- Produce empathetic, personalized responses powered by OpenAI LLMs

## Solution

- Preprocess and clean user ticket input
- Train a Multinomial Naive Bayes classifier on labeled Microsoft support tickets
- Vectorize ticket text with TF-IDF features
