# Resume Analyzer & Intelligent Job Matcher

An end-to-end NLP-based system that processes resumes and intelligently matches them with job descriptions using Word2Vec embeddings and skill boosting.


## Project Overview

This project implements a complete **Resume Analyzer and Job Matcher** as part of the NLP Mini Project. It is divided into two main components:

- **Resume Processor**: Extracts text from PDFs/TXT files, cleans the text, extracts skills, computes TF-IDF scores, and generates semantic vectors using Word2Vec.
- **Job Matcher**: Uses the same Word2Vec model to match processed resumes against job descriptions using **cosine similarity + skill-based boosting**.

The system runs as a single integrated pipeline and provides clean, professional output with detailed JSON reports.


## Features

- Robust PDF and TXT resume parsing (using pdfplumber)
- Text cleaning, tokenization, and lemmatization with NLTK
- Skill extraction using keyword matching
- TF-IDF analysis for important words
- Word2Vec embeddings with intelligent model caching
- Hybrid job matching (Cosine Similarity + Skill Boost)
- Clean console output with professional summary table
- Structured JSON outputs for further analysis
