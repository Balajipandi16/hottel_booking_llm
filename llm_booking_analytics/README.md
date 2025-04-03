# LLM-Powered Booking Analytics & QA System

## Overview
This project provides hotel booking data analysis and an LLM-powered question-answering system using Retrieval-Augmented Generation (RAG). It extracts insights from booking data and allows users to ask natural language questions about the dataset.

## Features
- **Data Preprocessing**: Cleans and structures hotel booking data.
- **Analytics & Reporting**: Generates insights such as revenue trends, cancellation rates, and booking lead time.
- **Retrieval-Augmented QA**: Uses Weaviate for vector storage and an open-source LLM for answering queries.
- **API Development**: Provides endpoints for analytics and question-answering.

## Project Structure
```
project_folder/
│
├── data/
│   └── hotel_bookings.csv
│
├── analytics/
│   ├── cancellation_rate.py
│   ├── distribution.py
│   ├── booking_lead_time.py
│   ├── additional_analytics.py
│
├── qa_system/
│   ├── vector_embeddings.py
│   ├── bert_qa.py
│
├── api/
│   ├── app.py
│   ├── endpoints.py
│
├── tests/
│   └── sample_queries.txt
│
└── reports/
    └── implementation_report.md
```

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Balajipandi16/LLM_Booking_Analytics_QA_System.git
cd LLM_Booking_Analytics_QA_System
pip install -r requirements.txt
```

## Running the API
Start the FastAPI server:
```bash
uvicorn api.app:app --reload
```

## Usage
- **POST `/analytics`** → Returns hotel booking analytics reports.
- **POST `/ask`** → Answers user queries based on booking data.

## Example Queries
```
"Show me total revenue for July 2017."
"Which locations had the highest booking cancellations?"
"What is the average price of a hotel booking?"
```

## Dependencies
- Python 3.8+
- Pandas, NumPy, Matplotlib, Seaborn
- Weaviate-client, Transformers, FastAPI

## Contribution
Feel free to submit issues and pull requests to improve this project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

