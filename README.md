# ChatCapitalHumain

## Overview

ChatCapitalHumain is an LLM-powered agent-based chatbot designed to analyze and provide insights on student survey data from Acadian schools in Atlantic Canada. The application connects to a PostgreSQL database containing questionnaire responses collected between 2004 and 2019 across seven schools in the region. 

[https://chatcapitalhumain.streamlit.app/](https://chatcapitalhumain.streamlit.app/) to interact with the system directly. The sidebar provides access to example questions and the database schema to help formulate effective queries.

## Features

### Database Schema

The application is built around a comprehensive database containing:
- Student responses to 52 questions across five categories (General Questions, Socio-Demographic Information, Post-Secondary Education, Job Market, Employment Search)
- Data from seven Acadian schools over a 15-year period (2004-2019)
- Aggregated responses by school, year, questionnaire, and gender

### Dual Agent Architecture

The system implements two distinct agent architectures:

1. **Single Agent (simple)**: 
   - Direct database query processing with the LangChain ReAct pattern
   - Streamlined question-to-SQL flow for straightforward queries
   - Memory persistence across sessions via LangGraph's MemorySaver

2. **Multi-Agent with Human-in-the-Loop** :
   - LangGraph-powered swarm workflow with specialized agent nodes
   - Human feedback integration during the query formulation phase with Human-in-the-Loop
   - Transparent workflow visualization of agent handoffs
   - Structured outputs ensuring data quality and accuracy

### Vector Search for Questions

- Pre-populated vector database containing example queries to help users understand what types of questions can be answered

### Data Visualization

- Automated visualization of query results using Plotly in the Multi-Agent Architecture
- Interactive charts and graphs representing temporal trends and comparisons

### Key Features

- **Human-in-the-Loop**: Expert validation of query interpretations before execution
- **Login System**: User authentication to save conversation history
- **Schema Reference**: Comprehensive database schema available in the sidebar
- **Model Selection**: Choice between different LLM models (Claude, GPT, Gemini)
- **Memory Persistence**: Conversations maintained across sessions
- **Evaluation Framework**: Benchmarking different models on question answering tasks

## Technical Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL
- **LLM Integration**: LangChain, LangGraph
- **Vector Search**: FAISS
- **Visualization**: Plotly
- **Authentication**: Streamlit authentication for user management
