# AmbedkarGPT-Intern-Task

A Retrieval-Augmented Generation (RAG) system for answering questions about Dr. B.R. Ambedkar's speech on caste and shastras.

## System Overview

This system implements a complete RAG pipeline that:
1. Loads text from `speech.txt` (Dr. B.R. Ambedkar's speech excerpt)
2. Splits text into manageable chunks
3. Creates embeddings using HuggingFace sentence-transformers
4. Stores embeddings in ChromaDB vector store
5. Retrieves relevant chunks based on user questions
6. Generates answers using Ollama with Mistral 7B

## Technical Stack

- **Programming Language:** Python 3.8+
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- **LLM:** Ollama with Mistral 7B
- **All components are free and run locally**

## Prerequisites

### 1. Install Ollama
Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

### 2. Pull Mistral Model
```bash
ollama pull mistral
```

### 3. Verify Ollama Installation
```bash
ollama list
```
You should see `mistral` in the list of installed models.

## Setup Instructions

### 1. Clone or Download the Repository
```bash
git clone [<repository-url>](https://github.com/dhrupatel031205/AIML-Submission.git)
cd AmbedkarGPT-Intern-Task
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv ambedkar_env

# Activate on Windows
ambedkar_env\Scripts\activate

# Activate on macOS/Linux
source ambedkar_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import langchain, chromadb; print('Dependencies installed successfully')"
```

## Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py              # Main RAG system implementation
├── requirements.txt     # Python dependencies
├── speech.txt          # Dr. Ambedkar's speech text
├── README.md           # This file
├── chroma_db/          # ChromaDB vector store (created automatically)
└── ambedkar_env/       # Virtual environment (created during setup)
```

## Running the System

### Method 1: Direct Execution
```bash
python main.py
```

### Method 2: Interactive Mode
The system will automatically start in interactive mode after running test questions.

## Usage Examples

### Test Questions
The system includes built-in test questions:
1. "What is the real remedy according to Dr. Ambedkar?"
2. "Why does Dr. Ambedkar compare social reform to gardening?"
3. "What must people choose between according to the speech?"
4. "What is the real enemy according to Dr. Ambedkar?"

### Interactive Mode
After the test questions complete, you can ask your own questions:
```
Your question: What does Dr. Ambedkar say about the shastras?
Your question: How does he describe the problem of caste?
Your question: quit
```

## System Components

### Document Processing
- **TextLoader:** Loads speech.txt with UTF-8 encoding
- **CharacterTextSplitter:** Splits text into 200-character chunks with 50-character overlap

### Embeddings
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Device:** CPU (compatible with most systems)
- **Purpose:** Converts text chunks to vector embeddings

### Vector Store
- **Database:** ChromaDB
- **Storage:** Local directory `./chroma_db`
- **Retrieval:** Top 3 most relevant chunks for each query

### LLM Integration
- **Model:** Mistral 7B via Ollama
- **Temperature:** 0.1 (focused responses)
- **Prompt Template:** Customized for context-based Q&A

## Features

### ✅ Core Functionality
- Document loading and processing
- Text chunking with overlap
- Embedding generation and storage
- Semantic search and retrieval
- Context-aware answer generation

### ✅ Error Handling
- Comprehensive error messages
- Graceful fallback for missing dependencies
- Clear setup instructions

### ✅ User Experience
- Interactive command-line interface
- Source document display
- Test question demonstration
- Easy quit functionality

## Expected Output Example

```
=== Initializing AmbedkarGPT RAG System ===
Loading document from speech.txt...
Successfully loaded 1 documents
Splitting text into chunks...
Created 4 chunks
Chunk 1: The real remedy is to destroy the belief in the sanctity of the shastras...
Chunk 2: You must take a stand against the scriptures...
Chunk 3: The problem of caste is not a problem of social reform...
Chunk 4: The work of social reform is like the work of a gardener...
Creating HuggingFace embeddings...
Embeddings model loaded successfully
Creating ChromaDB vector store...
Vector store created and populated successfully
Setting up Ollama with Mistral 7B...
LLM setup complete
Creating QA chain...
QA chain created successfully

=== System Initialization Complete ===

=== Running Test Questions ===

Question: What is the real remedy according to Dr. Ambedkar?
Searching for relevant context...
Answer: According to Dr. Ambedkar, the real remedy is to destroy the belief in the sanctity of the shastras.

Relevant source chunks:
Source 1: The real remedy is to destroy the belief in the sanctity of the shastras...
Source 2: So long as people believe in the sanctity of the shastras...
Source 3: The real enemy is the belief in the shastras.
```

## Future Enhancements

Potential improvements for production use:
- Web interface (Flask/FastAPI)
- Multiple document support
- Advanced chunking strategies
- Conversation memory
- Performance optimization
- Docker containerization
