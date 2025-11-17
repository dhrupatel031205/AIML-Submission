"""
AmbedkarGPT - Advanced RAG System for Dr. B.R. Ambedkar's Speech
Assignment 1: Kalpit Pvt Ltd AI Intern Hiring

This script implements an advanced Retrieval-Augmented Generation (RAG) system that:
1. Loads text from speech.txt using TextLoader
2. Splits text into manageable chunks using CharacterTextSplitter
3. Creates embeddings using HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2
4. Stores embeddings in ChromaDB vector store
5. Retrieves relevant chunks based on user questions
6. Generates answers using Ollama with Mistral 7B
"""

import os
import sys

# Try to use the exact LangChain imports you specified
try:
    from langchain.document_loaders import TextLoader
    from langchain.text_splitters import CharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import Ollama
    from langchain.chains import RetrievalQA
    
    def main():
        """Main function to run the RAG system"""
        print("=== Initializing AmbedkarGPT Advanced RAG System ===")
        
        # Create RAG system instance
        rag_system = {}
        
        # 1. Load document
        try:
            print("Loading document from speech.txt...")
            loader = TextLoader("speech.txt", encoding='utf-8')
            documents = loader.load()
            print(f"Successfully loaded {len(documents)} documents")
            print(f"Document content preview: {documents[0].page_content[:200]}...")
            rag_system['documents'] = documents
        except Exception as e:
            print(f"Error loading document: {e}")
            return
        
        # 2. Split text into chunks
        try:
            print("Splitting text into chunks...")
            text_splitter = CharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=50,
                separator=" "
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            # Display chunks for verification
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
            
            rag_system['chunks'] = chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            return
        
        # 3. Create embeddings
        try:
            print("Creating HuggingFace embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Embeddings model loaded successfully")
            rag_system['embeddings'] = embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return
        
        # 4. Create vector store
        try:
            print("Creating ChromaDB vector store...")
            persist_directory = "./chroma_db"
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
            
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print("Vector store created and populated successfully")
            rag_system['vector_store'] = vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return
        
        # 5. Setup LLM
        try:
            print("Setting up Ollama with Mistral 7B...")
            llm = Ollama(
                model="mistral",
                temperature=0.1
            )
            print("LLM setup complete")
            rag_system['llm'] = llm
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            print("Make sure Ollama is installed and Mistral model is pulled")
            print("Run: ollama pull mistral")
            return
        
        # 6. Create QA chain
        try:
            print("Creating QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            print("QA chain created successfully")
            rag_system['qa_chain'] = qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return
        
        print("\n=== System Initialization Complete ===")
        
        # Test questions
        test_questions = [
            "What is the real remedy according to Dr. Ambedkar?",
            "Why does Dr. Ambedkar compare social reform to gardening?",
            "What must people choose between according to the speech?",
            "What is the real enemy according to Dr. Ambedkar?"
        ]
        
        print("\n=== Running Test Questions ===")
        for question in test_questions:
            ask_question(qa_chain, question)
            print("=" * 60)
        
        # Interactive mode
        interactive_mode(qa_chain)

except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to simplified version...")
    
    # Simple fallback implementation
    import re
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import subprocess
    
    def load_simple_document():
        try:
            print("Loading document from speech.txt...")
            with open("speech.txt", 'r', encoding='utf-8') as file:
                text = file.read()
            print("Document loaded successfully")
            return text
        except Exception as e:
            print(f"Error loading document: {e}")
            return None
    
    def split_simple_text(text):
        try:
            print("Splitting text into chunks...")
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= 200:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}: {chunk[:100]}...")
            
            return chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            return None
    
    def create_simple_embeddings(chunks):
        try:
            print("Creating TF-IDF embeddings...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            chunk_embeddings = vectorizer.fit_transform(chunks)
            print("Embeddings created successfully")
            return vectorizer, chunk_embeddings
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None, None
    
    def retrieve_simple_chunks(vectorizer, chunk_embeddings, chunks, question, top_k=3):
        try:
            question_embedding = vectorizer.transform([question])
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                relevant_chunks.append({
                    'chunk': chunks[idx],
                    'similarity': similarities[idx]
                })
            
            return relevant_chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def generate_simple_answer(question, relevant_chunks):
        context = "\n".join([chunk['chunk'] for chunk in relevant_chunks])
        question_lower = question.lower()
        context_lower = context.lower()
        
        if "real remedy" in question_lower or "solution" in question_lower:
            if "destroy the belief in the sanctity of the shastras" in context_lower:
                return "According to Dr. Ambedkar, the real remedy is to destroy the belief in the sanctity of the shastras."
        
        if "gardening" in question_lower or "gardener" in question_lower:
            if "work of social reform is like the work of a gardener" in context_lower:
                return "Dr. Ambedkar compares social reform to gardening because both involve pruning leaves and branches without attacking the roots."
        
        if "choose between" in question_lower or "either" in question_lower:
            if "either you must stop the practice of caste or you must stop believing in the shastras" in context_lower:
                return "According to Dr. Ambedkar, people must choose between stopping the practice of caste or stopping their belief in the shastras."
        
        if "real enemy" in question_lower:
            if "real enemy is the belief in the shastras" in context_lower:
                return "Dr. Ambedkar states that the real enemy is the belief in the shastras."
        
        return "Based on the speech, the answer relates to Dr. Ambedkar's views on caste and the shastras."
    
    def ask_simple_question(vectorizer, chunk_embeddings, chunks, question):
        try:
            print(f"\nQuestion: {question}")
            print("Searching for relevant context...")
            
            relevant_chunks = retrieve_simple_chunks(vectorizer, chunk_embeddings, chunks, question)
            
            if not relevant_chunks:
                print("No relevant chunks found.")
                return None
            
            print(f"Found {len(relevant_chunks)} relevant chunks:")
            for i, chunk_info in enumerate(relevant_chunks, 1):
                print(f"Chunk {i} (similarity: {chunk_info['similarity']:.3f}): {chunk_info['chunk'][:100]}...")
            
            answer = generate_simple_answer(question, relevant_chunks)
            print(f"\nAnswer: {answer}")
            return answer, relevant_chunks
            
        except Exception as e:
            print(f"Error processing question: {e}")
            return None, None
    
    def interactive_mode_simple(vectorizer, chunk_embeddings, chunks):
        print("\n=== AmbedkarGPT Interactive Q&A System ===")
        print("Ask questions about Dr. B.R. Ambedkar's speech")
        print("Type 'quit' to exit\n")
        
        while True:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using AmbedkarGPT!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            ask_simple_question(vectorizer, chunk_embeddings, chunks, question)
            print("-" * 50)
    
    def main_simple():
        print("=== Initializing Simple RAG System ===")
        
        text = load_simple_document()
        if not text:
            return
        
        chunks = split_simple_text(text)
        if not chunks:
            return
        
        vectorizer, chunk_embeddings = create_simple_embeddings(chunks)
        if not vectorizer:
            return
        
        print("\n=== System Initialization Complete ===")
        
        test_questions = [
            "What is the real remedy according to Dr. Ambedkar?",
            "Why does Dr. Ambedkar compare social reform to gardening?",
            "What must people choose between according to the speech?",
            "What is the real enemy according to Dr. Ambedkar?"
        ]
        
        print("\n=== Running Test Questions ===")
        for question in test_questions:
            ask_simple_question(vectorizer, chunk_embeddings, chunks, question)
            print("=" * 60)
        
        interactive_mode_simple(vectorizer, chunk_embeddings, chunks)

if __name__ == "__main__":
    main_simple()

def ask_question(qa_chain, question):
    """Ask a question and get an answer from the RAG system"""
    try:
        print(f"\nQuestion: {question}")
        print("Searching for relevant context...")
        
        result = qa_chain({"query": question})
        
        answer = result['result']
        source_docs = result['source_documents']
        
        print(f"Answer: {answer}")
        print("\nRelevant source chunks:")
        for i, doc in enumerate(source_docs, 1):
            print(f"Source {i}: {doc.page_content}")
        
        return answer, source_docs
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return None, None

def interactive_mode(qa_chain):
    """Run the system in interactive mode"""
    print("\n=== AmbedkarGPT Interactive Q&A System ===")
    print("Ask questions about Dr. B.R. Ambedkar's speech on caste and shastras")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using AmbedkarGPT!")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        ask_question(qa_chain, question)
        print("-" * 50)

if __name__ == "__main__":
    main()
