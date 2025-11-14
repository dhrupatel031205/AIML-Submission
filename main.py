"""
AmbedkarGPT - RAG System for Dr. B.R. Ambedkar's Speech
Assignment 1: Kalpit Pvt Ltd AI Intern Hiring

This script implements a Retrieval-Augmented Generation (RAG) system that:
1. Loads text from speech.txt
2. Splits text into manageable chunks
3. Creates embeddings using HuggingFace sentence-transformers
4. Stores embeddings in ChromaDB vector store
5. Retrieves relevant chunks based on user questions
6. Generates answers using Ollama with Mistral 7B
"""

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class AmbedkarRAG:
    def __init__(self):
        """Initialize the RAG system with all components"""
        self.loader = None
        self.documents = None
        self.text_splitter = None
        self.chunks = None
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
    def load_document(self, file_path="speech.txt"):
        """Load the speech document from file"""
        try:
            print(f"Loading document from {file_path}...")
            self.loader = TextLoader(file_path, encoding='utf-8')
            self.documents = self.loader.load()
            print(f"Successfully loaded {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def split_text(self):
        """Split the document into manageable chunks"""
        try:
            print("Splitting text into chunks...")
            self.text_splitter = CharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=50,
                separator=" "
            )
            self.chunks = self.text_splitter.split_documents(self.documents)
            print(f"Created {len(self.chunks)} chunks")
            
            # Display chunks for verification
            for i, chunk in enumerate(self.chunks):
                print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
            
            return True
        except Exception as e:
            print(f"Error splitting text: {e}")
            return False
    
    def create_embeddings(self):
        """Create HuggingFace embeddings"""
        try:
            print("Creating HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Embeddings model loaded successfully")
            return True
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return False
    
    def create_vector_store(self):
        """Create and populate ChromaDB vector store"""
        try:
            print("Creating ChromaDB vector store...")
            # Create a directory for the vector store if it doesn't exist
            if not os.path.exists("./chroma_db"):
                os.makedirs("./chroma_db")
            
            self.vector_store = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print("Vector store created and populated successfully")
            return True
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def setup_llm(self):
        """Setup Ollama with Mistral 7B"""
        try:
            print("Setting up Ollama with Mistral 7B...")
            self.llm = Ollama(
                model="mistral",
                temperature=0.1
            )
            print("LLM setup complete")
            return True
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            print("Make sure Ollama is installed and Mistral model is pulled")
            print("Run: ollama pull mistral")
            return False
    
    def create_qa_chain(self):
        """Create the RetrievalQA chain"""
        try:
            print("Creating QA chain...")
            
            # Custom prompt template
            template = """Use the following context from Dr. B.R. Ambedkar's speech to answer the question. 
            If you don't know the answer based on the context, just say that you don't know. 
            Keep the answer concise and based only on the provided context.

            Context: {context}

            Question: {question}

            Answer:"""
            
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
                ),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True
            )
            print("QA chain created successfully")
            return True
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return False
    
    def initialize_system(self):
        """Initialize the complete RAG system"""
        print("=== Initializing AmbedkarGPT RAG System ===")
        
        if not self.load_document():
            return False
        
        if not self.split_text():
            return False
        
        if not self.create_embeddings():
            return False
        
        if not self.create_vector_store():
            return False
        
        if not self.setup_llm():
            return False
        
        if not self.create_qa_chain():
            return False
        
        print("\n=== System Initialization Complete ===")
        return True
    
    def ask_question(self, question):
        """Ask a question and get an answer from the RAG system"""
        try:
            print(f"\nQuestion: {question}")
            print("Searching for relevant context...")
            
            result = self.qa_chain({"query": question})
            
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
    
    def interactive_mode(self):
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
            
            self.ask_question(question)
            print("-" * 50)


def main():
    """Main function to run the RAG system"""
    # Create and initialize the RAG system
    rag_system = AmbedkarRAG()
    
    # Initialize all components
    if not rag_system.initialize_system():
        print("Failed to initialize the RAG system. Please check the error messages above.")
        return
    
    # Run some test questions
    print("\n=== Running Test Questions ===")
    test_questions = [
        "What is the real remedy according to Dr. Ambedkar?",
        "Why does Dr. Ambedkar compare social reform to gardening?",
        "What must people choose between according to the speech?",
        "What is the real enemy according to Dr. Ambedkar?"
    ]
    
    for question in test_questions:
        rag_system.ask_question(question)
        print("=" * 60)
    
    # Start interactive mode
    rag_system.interactive_mode()


if __name__ == "__main__":
    main()
