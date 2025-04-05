import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pineconeIndexing import (
    initialize_pinecone,
    connect_to_index,
    list_pinecone_indexes,
    clear_pinecone_index,
    sanitize_index_name
)
from huggingface_hub import InferenceClient
import logging

logging.basicConfig(level=logging.INFO)

class ChatBot:
    def __init__(self, file_path=None, file_name=None):
        # Load environment variables
        load_dotenv()
        if file_path and file_name:
            self._initialize_pinecone_and_connect_index(file_path, file_name)
            self._setup_huggingface_client()
            self._define_prompt_and_chain()

    def process_file(self, file_path, file_name):
        """Process a file and initialize the chatbot."""
        self._initialize_pinecone_and_connect_index(file_path, file_name)
        self._setup_huggingface_client()
        self._define_prompt_and_chain()

    def clear_index(self, index_name):
        """Clear the Pinecone index."""
        clear_pinecone_index(index_name)

    def ask_question(self, question):
        """Ask a question to the chatbot."""
        if not hasattr(self, 'rag_chain') or self.rag_chain is None:
            raise ValueError("Chatbot is not initialized. Please process a file first.")
        return self.rag_chain.invoke(question)

    def _initialize_pinecone_and_connect_index(self, file_path, file_name):
        """Initialize the Pinecone index."""
        try:
            file_name = sanitize_index_name(file_name)
            initialize_pinecone()
            self.docsearch = connect_to_index(file_path, "sentence-transformers/all-MiniLM-L12-v2", file_name)
            logging.info(f"Index '{file_name}' initialized successfully.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error initializing Pinecone index: {e}")
            raise

    def _setup_huggingface_client(self):
        """Set up the HuggingFace Inference Client."""
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is not set.")
        self.client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=api_key
        )

    def _define_prompt_and_chain(self):
        """Define the prompt template and RAG chain."""
        self.prompt = PromptTemplate(
            template="""
            You are a problem solver. Use the provided context to address the question or solve the problem. 
            If the context does not contain enough information, respond with "I don't know." 
            Keep your response clear and concise.

            Context: {context}
            Question: {question}
            Answer: 
            """,
            input_variables=["context", "question"]
        )

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | (lambda prompt_value: [{"role": "user", "content": str(prompt_value)}])
            | (lambda messages: self.client.chat_completion(messages))
            | (lambda output: output.choices[0].message["content"]) 
            | StrOutputParser()
        )
