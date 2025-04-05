import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pineconeIndexing import initialize_pinecone, clear_pinecone_index
from huggingface_hub import InferenceClient

class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self._main_menu()

    def _main_menu(self):
        """Loop the main menu until the user chooses to exit."""
        while True:
            user_choice = input("What do you want to do?\n  1. Process a text file\n  2. Remove Pinecone index\nType 'exit' to quit\n")
            if user_choice == '1':
                self._process_text_file()
            elif user_choice == '2':
                self._clear_index()
            elif user_choice == 'exit':
                print("Exiting program.")
                quit()
            else:
                print("Invalid choice. Please try again.")

    def _process_text_file(self):
        """Handle processing of a text file."""
        file_name = input("Please enter the file path: ")
        file_path = "./data/" + file_name
        self._initialize_pinecone(file_path, file_name)
        self._setup_huggingface_client()
        self._define_prompt_and_chain()
        self._question_loop()

    def _clear_index(self):
        """Clear the Pinecone index."""
        clear_pinecone_index()
        print("Cleared Pinecone index.")

    def _question_loop(self):
        """Loop to handle user questions."""
        while True:
            user_question = input("Ask me anything (Enter 'exit' or 'quit' to go back to main menu): ")
            if user_question.lower() in ['exit', 'quit']:
                break
            result = self.rag_chain.invoke(user_question)
            print(result)

    def _initialize_pinecone(self, file_path, file_name):
        """Initialize the Pinecone index."""
        try:
            self.docsearch = initialize_pinecone(file_path, "sentence-transformers/all-MiniLM-L12-v2", file_name)
        except FileNotFoundError as e:
            print(e)
            self.docsearch = None

        if self.docsearch is None:
            print("Failed to initialize Pinecone index. Exiting.")
            quit()

    def _setup_huggingface_client(self):
        """Set up the HuggingFace Inference Client."""
        self.client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=os.getenv('HUGGINGFACE_API_KEY')  # Ensure API key is loaded
        )

    def _define_prompt_and_chain(self):
        """Define the prompt template and RAG chain."""
        self.prompt = PromptTemplate(
            template="""
            You are a fortune teller. These Human will ask you a questions about their life. 
            Use following piece of context to answer the question. 
            If you don't know the answer, just say you don't know. 
            Keep the answer within 2 sentences and concise.

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
        
    def ask_question(self, question):
        """Ask a question to the chatbot."""
        return self.rag_chain.invoke({"question": question})
