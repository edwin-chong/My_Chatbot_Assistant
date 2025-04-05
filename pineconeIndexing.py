import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from PyPDF2 import PdfReader
import csv
import re
import json

def sanitize_index_name(name):
    # Remove file extension if present
    name = os.path.splitext(name)[0]
    # Convert to lowercase
    name = name.lower()
    # Replace invalid characters with hyphens
    name = re.sub(r"[^a-z0-9-]", "-", name)
    # Remove leading or trailing hyphens
    name = name.strip("-")
    return name

def load_file(file_path):
    """Load content from different file types."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".txt":
        # Load text file with fallback encoding
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            print(f"Warning: UTF-8 decoding failed for {file_path}. Trying 'ISO-8859-1' encoding.")
            with open(file_path, "r", encoding="ISO-8859-1") as file:
                return file.read()
    elif file_extension == ".pdf":
        # Load PDF file
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file_extension == ".csv":
        # Load CSV file
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            return "\n".join([", ".join(row) for row in reader])
    elif file_extension == ".json":
        # Load JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return json.dumps(data, indent=2)  # Convert JSON to a readable string
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
def generate_embeddings(file_content, model_name="sentence-transformers/all-MiniLM-L12-v2"):
    # Split the content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    chunks = text_splitter.split_text(file_content)

    # Convert chunks into Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize the HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return docs, embeddings

# Function to initialize or connect to a Pinecone index
def initialize_pinecone(file_path, model_name, index_name):
    """Initialize or connect to a Pinecone index."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path '{file_path}' does not exist. Please provide a valid file path.")

    api_key = _get_pinecone_api_key()
    pc = Pinecone(api_key=api_key)
    indexes = _get_pinecone_indexes(pc)

    index_name = sanitize_index_name(index_name)
    if index_name in indexes:
        return _connect_to_existing_index(index_name, model_name)

    return _create_new_index(pc, file_path, model_name, index_name)

def _connect_to_existing_index(index_name, model_name):
    """Connect to an existing Pinecone index."""
    print(f"Index '{index_name}' already exists. Connecting to the existing index.")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return PineconeVectorStore.from_existing_index(index_name, embeddings)

def _create_new_index(pc, file_path, model_name, index_name):
    """Create a new Pinecone index."""
    print(f"Index '{index_name}' does not exist. Creating a new index: " + index_name)
    try:
        file_content = load_file(file_path)
    except ValueError as e:
        print(e)
        return None

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    docs, embeddings = generate_embeddings(file_content, model_name=model_name)
    print(f"Index '{index_name}' created successfully.")
    return PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

def clear_pinecone_index():
    """
    Clears all data from the specified Pinecone index.
    """
    api_key = _get_pinecone_api_key()
    pc = Pinecone(api_key=api_key)
    indexes = _get_pinecone_indexes(pc)
    index_name = input("Clear Pinecone index? Enter index name: ").strip()
    _validate_and_clear_index(pc, indexes, index_name)

def _get_pinecone_api_key():
    """Retrieve the Pinecone API key from environment variables."""
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
    return api_key

def _get_pinecone_indexes(pc):
    """Retrieve the list of available Pinecone indexes."""
    indexes = [index["name"] for index in pc.list_indexes()]
    print("Getting list of indexes: " + str(indexes))
    return indexes

def _validate_and_clear_index(pc, indexes, index_name):
    """Validate the index name and clear the specified Pinecone index."""
    matching_index = next((index for index in indexes if index == index_name), None)
    if not matching_index:
        raise ValueError(f"Index '{index_name}' does not exist. Available indexes: {indexes}")
    print(f"Clearing all data from index '{index_name}'...")
    pc.delete_index(index_name)
    print(f"Index '{index_name}' has been cleared successfully.")