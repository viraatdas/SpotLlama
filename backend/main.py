from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import LlamaCppEmbeddings


DIR_PATH = "test"
llm = ChatOllama(model="llama3")

def load_documents_from_directory(directory):
    # Initialize the loader with the directory path
    loader = DirectoryLoader(directory)
    
    # Load the documents
    documents = loader.load()

    # Print out details of each document
    for doc in documents:
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:500]}")  # Print the first 500 characters for a quick summary
        print("-" * 40)
    
    return documents

docs = load_documents_from_directory(DIR_PATH)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create embeddings and vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=LlamaCppEmbeddings())

# Setup retriever
retriever = vectorstore.as_retriever()

# Load the prompt
prompt = load_prompt("rlm/rag-prompt")

# Define the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm  
    | StrOutputParser()
)



prompt = "Tell me about Venkat?"
response = rag_chain.invoke(prompt)
print(response)


