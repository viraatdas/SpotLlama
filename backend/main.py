from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import bs4
from langchain import hub

DIR_PATH = "test"
LLAMA_MODEL_PATH = "/path/to/llama_model.bin"  # Specify the path to your Llama model file

llm = ChatOllama(model="llama3")

def load_documents_from_directory(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    for doc in documents:
        print(f"Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:500]}")
        print("-" * 40)
    return documents

docs = load_documents_from_directory(DIR_PATH)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_rag_chain(input):
    # Using LangChain Expressive Language chain syntax
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(input)

input = "Tell me about Venkat"
print(invoke_rag_chain(input))
