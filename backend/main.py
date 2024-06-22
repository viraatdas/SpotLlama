import uuid
from flask import Flask, jsonify, request
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain import hub

DIR_PATH = "test"
llm = ChatOllama(model="llama3")

app = Flask(__name__)

CUSTOMER_TO_VECTORSTORE = dict()

@app.route('/generate_customer_id', methods=['GET'])
def generate_customer_id():
    customer_id = str(uuid.uuid4())
    return jsonify({"customer_id": customer_id})

@app.route('/create_index', methods=['POST'])
def create_index():
    customer_id = request.json.get('customer_id')
    

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

@app.route('/query', methods=['POST'])
def invoke_rag_chain():
    customer_id = request.json.get('customer_id')
    input_text = request.json.get('query')

    if customer_id not in CUSTOMER_TO_VECTORSTORE:
        return jsonify({"error": "Retriever not found"}), 404

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return jsonify({"response": rag_chain.invoke(input_text)})

if __name__ == '__main__':
    app.run(port=5000)
