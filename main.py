from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
import os
import uvicorn
import fitz  # PyMuPDF
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering.chain import load_qa_chain
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

UPLOAD_DIRECTORY = "/home/khaing/ragApp"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


@app.get("/", response_class=HTMLResponse)
async def main():
    with open("index.html") as file:
        return file.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process the uploaded PDF file
    pdf_text = extract_text_from_pdf(file_location)
    process_text_with_rag(pdf_text)

    return {"info": f"file '{file.filename}' uploaded and processed successfully"}


@app.post("/ask")
async def ask_question_endpoint(question: str = Form(...)):
    answer = ask_question(question)
    return {"answer": answer}


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def process_text_with_rag(document_text):
    # Retrieve NVIDIA API key from environment variables
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')

    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment variables")

    print(f"NVIDIA_API_KEY: {nvidia_api_key}")  # Debugging line to check API key

    # Initialize the ChatNVIDIA model
    llm = ChatNVIDIA(model="meta/llama2-70b", nvidia_api_key=nvidia_api_key, max_tokens=1000)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )

    # Create embeddings
    embeddings = NVIDIAEmbeddings()

    dest_embed_dir = 'embed'

    chunks = text_splitter.split_text(document_text)
    texts = chunks
    metadatas = [{"source": "uploaded_pdf"}] * len(chunks)

    # Create embeddings and add to vector store
    if os.path.exists(dest_embed_dir):
        docsearch = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        docsearch.add_texts(texts, metadatas=metadatas)
        docsearch.save_local(folder_path=dest_embed_dir)
    else:
        docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        docsearch.save_local(folder_path=dest_embed_dir)


# Function to ask a question
def ask_question(query):
    # Load embeddings from the vector store
    embedding_model = NVIDIAEmbeddings()
    embedding_path = "embed/"
    docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

    # Retrieve NVIDIA API key from environment variables
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')

    # Initialize the ChatNVIDIA model
    llm = ChatNVIDIA(model="meta/llama2-70b", nvidia_api_key=nvidia_api_key, max_tokens=1000)

    # Define the default QA prompt
    QA_PROMPT = """
    You are a knowledgeable assistant and chef who is expert in Italian food. Use the following context to answer the question.If user asks about recipe of other food or which is not from your retrieval and context,just reply you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Create the PromptTemplate object
    prompt_template = PromptTemplate(template=QA_PROMPT, input_variables=["context", "question"])

    # Load the QA chain
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template, document_variable_name="context")

    # Create a ConversationalRetrievalChain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type="stuff",
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt_template},
    )
    
    # Use the query directly without manually retrieving the context
    result = qa.invoke({"question": query})
    return result.get("answer")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

