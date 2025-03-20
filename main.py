import os
import logging
import warnings
from tqdm import tqdm
from langsmith import traceable
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM 
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

PERSIST_DIRECTORY = "./data/chroma_db"
ITEMS_FOLDER = "./items"

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(ITEMS_FOLDER, exist_ok=True)

def process_pdfs_in_folder(folder_path, model_name="mistral"):
    """Processes PDFs in the specified folder and stores embeddings in ChromaDB."""
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OllamaEmbeddings(model=model_name))
    existing_docs = set(vectorstore.get()['ids'])
    
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    processed_any = False
    for pdf_path in pdf_files:
        if pdf_path in existing_docs:
            logging.info(f"Skipping already processed PDF: {pdf_path}")
            continue
        
        logging.info(f"Processing PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, length_function=len)
        splits = text_splitter.split_documents(pages)

        for chunk in tqdm(splits, desc=f"Processing chunks in {pdf_path}"):
            vectorstore.add_documents([chunk], embedding=OllamaEmbeddings(model=model_name))
        processed_any = True
    
    return processed_any

@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def create_qa_agent(model_name="mistral"):
    """Create or load the question-answering agent."""
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OllamaEmbeddings(model=model_name))
    
    llm = OllamaLLM(model=model_name)

    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided PDF document.
    Use only the context provided to answer the question. If you don't know the answer, say so.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 50}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

@traceable(run_type="chain")
def ask_question(qa_chain, question):
    """Ask the question to the QA agent and return the result."""
    try:
        response = qa_chain({"query": question})
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    except Exception as e:
        logging.error(f"An error occurred while asking the question: {str(e)}")
        return {"error": str(e), "answer": None, "sources": None}

print("==========================================")
print("                NIKI AI")
print("==========================================")

if process_pdfs_in_folder(ITEMS_FOLDER, model_name="mistral"):
    logging.info("PDF processing complete.")
    qa_agent = create_qa_agent(model_name="mistral")
    
    while True:
        user_input = input("\nEnter your question (or type 'exit' to quit):\n>> ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter a valid question.")
            continue
        
        result = ask_question(qa_agent, user_input)
        if result.get("error"):
            print("Error:", result["error"])
        else:
            print("\nNiki AI Response:")
            print(result["answer"])
