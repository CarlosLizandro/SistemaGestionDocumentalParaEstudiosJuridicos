import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pdf2image import convert_from_path
import base64
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajusta la ruta si es diferente

# Configuración de Poppler (asegúrate de tenerlo instalado y en PATH)
POPPLER_PATH = r"C:\Users\Carlos\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # Cambia esta ruta según tu instalación

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'multi-modal-rag/pdfs/'
figures_directory = 'multi-modal-rag/figures/'

# Inicialización de modelos
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",  # Alternativa: "llama3:8b-instruct" (5.1GB)
    temperature=0.1
)

# 2. Modelo de generación liviano (5-8GB RAM)
model = Ollama(
    model="llama3:8b-instruct-q4_K_M",  # Solo 5.1GB RAM
    num_ctx=2048,  # Reduce el contexto para ahorrar memoria
    temperature=0.7
)
vector_store = None

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    os.makedirs(figures_directory, exist_ok=True)
    
    try:
        # Procesamiento del PDF con Poppler configurado
        elements = partition_pdf(
            file_path,
            strategy=PartitionStrategy.HI_RES,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=figures_directory,
            pdf2image_kwargs={"poppler_path": POPPLER_PATH}  # Añadido parámetro poppler
        )

        text_elements = [element.text for element in elements if hasattr(element, 'text') and element.category not in ["Image", "Table"]]

        # Procesamiento de imágenes extraídas
        for file in os.listdir(figures_directory):
            img_path = os.path.join(figures_directory, file)
            extracted_text = extract_text(img_path)
            text_elements.append(extracted_text)

        return "\n\n".join(text_elements)
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def extract_text(file_path):
    try:
        # 1. Leer la imagen como bytes
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        # 2. Codificar a base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # 3. Pasar a Ollama
        model_with_image_context = model.bind(images=[base64_image])
        return model_with_image_context.invoke("Describe the content of this image in detail.")
    
    except Exception as e:
        st.warning(f"Could not process image: {str(e)}")
        return ""

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def index_docs(texts):
    global vector_store
    if texts:
        vector_store = FAISS.from_texts(texts, embeddings)
    else:
        st.warning("No text content to index.")

def retrieve_docs(query):
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=3)  # Retorna los 3 documentos más relevantes

def answer_question(question, documents):
    if not documents:
        return {"content": "No relevant documents found to answer the question."}
    
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Interfaz Streamlit
st.title("PDF QA Assistant")
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    pdf_path = os.path.join(pdfs_directory, uploaded_file.name)
    
    with st.spinner("Processing PDF..."):
        text = load_pdf(pdf_path)
        if text:
            chunked_texts = split_text(text)
            index_docs(chunked_texts)
            st.success("PDF processed successfully!")
        else:
            st.error("Failed to extract content from PDF")

    question = st.text_input("Ask a question about the PDF:")
    
    if question:
        with st.spinner("Searching for answers..."):
            st.chat_message("user").write(question)
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            if isinstance(answer, dict):
                st.chat_message("assistant").write(answer.get("answer", str(answer)))
            else:
                st.chat_message("assistant").write(str(answer))

        # Opcional: Mostrar documentos relevantes
        with st.expander("See relevant documents"):
            for i, doc in enumerate(related_documents):
                st.write(f"Document {i+1}:")
                st.text(doc.page_content[:500] + "...")  # Muestra solo un fragmento