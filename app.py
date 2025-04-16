from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import os
import shutil
import uuid
import psycopg2
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "7sg"
LLM_MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
UPLOAD_FOLDER = "uploads"
TEMP_DIR = "temp_pdf_files"
ALLOWED_EXTENSIONS = {'pdf'}
BASE_URL = "http://localhost:5000"  # Update for production

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Get environment variables
CONNECTION_STRING = os.getenv("POSTGRES_DSN", "postgresql+psycopg2://your_username:your_password@localhost:5432/7sg")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

# Global state
class AppState:
    vectorstore = None
    qa_chain = None
    processed_files = []

state = AppState()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_db_connection(conn_string):
    try:
        conn = psycopg2.connect(conn_string.replace("postgresql+psycopg2://", "postgresql://"))
        conn.close()
        return True, "Database connection successful."
    except psycopg2.OperationalError as e:
        return False, f"Database connection error: {e}"
    except Exception as e:
        return False, f"Unexpected error during DB connection check: {e}"

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def init_vectorstore():
    if state.vectorstore is None:
        try:
            embeddings = get_embeddings_model()
            state.vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
            )
            state.vectorstore.similarity_search("test query", k=1)
            logger.debug("Connected to existing vector store.")
            return True, "Connected to existing vector store."
        except Exception as e:
            state.vectorstore = None
            logger.error(f"Could not connect to vector store: {e}")
            return False, f"Could not connect to vector store: {e}"
    return True, "Vector store already initialized."

def clear_collection_fun():
    try:
        embeddings = get_embeddings_model()
        temp_vs = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
        )
        temp_vs.delete_collection()
        # Clear uploads/ directory
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        state.processed_files = []
        logger.debug("Cleared vector store collection and uploads directory.")
        return True, "Collection and uploads cleared."
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        return False, f"Error clearing collection: {e}"

def process_pdfs(pdf_files, clear_collection=False):
    embeddings = get_embeddings_model()
    all_docs = []

    # Clear collection before processing if requested
    if clear_collection:
        success, clear_message = clear_collection_fun()
        if not success:
            return False, clear_message

    for pdf_file in pdf_files:
        try:
            filename = secure_filename(pdf_file.filename)
            temp_filename = f"{uuid.uuid4()}.pdf"
            temp_filepath = os.path.join(TEMP_DIR, temp_filename)
            upload_filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Save permanently to uploads/
            logger.debug(f"Saving {filename} to {upload_filepath}")
            pdf_file.save(upload_filepath)
            # Save temporarily for processing
            pdf_file.seek(0)
            pdf_file.save(temp_filepath)

            loader = PDFPlumberLoader(temp_filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            os.remove(temp_filepath)
            logger.debug(f"Processed {filename} with {len(docs)} pages")
        except Exception as e:
            logger.error(f"Error processing '{pdf_file.filename}': {e}")
            return False, f"Error processing '{pdf_file.filename}': {e}"

    if not all_docs:
        logger.warning("No documents loaded successfully.")
        return False, "No documents could be loaded successfully."

    try:
        text_splitter = SemanticChunker(embeddings)
        documents = text_splitter.split_documents(all_docs)
        logger.debug(f"Split into {len(documents)} chunks")
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return False, f"Error splitting documents: {e}"

    try:
        state.vectorstore = PGVector.from_documents(
            embedding=embeddings,
            documents=documents,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            pre_delete_collection=False  # Avoid accidental clearing
        )
        if clear_collection:
            state.processed_files = [secure_filename(f.filename) for f in pdf_files]
        else:
            state.processed_files = list(set(state.processed_files + [secure_filename(f.filename) for f in pdf_files]))
        state.qa_chain = None  # Reset QA chain
        logger.debug(f"Vector store updated with {len(documents)} chunks")
        return True, f"Vector store updated with {len(documents)} chunks."
    except psycopg2.OperationalError as e:
        logger.error(f"Database error during vector store creation: {e}")
        return False, f"Database error during vector store creation: {e}"
    except Exception as e:
        logger.error(f"Error interacting with PGVector: {e}")
        return False, f"Error interacting with PGVector: {e}"

def setup_qa_chain():
    if not state.vectorstore:
        logger.error("Vector store not initialized.")
        return False, "Vector store not initialized."

    try:
        retriever = state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        if not GOOGLE_API_KEY or "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
            logger.error("Invalid Google API Key.")
            return False, "Invalid Google API Key."

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True
        )

        prompt_template = """
        Purpose
        You are an AI assistant designed to provide comprehensive, professional, and informative responses to questions specifically related to the 7 Stages of Growth (7SG) model for business growth. Your primary knowledge base is a collection of curated documents accessible through RAG (Retrieval-Augmented Generation). These documents include structured material derived from content relevant to the 7SG framework and associated methodologies.

        📌 Instructions for Every Response
        1.  **Primary Focus:** Prioritize and reference content retrieved from the RAG corpus (provided in the 'Context' below). All parts of your response MUST be informed by and grounded in the documents returned by the RAG. Do NOT use any prior knowledge outside of the provided context.
        2.  **Topical Scope:** You may ONLY respond to questions related to the following found within the Context:
            * The 7 Stages of Growth framework
            * Concepts, tools, and models that are part of or directly support 7SG
            * Organizational challenges, leadership strategies, or business patterns as identified through 7SG methodologies
        3.  **Out-of-Scope Topics:** If the user asks about a topic unrelated to 7SG based on the provided Context, or if the Context does not contain relevant information to answer the question:
            * Do NOT attempt to answer the question.
            * Respond ONLY with: "This assistant is designed to support topics strictly related to the 7 Stages of Growth framework, based on the provided documents. The current documents do not contain information on that topic. For other topics, we recommend seeking a different resource."
        4.  **Attribution:** Do NOT cite external authors, books, or content not found in the RAG Context. While you should base your answer *entirely* on the context, explicit inline citation like "[doc_name, section_id]" is not required unless the context itself contains such markers. Focus on synthesizing the information found.
        5.  **Tone & Style:** Maintain a professional and informative tone. Avoid informal phrasing, humor, or speculation. Be clear, structured, and thorough in your explanations, directly using information from the Context.
        6.  **Grounding:** If the context does not provide enough information to answer the question fully or accurately according to these instructions, state that the provided documents do not contain the necessary details.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        QA_PROMPT = PromptTemplate.from_template(prompt_template)
        state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        logger.debug("QA chain initialized.")
        return True, "QA chain initialized."
    except Exception as e:
        state.qa_chain = None
        logger.error(f"Error setting up QA chain: {e}")
        return False, f"Error setting up QA chain: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    db_ready, db_message = check_db_connection(CONNECTION_STRING)
    vectorstore_status = "Not initialized"
    processed_files = state.processed_files
    message = None
    message_type = None

    if db_ready:
        success, vs_message = init_vectorstore()
        if success:
            vectorstore_status = "Initialized"
        else:
            vectorstore_status = f"Error: {vs_message}"

    if request.method == 'POST':
        logger.debug(f"POST request received: form={dict(request.form)}, files={[f.filename for f in request.files.getlist('pdfs')]}")
        clear_collection = request.form.get('clear_collection') == 'on'
        pdf_files = request.files.getlist('pdfs') if 'pdfs' in request.files else []

        if pdf_files:
            valid_files = [f for f in pdf_files if allowed_file(f.filename) and f.filename]
            if not valid_files:
                message = "No valid PDF files uploaded."
                message_type = "error"
            else:
                logger.debug(f"Processing {len(valid_files)} valid files, clear_collection={clear_collection}")
                success, proc_message = process_pdfs(valid_files, clear_collection)
                if success:
                    message = proc_message
                    message_type = "success"
                    processed_files = state.processed_files
                    vectorstore_status = "Updated"
                else:
                    message = proc_message
                    message_type = "error"
        else:
            logger.debug("No files uploaded in POST request.")
            message = "No files uploaded."
            message_type = "error"

        # Redirect to avoid form resubmission
        return redirect(url_for('index', message=message, message_type=message_type))

    # Handle GET request
    message = request.args.get('message')
    message_type = request.args.get('message_type')
    logger.debug(f"GET request: uploads contains {[f for f in os.listdir(UPLOAD_FOLDER)]}")

    return render_template('index.html',
                           db_status=db_ready,
                           db_message=db_message,
                           vectorstore_status=vectorstore_status,
                           processed_files=processed_files,
                           base_url=BASE_URL,
                           message=message,
                           message_type=message_type)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        logger.error(f"Download failed: File '{filename}' not found in {UPLOAD_FOLDER}")
        return jsonify({"error": f"File '{filename}' not found."}), 404

@app.route('/api/status', methods=['GET'])
def status():
    db_ready, db_message = check_db_connection(CONNECTION_STRING)
    response = {
        "status": "running",
        "database": {
            "connected": db_ready,
            "message": db_message
        },
        "vectorstore": {
            "initialized": state.vectorstore is not None,
            "collection_name": COLLECTION_NAME,
            "processed_files": state.processed_files
        },
        "qa_chain": {
            "ready": state.qa_chain is not None
        }
    }
    return jsonify(response), 200

@app.route('/api/ask', methods=['POST'])
def ask_question():
    db_ready, db_message = check_db_connection(CONNECTION_STRING)
    if not db_ready:
        logger.error(f"Database error: {db_message}")
        return jsonify({"error": db_message}), 500

    data = request.form
    question = data.get('question')

    if not question:
        logger.error("No question provided.")
        return jsonify({"error": "Question is required."}), 400

    success, vs_message = init_vectorstore()
    if not success:
        logger.error(f"Vector store error: {vs_message}")
        return jsonify({"error": vs_message}), 500

    if not state.qa_chain:
        success, qa_message = setup_qa_chain()
        if not success:
            logger.error(f"QA chain error: {qa_message}")
            return jsonify({"error": qa_message}), 500

    try:
        result = state.qa_chain.invoke({"query": question})
        response = result['result']
        source_docs = result['source_documents']
        logger.debug(f"Retrieved {len(source_docs)} source documents: {[doc.metadata.get('source', 'No source') for doc in source_docs]}")
        sources = [
            {
                "source": doc.metadata.get('source', 'Unknown PDF'),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "download_url": f"{BASE_URL}/download/{doc.metadata.get('source', '')}"
            }
            for doc in source_docs
        ]
        if not sources:
            logger.warning("No source documents retrieved for query.")
        return jsonify({
            "question": question,
            "answer": response,
            "sources": sources
        }), 200
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": f"Error processing question: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)