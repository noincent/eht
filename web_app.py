from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import logging
import time
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from document_processor import DocumentProcessor
from retrieval_system import HandbookRetriever
from llm_handler import HandbookLLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
HANDBOOK_PATH_ZH = os.getenv("HANDBOOK_PATH_ZH", "employee_handbook_zh.docx")
HANDBOOK_PATH_EN = os.getenv("HANDBOOK_PATH_EN", "employee_handbook_en.docx")
INDEX_PATH = os.getenv("INDEX_PATH", "./indexes")
CACHE_DIR = os.getenv("CACHE_DIR", "./llm_cache")
FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", "./feedback")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

# Create required directories
for directory in [INDEX_PATH, CACHE_DIR, FEEDBACK_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # For session

# Initialize our services
document_processor = DocumentProcessor(index_path=INDEX_PATH)
retriever = HandbookRetriever(index_path=INDEX_PATH)
llm_service = HandbookLLMService(cache_dir=CACHE_DIR)


def initialize_handbook() -> bool:
    """Initialize or reinitialize the handbook processing."""
    try:
        # Check if indexes exist
        if not os.path.exists(os.path.join(INDEX_PATH, "en_index.faiss")) or \
           not os.path.exists(os.path.join(INDEX_PATH, "zh_index.faiss")):
            logger.info("Initializing handbook indexes...")
            document_processor.process_documents(
                zh_doc_path=HANDBOOK_PATH_ZH, 
                en_doc_path=HANDBOOK_PATH_EN
            )
            logger.info("Handbook indexes created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error initializing handbook: {str(e)}")
        return False


def log_query(query_data: Dict) -> None:
    """Log query and response for analysis."""
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(LOG_DIR, f"query_log_{timestamp}.json")
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(query_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Error logging query: {str(e)}")


def save_feedback(feedback_data: Dict) -> None:
    """Save user feedback for future improvements."""
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        feedback_file = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
        
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Error saving feedback: {str(e)}")


@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')


@app.route('/reload-handbook', methods=["POST"])
def reload_handbook():
    """Endpoint to reload the handbook indexes."""
    try:
        # Get parameters - allow specifying which languages to reload
        langs = request.args.get("langs", "zh,en").split(",")
        
        # Remove existing indexes
        for file in os.listdir(INDEX_PATH):
            file_path = os.path.join(INDEX_PATH, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Prepare document paths
        zh_path = HANDBOOK_PATH_ZH if 'zh' in langs else None
        en_path = HANDBOOK_PATH_EN if 'en' in langs else None
        
        # Process appropriate documents
        zh_count, en_count = document_processor.process_documents(
            zh_doc_path=zh_path,
            en_doc_path=en_path
        )
        
        # Check if processing was successful
        if (zh_path and zh_count == 0) or (en_path and en_count == 0):
            message = "Warning: Some documents were not processed correctly."
            status = "warning"
        else:
            message = f"Handbook reloaded successfully: {zh_count} Chinese and {en_count} English chunks"
            status = "success"
            
        return jsonify({"status": status, "message": message}), 200
    except Exception as e:
        logger.error(f"Error reloading handbook: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500


@app.route('/query', methods=["GET"])
def query_handbook():
    """Handle handbook queries and return smart responses."""
    try:
        # Get parameters
        query = request.args.get("q", "")
        lang = request.args.get("lang", "en")  # Default to English
        
        # Store in session for potential feedback
        session["last_query"] = query
        session["last_lang"] = lang
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Ensure indexes are initialized based on query language
        expected_index = "en_index.faiss" if lang == "en" else "zh_index.faiss"
        if not os.path.exists(os.path.join(INDEX_PATH, expected_index)):
            logger.info(f"Index for {lang} not found. Initializing...")
            if lang == "en" and os.path.exists(HANDBOOK_PATH_EN):
                document_processor.process_documents(en_doc_path=HANDBOOK_PATH_EN)
            elif lang == "zh" and os.path.exists(HANDBOOK_PATH_ZH):
                document_processor.process_documents(zh_doc_path=HANDBOOK_PATH_ZH)
            else:
                error_msg = "Handbook not found for this language" if lang == "en" else "找不到该语言的员工手册"
                return jsonify({"error": error_msg}), 404
        
        # Process query
        start_time = time.time()
        
        # Step 1: Retrieve relevant sections
        retrieval_result = retriever.process_query(query)
        retrieved_context = retrieval_result.get("context", "")
        language = retrieval_result.get("language", lang)  # Use detected language
        
        # Step 2: Generate LLM response
        llm_result = llm_service.get_handbook_answer(query, retrieved_context, language)
        
        processing_time = time.time() - start_time
        
        # Build response object
        response = {
            "query": query,
            "language": language,
            "answer": llm_result.get("response", ""),
            "relevant_sections": retrieval_result.get("results", []),
            "suggested_followups": llm_result.get("suggested_followups", []),
            "processing_time": processing_time
        }
        
        # Log the query and response
        log_query({
            "query": query,
            "language": language,
            "retrieval_results": retrieval_result.get("results", []),
            "llm_response": llm_result,
            "processing_time": processing_time
        })
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_msg = f"Error processing query: {str(e)}" if lang == "en" else f"处理查询时出错：{str(e)}"
        return jsonify({"error": error_msg}), 500


@app.route('/feedback', methods=["POST"])
def submit_feedback():
    """Endpoint to collect user feedback on responses."""
    try:
        data = request.json
        
        # Get the query from session if available
        query = session.get("last_query", "Unknown query")
        lang = session.get("last_lang", "en")
        
        feedback_data = {
            "query": query,
            "language": lang,
            "rating": data.get("rating"),
            "comments": data.get("comments", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_feedback(feedback_data)
        
        success_msg = "Thank you for your feedback!" if lang == "en" else "感谢您的反馈！"
        return jsonify({"status": "success", "message": success_msg}), 200
        
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        error_msg = f"Error saving feedback: {str(e)}" if lang == "en" else f"保存反馈时出错：{str(e)}"
        return jsonify({"error": error_msg}), 500


# Endpoint for status check
@app.route('/handbook-status')
def handbook_status():
    """Get the status of handbook documents and indexes."""
    try:
        zh_doc_exists = os.path.exists(HANDBOOK_PATH_ZH)
        en_doc_exists = os.path.exists(HANDBOOK_PATH_EN)
        zh_index_exists = os.path.exists(os.path.join(INDEX_PATH, "zh_index.faiss"))
        en_index_exists = os.path.exists(os.path.join(INDEX_PATH, "en_index.faiss"))
        
        status = {
            "documents": {
                "zh": {
                    "exists": zh_doc_exists,
                    "path": HANDBOOK_PATH_ZH
                },
                "en": {
                    "exists": en_doc_exists,
                    "path": HANDBOOK_PATH_EN
                }
            },
            "indexes": {
                "zh": zh_index_exists,
                "en": en_index_exists
            }
        }
        
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error checking handbook status: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure handbook is initialized
    initialize_handbook()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8007, debug=True)