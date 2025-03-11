from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import logging
import time
import json
import uuid
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
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # Session lasts for 24 hours

# Initialize our services
document_processor = DocumentProcessor(index_path=INDEX_PATH)
retriever = HandbookRetriever(index_path=INDEX_PATH)
llm_service = HandbookLLMService(cache_dir=CACHE_DIR)

# Chat session storage
chat_sessions = {}

# Bert/Jina backup model for chinese

from read_doc import extract_hierarchical_structure, store_hierarchical_embeddings


def initialize_handbook() -> bool:
    """Initialize or reinitialize the handbook processing."""
    try:
        # Check if indexes exist
        if not os.path.exists(os.path.join(INDEX_PATH, "en_index.faiss")) or \
           not os.path.exists(os.path.join(INDEX_PATH, "zh_index.faiss")) or \
           not os.path.exists(os.path.join(INDEX_PATH, "contract_embeddings.faiss")):
            logger.info("Initializing handbook indexes...")
            document_processor.process_documents(
                zh_doc_path=HANDBOOK_PATH_ZH, 
                en_doc_path=HANDBOOK_PATH_EN
            )
            logger.info("Handbook indexes created successfully.")
            sections = extract_hierarchical_structure(HANDBOOK_PATH_ZH)
            store_hierarchical_embeddings(sections)
            logger.info("Backup indexes created successfully.")
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
            #zh_doc_path=zh_path,
            en_doc_path=en_path
        )

        zh_count = extract_hierarchical_structure(zh_path)
        store_hierarchical_embeddings(zh_path)
        
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


def get_or_create_chat_session():
    """Get existing chat session or create a new one."""
    if 'chat_session_id' not in session:
        session_id = str(uuid.uuid4())
        session['chat_session_id'] = session_id
        chat_sessions[session_id] = {
            'messages': [],
            'language': 'en'
        }
    
    return chat_sessions.get(session['chat_session_id'])


@app.route('/query', methods=["GET"])
def query_handbook():
    """Handle handbook queries and return smart responses."""
    try:
        # Get parameters
        query = request.args.get("q", "")
        lang = request.args.get("lang", "en")  # Default to English
        chat_id = request.args.get("chat_id", None)  # Optional chat session ID
        
        # Store in session for potential feedback
        session["last_query"] = query
        session["last_lang"] = lang
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Get or initialize chat session
        chat_session = get_or_create_chat_session()
        chat_session['language'] = lang
        
        # If chat_id is provided and matches a different session, load that session instead
        if chat_id and chat_id != session.get('chat_session_id') and chat_id in chat_sessions:
            session['chat_session_id'] = chat_id
            chat_session = chat_sessions[chat_id]
        
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
        
        # Add message to chat history
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": time.time()
        }
        
        assistant_message = {
            "role": "assistant",
            "content": llm_result.get("response", ""),
            "relevant_sections": retrieval_result.get("results", []),
            "suggested_followups": llm_result.get("suggested_followups", []),
            "timestamp": time.time()
        }
        
        chat_session['messages'].append(user_message)
        chat_session['messages'].append(assistant_message)
        
        # Build response object
        response = {
            "query": query,
            "language": language,
            "answer": llm_result.get("response", ""),
            "relevant_sections": retrieval_result.get("results", []),
            "suggested_followups": llm_result.get("suggested_followups", []),
            "processing_time": processing_time,
            "chat_id": session.get('chat_session_id'),
            "chat_history": chat_session['messages']
        }
        
        # Log the query and response
        log_query({
            "query": query,
            "language": language,
            "retrieval_results": retrieval_result.get("results", []),
            "llm_response": llm_result,
            "processing_time": processing_time,
            "chat_id": session.get('chat_session_id')
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


@app.route('/chat-sessions', methods=["GET"])
def list_chat_sessions():
    """List all chat sessions for the current user."""
    try:
        current_session_id = session.get('chat_session_id')
        
        # Get active sessions for this user
        result = []
        for session_id, chat_data in chat_sessions.items():
            if len(chat_data['messages']) > 0:
                # Get first user message to use as title
                first_user_message = next((msg for msg in chat_data['messages'] if msg['role'] == 'user'), None)
                title = first_user_message['content'] if first_user_message else "New chat"
                
                # Truncate long titles
                if len(title) > 50:
                    title = title[:47] + "..."
                
                result.append({
                    "id": session_id,
                    "title": title,
                    "language": chat_data['language'],
                    "message_count": len(chat_data['messages']),
                    "last_updated": max(msg['timestamp'] for msg in chat_data['messages']) if chat_data['messages'] else 0,
                    "is_current": session_id == current_session_id
                })
        
        # Sort by last updated time (newest first)
        result.sort(key=lambda x: x['last_updated'], reverse=True)
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error listing chat sessions: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat-sessions/<session_id>', methods=["GET"])
def get_chat_session(session_id):
    """Get a specific chat session by ID."""
    try:
        if session_id in chat_sessions:
            # Set as current session
            session['chat_session_id'] = session_id
            
            return jsonify({
                "id": session_id,
                "language": chat_sessions[session_id]['language'],
                "messages": chat_sessions[session_id]['messages']
            }), 200
        else:
            return jsonify({"error": "Chat session not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving chat session: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/new-chat', methods=["POST"])
def create_new_chat():
    """Create a new chat session."""
    try:
        # Create new session
        session_id = str(uuid.uuid4())
        session['chat_session_id'] = session_id
        
        lang = request.json.get('language', 'en') if request.json else 'en'
        
        chat_sessions[session_id] = {
            'messages': [],
            'language': lang
        }
        
        return jsonify({
            "id": session_id,
            "language": lang,
            "messages": []
        }), 201
    except Exception as e:
        logger.error(f"Error creating new chat session: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Ensure handbook is initialized
    initialize_handbook()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8007, debug=False)