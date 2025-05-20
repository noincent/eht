# Employee Handbook Tool

A bilingual application for querying employee handbooks using generative AI. This tool allows users to ask questions about company policies in either English or Chinese and get relevant, accurate responses based on the content of the handbook.

## Features

- **Bilingual Support**: Process and query handbooks in both English and Chinese
- **AI-Powered Responses**: Uses LLM technology to generate natural responses based on handbook content
- **Smart Retrieval System**: Finds the most relevant sections of the handbook for each query
- **Web Interface**: Simple, intuitive user interface for asking questions
- **Follow-up Suggestions**: AI suggests relevant follow-up questions
- **User Feedback Collection**: Gather feedback on response quality
- **PDF Support**: Extract text from PDF handbooks for better table handling

## Setup

### Prerequisites

- Python 3.10+
- Microsoft Word documents (.docx) or PDF files for your employee handbook(s)
- Anthropic API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Employee_Handbook_Tool.git
   cd Employee_Handbook_Tool
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and configure your settings:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` to add your Anthropic API key and handbook paths:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   HANDBOOK_PATH_ZH=path/to/chinese_handbook.docx
   HANDBOOK_PATH_EN=path/to/english_handbook.docx
   HANDBOOK_PATH_ZH_PDF=path/to/chinese_handbook.pdf  # Optional
   HANDBOOK_PATH_EN_PDF=path/to/english_handbook.pdf  # Optional
   ```

### Running the Application

1. Start the web server:
   ```bash
   python web_app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:8007
   ```

3. Use the interface to ask questions about the employee handbook in either English or Chinese.

## System Components

### 1. Document Processing

The `document_processor.py` module handles:
- Parsing Word documents
- Chunking text into searchable segments
- Generating embeddings for semantic search
- Building search indexes

### 2. Retrieval System

The `retrieval_system.py` module:
- Detects query language
- Performs hybrid search (vector + keyword)
- Reranks search results
- Builds optimized context for the LLM

### 3. LLM Handler

The `llm_handler.py` module:
- Manages Anthropic API calls
- Generates natural language responses
- Suggests follow-up questions
- Handles response caching

### 4. PDF Handler
The `read_pdf.py` module:
- Extracts text from PDF versions of handbooks
- Provides better handling of tables and complex formatting
- Maintains cache for efficiency

### 5. Web Application

The `web_app.py` module:
- Provides a web interface
- Coordinates between all other components
- Handles user sessions and feedback
- Manages handbook reloading

## Customization

### Adding/Changing Handbooks

1. Replace the handbook files specified in your `.env`
2. Click "Reload Handbook" in the web interface or restart the application

### Modifying the UI

Edit the HTML template in `templates/index.html` to customize the user interface.

### Changing Embedding Models

To use different embedding models:
1. Update the `EnglishEmbedder` or `ChineseEmbedder` classes in `document_processor.py`
2. Specify different model names in the class initialization

### Adjusting Retrieval Parameters

Fine-tune search behavior by modifying:
- `top_k` parameters in `HybridRetriever` and `CrossEncoderReranker`
- BM25 and vector search weights in `_combine_results`
- Context building in `SmartContextBuilder`

## Troubleshooting

### Common Issues

1. **Missing Indexes**: If you see a "Handbook not found" error, check that your handbook files exist at the paths specified in `.env`.

2. **API Errors**: Verify your Anthropic API key is correct in the `.env` file.

3. **Package Compatibility**: If you encounter dependency issues, try installing the exact versions specified in `requirements.txt`.

4. **Model Loading Errors**: If embedding models fail to load, ensure you have enough memory and disk space.

## License

[MIT License](LICENSE)

## Acknowledgements

- Uses Anthropic's API for natural language understanding
- Uses sentence-transformers for embedding generation
- Uses FAISS for efficient vector search