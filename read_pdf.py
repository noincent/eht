import os
import base64
import logging
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing using Claude's native PDF handling capabilities."""
    
    def __init__(self):
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def load_pdf(self, file_path: str) -> Optional[str]:
        """Load a PDF file and encode it as base64."""
        try:
            with open(file_path, "rb") as f:
                pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
            return pdf_data
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return None
    
    def extract_full_handbook_text(self, pdf_data: str, language: str) -> str:
        """Extract the full text content from a PDF handbook using Claude."""
        try:
            # Choose prompt based on language
            if language == "zh":
                prompt = "请将这份PDF文档中的所有文本提取出来，保持原始的结构和格式。注意表格的内容也要正确提取。"
            else:
                prompt = "Please extract all text from this PDF document, maintaining the original structure and format. Make sure to properly extract content from tables as well."
            
            # Create message with PDF attachment
            message = self.client.messages.create(
                model=os.getenv("LLM_MODEL", "latest"),
                max_tokens=20000,  # High token limit to handle full document
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            )
            
            # Extract text from response
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_handbook(self, pdf_path: str, language: str) -> Optional[str]:
        """Process a handbook PDF and return its full text content."""
        logger.info(f"Processing {language} handbook PDF: {pdf_path}")
        
        # Load PDF
        pdf_data = self.load_pdf(pdf_path)
        if not pdf_data:
            logger.error(f"Failed to load PDF: {pdf_path}")
            return None
        
        # Extract text
        text_content = self.extract_full_handbook_text(pdf_data, language)
        if not text_content:
            logger.error(f"Failed to extract text from PDF: {pdf_path}")
            return None
        
        logger.info(f"Successfully processed {language} handbook PDF")
        return text_content

class HandbookPDFHandler:
    """Main interface for handling handbook PDFs in the application."""
    
    def __init__(self, cache_dir: str = "./pdf_cache"):
        self.processor = PDFProcessor()
        self.cache_dir = cache_dir
        self.en_handbook_path = os.getenv("HANDBOOK_PATH_EN_PDF", "employee_handbook_en.pdf")
        self.zh_handbook_path = os.getenv("HANDBOOK_PATH_ZH_PDF", "employee_handbook_zh.pdf")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_path(self, language: str) -> str:
        """Get the path to the cached content file."""
        return os.path.join(self.cache_dir, f"handbook_{language}.txt")
    
    def get_handbook_content(self, language: str) -> Optional[str]:
        """Get the handbook content for the specified language, using cache if available."""
        cache_path = self._get_cache_path(language)
        
        # Check cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content:
                    logger.info(f"Using cached {language} handbook content")
                    return content
            except Exception as e:
                logger.warning(f"Error reading cached content: {e}")
        
        # Process PDF if cache doesn't exist or is invalid
        pdf_path = self.en_handbook_path if language == "en" else self.zh_handbook_path
        content = self.processor.process_handbook(pdf_path, language)
        
        # Cache the content
        if content:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Error writing cache file: {e}")
        
        return content
    
    def clear_cache(self, language: Optional[str] = None) -> None:
        """Clear the cached content for the specified language or all languages."""
        if language:
            cache_path = self._get_cache_path(language)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for {language} handbook")
        else:
            # Clear all caches
            for lang in ["en", "zh"]:
                cache_path = self._get_cache_path(lang)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            logger.info("Cleared all handbook caches")

# Example usage
if __name__ == "__main__":
    # Test PDF processing
    handler = HandbookPDFHandler()
    
    # Process English handbook
    en_content = handler.get_handbook_content("en")
    if en_content:
        print(f"English handbook length: {len(en_content)} characters")
        print(f"First 500 chars: {en_content[:500]}...")
    
    # Process Chinese handbook
    zh_content = handler.get_handbook_content("zh")
    if zh_content:
        print(f"Chinese handbook length: {len(zh_content)} characters")
        print(f"First 500 chars: {zh_content[:500]}...")