from typing import Dict, List, Optional, Any
import os
import json
from anthropic import Anthropic
import re
import time
import hashlib
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptBuilder:
    """Creates optimized prompts for the LLM."""
    
    def __init__(self):
        pass
    
    def build_system_prompt(self, language: str) -> str:
        """Build a system prompt for the LLM to answer handbook queries."""
        if language == "zh":
            return """你是一个员工手册助手，负责准确、全面地回答关于公司政策和规定的问题。

请遵循以下指引：
1. 只基于提供的员工手册中的信息回答问题
2. 如果手册中没有提及某个主题，清楚地说明这点
3. 不要添加虚构的信息或个人意见
4. 直接引用员工手册的相关条款以支持你的答案
5. 以简明清晰的中文回应
6. 在回答中加入引用的具体章节标题，便于用户参考

当员工手册的信息不足以完全回答问题时，你可以：
- 明确指出哪些信息是手册中直接提供的
- 建议用户与HR部门联系以获取更完整的信息
- 提供可能的后续问题以帮助用户进一步了解主题"""
        else:
            return """You are an Employee Handbook Assistant, responsible for answering questions about company policies and regulations accurately and comprehensively.

Please follow these guidelines:
1. Answer questions solely based on the information provided in the employee handbook
2. Clearly state if a topic is not mentioned in the handbook
3. Do not add fictional information or personal opinions
4. Directly quote relevant provisions from the employee handbook to support your answers
5. Respond in clear and concise English
6. Include references to specific section titles in your answer for user reference

When the handbook information is insufficient to fully answer a question, you may:
- Clearly indicate which information is directly provided in the handbook
- Suggest that the user contact the HR department for more complete information
- Provide possible follow-up questions to help the user learn more about the topic"""
    
    def build_user_prompt(self, query: str, context: str, language: str) -> str:
        """Build a user prompt with the query and context."""
        if language == "zh":
            return f"""请回答以下关于员工手册的问题：

{query}

以下是员工手册中的相关内容：

{context}

基于以上信息，请提供全面而准确的回答，包含相关章节的引用。如果无法从以上信息中找到答案，请明确说明。"""
        else:
            return f"""Please answer the following question about the employee handbook:

{query}

Here are the relevant sections from the employee handbook:

{context}

Based on this information, please provide a comprehensive and accurate answer, including references to the relevant sections. If the answer cannot be found in the information provided, please state this clearly."""


class ResponseGenerator:
    """Handles LLM API calls and processes responses."""
    
    def __init__(self, 
                 model: str = None,
                 cache_dir: str = None,
                 use_cache: bool = True):
        self.model = model or os.getenv("LLM_MODEL", "claude-3-7-sonnet-20250219")
        try:
            # Use the official Anthropic client initialization
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
                
            self.client = anthropic.Anthropic(api_key=api_key)
            print("✅ Successfully initialized Anthropic client")
                
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            # Create a dummy client for testing without API
            from types import SimpleNamespace
            self.client = SimpleNamespace()
            self.client.messages = SimpleNamespace()
            self.client.messages.create = lambda **kwargs: SimpleNamespace(content=[SimpleNamespace(text="API Error: Could not connect to Anthropic API")])
            print("⚠️ Using dummy Anthropic client - responses will be placeholder only")
            
        self.prompt_builder = PromptBuilder()
        self.cache_dir = cache_dir or os.getenv("CACHE_DIR", "./response_cache")
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, query: str, context: str, language: str) -> str:
        """Generate a cache key for a specific query and context."""
        content = f"{query}|{context}|{language}|{self.model}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Try to get a cached response."""
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
                return None
                
        return None
    
    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache a response for future use."""
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error writing cache file: {e}")
    
    def generate_response(self, query: str, context: str, language: str, 
                         max_retries: int = 3, retry_delay: int = 2) -> Dict:
        """Generate a response using the LLM, with retries and caching."""
        # Try to get from cache first
        cache_key = self._get_cache_key(query, context, language)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Using cached response for query: {query}")
            return cached_response
        
        # Build prompts
        system_prompt = self.prompt_builder.build_system_prompt(language)
        user_prompt = self.prompt_builder.build_user_prompt(query, context, language)
        
        # Get configuration from environment variables
        max_tokens = int(os.getenv("MAX_TOKENS", 1500))
        temperature = float(os.getenv("TEMPERATURE", 0.1))
        
        # Make API call with retries
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        
        if response:
            # Extract and process the response
            response_text = self._extract_response_text(response)
            
            # Structured response
            result = {
                "query": query,
                "language": language,
                "response": response_text,
                "sections_referenced": self._extract_section_references(response_text, language),
                "model": self.model,
                "timestamp": time.time()
            }
            
            # Cache the response
            self._cache_response(cache_key, result)
            
            return result
        
        # If we reached here, all retries failed
        return {
            "query": query,
            "language": language,
            "response": "Sorry, I couldn't generate a response at this time." if language == "en" else 
                        "抱歉，我现在无法生成回答。",
            "error": True
        }
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from the LLM response object."""
        try:
            return response.content[0].text
        except (AttributeError, IndexError, KeyError):
            # Fall back to string representation if structure not as expected
            return str(response)
    
    def _extract_section_references(self, text: str, language: str) -> List[str]:
        """Extract section references from the response text."""
        if language == "zh":
            # Pattern for Chinese section references (customize based on your handbook structure)
            pattern = r'(?:第[一二三四五六七八九十]+章|[一二三四五六七八九十]+、|\d+[\.、]|[（(]\d+[）)])[^，。；\n]+'
        else:
            # Pattern for English section references (customize based on your handbook structure)
            pattern = r'(?:Chapter|Section|Article)\s+[\w\d\.\-]+(?:\s*:\s*[^\.]+)?'
            
        matches = re.findall(pattern, text)
        # Remove duplicates and return
        return list(set(matches))


class FollowupSuggester:
    """Generates intelligent follow-up question suggestions."""
    
    def __init__(self, llm_client: ResponseGenerator):
        self.llm_client = llm_client
    
    def suggest_followups(self, query: str, response_text: str, language: str, 
                          max_suggestions: int = 3) -> List[str]:
        """Generate follow-up question suggestions based on the query and response."""
        # Build a prompt for the LLM to generate follow-up questions
        if language == "zh":
            prompt = f"""基于以下员工对员工手册的问题和得到的回答，请建议{max_suggestions}个合理的后续问题。这些问题应该能帮助员工更深入地了解相关政策。请只列出问题，不要包含解释或其他内容。

原始问题：{query}

回答：{response_text}

建议的后续问题："""
        else:
            prompt = f"""Based on the following employee question about the handbook and the response provided, suggest {max_suggestions} reasonable follow-up questions. These questions should help the employee gain a deeper understanding of the relevant policies. Please list only the questions without explanations or other content.

Original question: {query}

Response: {response_text}

Suggested follow-up questions:"""
        
        # Simplified context for follow-up generation
        empty_context = ""
        
        try:
            result = self.llm_client.generate_response(prompt, empty_context, language)
            
            # Parse the response to extract the questions
            response = result.get("response", "")
            
            # Extract questions using patterns
            if language == "zh":
                questions = re.findall(r'\d+[\.。、]?\s*(.+?)(?=\d+[\.。、]|\n|$)', response)
            else:
                questions = re.findall(r'\d+[\.)]?\s*(.+?)(?=\d+[\.)]|\n|$)', response)
            
            # Clean up questions
            questions = [q.strip() for q in questions if q.strip()]
            
            # If extraction failed, try a simpler approach: split by newlines
            if not questions:
                questions = [line.strip() for line in response.split('\n') 
                             if line.strip() and '?' in line]
            
            # Limit to max_suggestions
            return questions[:max_suggestions]
        except Exception as e:
            logger.warning(f"Error generating follow-up suggestions: {e}")
            return []


class HandbookLLMService:
    """Main service class that orchestrates LLM operations."""
    
    def __init__(self, cache_dir: str = "./llm_cache"):
        self.response_generator = ResponseGenerator(cache_dir=cache_dir)
        self.followup_suggester = FollowupSuggester(self.response_generator)
    
    def get_handbook_answer(self, query: str, context: str, language: str) -> Dict:
        """
        Generate a comprehensive answer to a handbook query.
        Returns a dictionary with the response and suggested follow-ups.
        """
        # Generate the main response
        result = self.response_generator.generate_response(query, context, language)
        
        # Generate follow-up suggestions
        followups = self.followup_suggester.suggest_followups(
            query, result.get("response", ""), language
        )
        
        # Add follow-ups to the result
        result["suggested_followups"] = followups
        
        return result


# Example usage
if __name__ == "__main__":
    # API key is loaded from .env file
    
    service = HandbookLLMService()
    
    # Example query and context
    query = "How many vacation days do I get per year?"
    context = """
    ID: 12345
    Level: section
    Section: Employee Benefits
    Content: Full-time employees are eligible for paid vacation time. Employees accrue vacation days based on their length of service with the company:
    - 0-1 years of service: 10 days per year (accrued at 0.83 days per month)
    - 1-5 years of service: 15 days per year (accrued at 1.25 days per month)
    - 5+ years of service: 20 days per year (accrued at 1.67 days per month)
    
    Vacation time must be approved by your manager at least two weeks in advance. Unused vacation days may be carried over to the next calendar year, up to a maximum of 5 days.
    """
    
    result = service.get_handbook_answer(query, context, "en")
    
    print(f"Response: {result['response']}")
    print("\nSuggested follow-up questions:")
    for i, followup in enumerate(result["suggested_followups"], 1):
        print(f"{i}. {followup}")