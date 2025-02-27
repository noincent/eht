import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
import torch
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import CrossEncoder
from read_doc import retrieve_with_subsections_json

# Import from document processor
from document_processor import EnglishEmbedder, ChineseEmbedder, LanguageDetector


class QueryProcessor:
    """Handles query preprocessing and language detection."""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
    
    def process_query(self, query: str) -> Tuple[str, str]:
        """Process the query and detect its language."""
        # Clean query
        query = query.strip()
        
        # Detect language
        language = self.language_detector.detect_language(query)
        
        return query, language


class HybridRetriever:
    """Combines vector search with keyword search for optimal retrieval."""
    
    def __init__(self, 
                 index_path: str = "./indexes",
                 top_k: int = 10,
                 use_bm25: bool = True):
        self.index_path = index_path
        self.top_k = top_k
        self.use_bm25 = use_bm25
        
        # Initialize embedders
        self.en_embedder = EnglishEmbedder()
        self.zh_embedder = ChineseEmbedder()
        
        # Load chunks metadata
        self.en_chunks = self._load_chunks("en_chunks.json")
        self.zh_chunks = self._load_chunks("zh_chunks.json")
        
        # Build BM25 indexes (if using BM25)
        if self.use_bm25:
            self.en_bm25, self.zh_bm25 = self._build_bm25_indexes()
            
        # Load FAISS indexes
        self.en_index, self.zh_index = self._load_faiss_indexes()
    
    def _load_chunks(self, filename: str) -> List[Dict]:
        """Load chunk metadata from file."""
        filepath = os.path.join(self.index_path, filename)
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _build_bm25_indexes(self) -> Tuple[Any, Any]:
        """Build BM25 indexes for keyword search."""
        en_tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.en_chunks]
        zh_tokenized_corpus = [self._tokenize_chinese(chunk["text"]) for chunk in self.zh_chunks]
        
        en_bm25 = BM25Okapi(en_tokenized_corpus) if en_tokenized_corpus else None
        zh_bm25 = BM25Okapi(zh_tokenized_corpus) if zh_tokenized_corpus else None
        
        return en_bm25, zh_bm25
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize English text for BM25."""
        return re.findall(r'\w+', text.lower())
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """Tokenize Chinese text for BM25 (character-level for simplicity)."""
        return list(text)
    
    def _load_faiss_indexes(self) -> Tuple[Any, Any]:
        """Load FAISS indexes for vector search."""
        en_index_path = os.path.join(self.index_path, "en_index.faiss")
        zh_index_path = os.path.join(self.index_path, "zh_index.faiss")
        
        en_index = faiss.read_index(en_index_path) if os.path.exists(en_index_path) else None
        zh_index = faiss.read_index(zh_index_path) if os.path.exists(zh_index_path) else None
        
        return en_index, zh_index
    
    def retrieve(self, query: str, language: str) -> List[Dict]:
        """
        Retrieve relevant chunks using a hybrid approach.
        Combines vector search and keyword search (if enabled).
        """
        # Select appropriate resources based on language
        if language == 'en':
            index = self.en_index
            chunks = self.en_chunks
            embedder = self.en_embedder
            bm25 = self.en_bm25 if self.use_bm25 else None
            tokenize_func = self._tokenize
        else:  # 'zh'
            index = self.zh_index
            chunks = self.zh_chunks
            embedder = self.zh_embedder
            bm25 = self.zh_bm25 if self.use_bm25 else None
            tokenize_func = self._tokenize_chinese
        
        # If no index or chunks for this language, return empty results
        if index is None or not chunks:
            return []
        
        # Vector search
        query_embedding = embedder.get_embeddings([query])[0].reshape(1, -1).astype('float32')
        distances, indices = index.search(query_embedding, min(self.top_k, len(chunks)))
        
        vector_results = [
            {
                **chunks[idx],
                "vector_score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity
            }
            for i, idx in enumerate(indices[0])
            if idx < len(chunks)
        ]
        
        # BM25 keyword search (if enabled)
        if self.use_bm25 and bm25 is not None:
            query_tokens = tokenize_func(query)
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores
            max_score = max(bm25_scores) if bm25_scores.size > 0 else 1.0
            bm25_scores = bm25_scores / max_score if max_score > 0 else bm25_scores
            
            top_bm25_indices = np.argsort(bm25_scores)[-self.top_k:][::-1]
            
            bm25_results = [
                {
                    **chunks[idx],
                    "bm25_score": float(bm25_scores[idx])
                }
                for idx in top_bm25_indices
                if bm25_scores[idx] > 0  # Only include if score > 0
            ]
            
            # Combine results
            combined_results = self._combine_results(vector_results, bm25_results)
            return combined_results
        
        return vector_results
    
    def _combine_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """Combine vector and BM25 results with score fusion."""
        # Create mapping from chunk ID to result
        results_map = {}
        
        # Add vector results
        for result in vector_results:
            results_map[result["id"]] = {
                **result,
                "combined_score": result["vector_score"]
            }
        
        # Add/update with BM25 results
        for result in bm25_results:
            if result["id"] in results_map:
                # If already in results, update combined score
                results_map[result["id"]]["bm25_score"] = result["bm25_score"]
                results_map[result["id"]]["combined_score"] = (
                    results_map[result["id"]]["vector_score"] * 0.7 + 
                    result["bm25_score"] * 0.3
                )
            else:
                # If not in results yet, add it
                results_map[result["id"]] = {
                    **result,
                    "vector_score": 0.0,
                    "combined_score": result["bm25_score"] * 0.3
                }
        
        # Convert back to list and sort by combined score
        combined_results = list(results_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Take top results
        return combined_results[:self.top_k]


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model."""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 top_k: int = 5):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
    
    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder model."""
        if not results:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [(query, result["text"]) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
        # Take top results
        return reranked_results[:self.top_k]


class SmartContextBuilder:
    """Builds an optimized context for the LLM from retrieved chunks."""
    
    def __init__(self, 
                 max_tokens: int = 3000,
                 include_metadata: bool = True):
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata
        
    def build_context(self, query: str, reranked_results: List[Dict]) -> str:
        """
        Build a context string for the LLM from reranked results.
        Smartly selects and organizes content to maximize relevance while staying within token limit.
        """
        if not reranked_results:
            return ""
            
        context_parts = []
        estimated_tokens = 0
        
        # Include the most relevant sections first
        for result in reranked_results:
            section_text = result.get("text", "")
            
            # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
            section_tokens = len(section_text) // 4
            
            if estimated_tokens + section_tokens > self.max_tokens:
                # If adding this section would exceed token limit, try to truncate it
                available_tokens = self.max_tokens - estimated_tokens
                if available_tokens > 100:  # Only add if we have enough room for meaningful content
                    truncated_text = self._truncate_text(section_text, available_tokens)
                    
                    section_part = ""
                    if self.include_metadata:
                        section_part += f"ID: {result.get('id', 'unknown')}\n"
                        section_part += f"Level: {result.get('metadata', {}).get('level', 'unknown')}\n"
                        section_part += f"Section: {result.get('metadata', {}).get('section_id', 'unknown')}\n"
                    
                    section_part += f"Content: {truncated_text}\n\n"
                    context_parts.append(section_part)
                    
                    estimated_tokens += available_tokens
                    break
            else:
                # Add the whole section
                section_part = ""
                if self.include_metadata:
                    section_part += f"ID: {result.get('id', 'unknown')}\n"
                    section_part += f"Level: {result.get('metadata', {}).get('level', 'unknown')}\n"
                    section_part += f"Section: {result.get('metadata', {}).get('section_id', 'unknown')}\n"
                
                section_part += f"Content: {section_text}\n\n"
                context_parts.append(section_part)
                
                estimated_tokens += section_tokens
        
        return "".join(context_parts)
    
    def _truncate_text(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target tokens, trying to preserve complete sentences."""
        # Rough approximation: 4 chars ≈ 1 token
        target_chars = target_tokens * 4
        
        if len(text) <= target_chars:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:target_chars]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > target_chars // 2:  # Only truncate at sentence if it's not cutting off too much
            return truncated[:last_sentence_end + 1]
        
        # Fall back to truncating at word boundary
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return truncated[:last_space] + "..."
        
        # Last resort: just truncate
        return truncated + "..."


class HandbookRetriever:
    """Main class that orchestrates the entire retrieval process."""
    
    def __init__(self, index_path: str = "./indexes"):
        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever(index_path=index_path)
        self.reranker = CrossEncoderReranker()
        self.context_builder = SmartContextBuilder()
    
    def process_query(self, query: str) -> Dict:
        """
        Process a query and retrieve relevant handbook sections.
        Returns a dictionary with query info, language, and relevant sections with context.
        """
        # Process query and detect language
        processed_query, language = self.query_processor.process_query(query)
        
        if language == 'en':
            # Retrieve initial results
            retrieval_results = self.retriever.retrieve(processed_query, language)
            
            # Rerank results
            reranked_results = self.reranker.rerank(processed_query, retrieval_results)
            
            # Build context for LLM
            context = self.context_builder.build_context(processed_query, reranked_results)
            
            return {
                "query": processed_query,
                "language": language,
                "results": reranked_results,
                "context": context
            }
        else:
            # zh use backup model
            retrieval_results = retrieve_with_subsections_json(processed_query)
            reranked_results = self.reranker.rerank(processed_query, retrieval_results)
            
            # Build context for LLM
            context = str(reranked_results)
            return {
                "query": processed_query,
                "language": language,
                "results": reranked_results,
                "context": context
            }

# Example usage
if __name__ == "__main__":
    retriever = HandbookRetriever()
    result = retriever.process_query("What is the vacation policy?")
    print(f"Retrieved {len(result['results'])} relevant sections")
    print(f"Context length: {len(result['context'])}")