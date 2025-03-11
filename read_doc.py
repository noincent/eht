from docx import Document
import re
import faiss
import pickle
import numpy as np
import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

EMBEDDINGS_DB = "indexes/contract_embeddings.faiss"
TEXT_DB = "indexes/contract_text.pkl"
ID_MAP = "indexes/faiss_id_map.pkl"


class SectionNode:
    """Represents a section in the contract with children subsections."""

    def __init__(self, title, parent=None, _id=0):
        self.id = _id
        self.title = title
        self.content = []
        self.children = []
        self.parent = parent  # Link to the parent section
        self.context = ""

    def add_child(self, title, _id):
        """Adds a subsection to the current section."""
        child = SectionNode(title, parent=self, _id=_id)
        self.children.append(child)
        return child

    def add_content(self, text):
        """Adds content to the current section."""
        self.content.append(text)

    def add_context(self, context):
        """Stores surrounding context for this section."""
        self.context = context

    def get_summary(self):
        """Returns the first 10 words of each subsection under this section."""
        summaries = []
        for child in self.children:
            if child.title:
                preview = " ".join(child.content[0].split()[:10]) + "..." if child.content else ''
                summaries.append(f"{child.title}: {preview}")
        return summaries

    def __repr__(self):
        return f"SectionNode(id={self.id}, title={self.title}, children={len(self.children)})"


class UniqueIDGenerator:
    """Generates unique IDs for section nodes."""

    def __init__(self):
        self.current_id = -1

    def get_next_id(self):
        self.current_id += 1
        return self.current_id


def get_embedding(text):
    """Enhanced embedding generation for Chinese text."""
    # Clean and normalize the text
    text = text.strip()
    if not text:
        return np.zeros(768)  # Return zero vector for empty text

    # Tokenize with handling of Chinese characters
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        add_special_tokens=True
    )

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling of last hidden states
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Convert to numpy and normalize
    embedding = embedding.cpu().numpy().flatten()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding

def get_section_level(text):
    """Helper function to determine the level of a section heading."""
    if not text:
        return None

    # Chapter level (highest)
    if re.match(r'^第[一二三四五六七八九十]+章', text):
        return 1

    # Main section level
    if re.match(r'^[一二三四五六七八九十]+、', text):
        return 2

    if re.match(r'^（[一二三四五六七八九十]+）', text):
        return 2.5

    # Numbered sections
    if re.match(r'^\d+[\.、]', text):
        return 3

    # Parenthesized sections
    if re.match(r'^[（(]\d+[）)]', text):
        return 4

    return None

def determine_hierarchy(text, current_node):
    """Determines if the paragraph is a subsection, same level, or should move up."""
    # Improved patterns for Chinese document structure
    patterns = {
        'chapter': re.compile(r'^第[一二三四五六七八九十]+章\s*(.+)$'),
        'section': re.compile(r'^[一二三四五六七八九十]+、\s*(.+)$'),
        'subsection': re.compile(r'^（[一二三四五六七八九十]+）\s*(.+)$'),
        'numbered': re.compile(r'^(\d+)([\.、])\s*(.+)$'),
        'parenthesized': re.compile(r'^[（(](\d+)[）)]\s*(.+)$')
    }

    # Extract current level information
    current_level = get_section_level(current_node.title)
    new_level = get_section_level(text)

    # If it's a chapter heading
    if patterns['chapter'].match(text):
        return 'chapter'

    # If we can't determine the hierarchy, it's probably content
    if new_level is None:
        return 'content'

    # Compare levels
    if current_level is None:
        return 'subsection'
    elif new_level > current_level:
        return 'subsection'
    elif new_level < current_level:
        return 'up'
    else:
        return 'same'


def extract_hierarchical_structure(doc_path):
    """Enhanced parser for Chinese employee handbook structure."""
    doc = Document(doc_path)
    id_generator = UniqueIDGenerator()
    root = SectionNode("Handbook Root", _id=id_generator.get_next_id())
    current_node = root
    chapter_node = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or text.isdigit():
            continue
        
        hierarchy_type = determine_hierarchy(text, current_node)

        if hierarchy_type == 'chapter':
            # Start a new chapter
            chapter_node = root.add_child(text, id_generator.get_next_id())
            current_node = chapter_node
        elif hierarchy_type == 'subsection':
            # Add as child of current node
            current_node = current_node.add_child(text, id_generator.get_next_id())
        elif hierarchy_type == 'same':
            # Add as sibling at same level
            if current_node.parent:
                current_node = current_node.parent.add_child(text, id_generator.get_next_id())
        elif hierarchy_type == 'up':
            # Move up the tree and add as sibling
            while current_node.parent and get_section_level(current_node.title) > get_section_level(text):
                current_node = current_node.parent
            # Check if parent exists before adding
            if current_node.parent:
                current_node = current_node.parent.add_child(text, id_generator.get_next_id())
            else:
                # Add to root if no parent exists
                current_node = root.add_child(text, id_generator.get_next_id())
        else:  # content
            current_node.add_content(text)

    return root


def store_hierarchical_embeddings(root):
    """Enhanced embedding storage with better context handling."""
    dimension = 768  # BERT embedding size
    index = faiss.IndexFlatL2(dimension)
    section_data = {}
    faiss_id_to_node_id = []

    def traverse_and_store(node, parent_context=""):
        # Combine title and content for better context
        node_text = f"{node.title}\n{' '.join(node.content)}"

        # Include parent context for better relevance
        full_context = f"{parent_context}\n{node_text}" if parent_context else node_text

        # Generate embedding for the full context
        embedding = get_embedding(node_text)
        index.add(np.array([embedding]).astype("float32"))
        faiss_id_to_node_id.append(node.id)

        # Store section data with enhanced context
        section_data[node.id] = {
            "id": node.id,
            "title": node.title,
            "content": node.content,
            "context": full_context,
            "subsections": [child.id for child in node.children],
            "summary": node.get_summary(),
            "parent": node.parent.id if node.parent else None
        }

        # Recursively process children with updated context
        for child in node.children:
            traverse_and_store(child, full_context)

    traverse_and_store(root)
    print_tree(section_data)

    # Save to files
    faiss.write_index(index, EMBEDDINGS_DB)
    with open(TEXT_DB, "wb") as f:
        pickle.dump(section_data, f)
    with open(ID_MAP, "wb") as f:
        pickle.dump(faiss_id_to_node_id, f)

    print(f"✅ Stored {len(section_data)} sections with hierarchical structure.")


def retrieve_with_subsections_json(query, index_path=EMBEDDINGS_DB, data_path=TEXT_DB, id_map_path=ID_MAP, top_k=5, distance_threshold=1):
    """Enhanced retrieval function with better Chinese text handling."""
    if not (os.path.exists(index_path) and os.path.exists(data_path) and os.path.exists(id_map_path)):
        return None

    # Load stored data
    index = faiss.read_index(index_path)
    with open(data_path, "rb") as f:
        section_data = pickle.load(f)
    with open(id_map_path, "rb") as f:
        faiss_id_to_node_id = pickle.load(f)

    # Enhanced query preprocessing for Chinese
    query = query.strip()

    # Generate query embedding
    query_embedding = get_embedding(query)
    query_embedding = query_embedding.astype('float32').reshape(1, -1)

    # Search with reduced threshold and increased candidates
    distances, indices = index.search(query_embedding, top_k)  # Get more candidates initially

    # def get_section_score(section, query_terms):
    #     """Calculate relevance score combining semantic and text matching."""
    #     # Text for matching
    #     section_text = f"{section['title']} {' '.join(section['content'])}"

    #     # Simple text matching score
    #     query_terms = set(query.lower())
    #     section_terms = set(section_text.lower())
    #     text_match = len(query_terms & section_terms) / len(query_terms) if query_terms else 0

    #     return text_match
    seen_section = set()

    def build_json(node_id, dist):
        """Recursively constructs JSON response, filtering subsections by distance."""
        if node_id not in section_data or node_id in seen_section:
            return None
        
        seen_section.add(node_id)
        node = section_data[node_id]
        filtered_subsections = [
            build_json(sub_id, np.linalg.norm(query_embedding-get_embedding(section_data[sub_id]["title"]))**2) for sub_id in node["subsections"]
            if sub_id in section_data and np.linalg.norm(query_embedding-get_embedding(section_data[sub_id]["title"]))**2 < distance_threshold
        ]
        filtered_subsections = [x for x in filtered_subsections if x]
        filtered_subsections.sort(key=lambda x:x['distance'])
        
        return {
            "id": node["id"],
            "section_text": node["title"],
            "content": node["content"],
            "context": node['summary'],
            "distance": dist,
            "subsections": [sub for sub in filtered_subsections if sub] if filtered_subsections else None
        }

    results = [build_json(section_data[faiss_id_to_node_id[idx]]['parent'], dist) for dist, idx in zip(distances[0], indices[0]) if dist < distance_threshold]
    results = [x for x in results if x]

    return results if results else None


def print_tree(section_data, node=0, level=0):
    if level > 5:
        return
    """Prints the tree structure with indents, showing the first 10 words of content."""
    indent = "    " * level
    content_preview = " ".join(section_data[node]['content'][0].split()[:10]) + "..." if section_data[node][
        'content'] else "(No content)"
    print(f"{indent}- {section_data[node]['title']}: {content_preview}")
    for child in section_data[node]['subsections']:
        print_tree(section_data, child, level + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contract Review with RAG using BERT-Chinese & FAISS.")
    parser.add_argument("-f", "--file", type=str, help="Path to contract DOCX file for processing.")
    parser.add_argument("-p", "--prompt", type=str, help="Query prompt for contract review.")

    args = parser.parse_args()

    if args.file and not args.prompt:
        print(f"📄 Processing contract: {args.file}")
        sections = extract_hierarchical_structure(args.file)
        store_hierarchical_embeddings(sections)

    elif args.prompt and args.file:
        print("⚠️ Cannot process and query at the same time. Run them separately.")

    elif args.prompt:
        print(f"🔎 Querying: {args.prompt}")
        index = faiss.read_index(EMBEDDINGS_DB)
        with open(TEXT_DB, "rb") as f:
            section_data = pickle.load(f)

        # print_tree(section_data, 0)
        results = retrieve_with_subsections_json(args.prompt)
        # print(results)
        if results:
            print("\n📌 **Relevant Sections:**")
            for res in results:
                print(res,'\n')
        else:
            print("⚠️ No relevant sections found.")

    else:
        print("⚠️ Provide either `-f` to process a file or `-p` to query.")