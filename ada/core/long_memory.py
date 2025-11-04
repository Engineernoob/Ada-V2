"""
🧠 Long-Term Memory System for Ada v3.0
Vector-based semantic memory with embedding similarity search
"""

import uuid
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

# Try to import FAISS, fallback to simple similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("📝 FAISS not available, using fallback similarity search")

# Try to import sentence transformers, fallback to simple embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("📝 sentence-transformers not available, using fallback embeddings")

@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    text: str
    vector: List[float]
    timestamp: str
    session_id: str
    turn_number: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(**data)

class EmbeddingGenerator:
    """Handles text embedding generation"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"🔍 Loaded embedding model: {model_name} (dim: {self.dimension})")
            except Exception as e:
                print(f"⚠️ Error loading embedding model: {e}")
                print("📝 Using fallback embeddings")
        else:
            print("📝 Using fallback embedding generation")
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.model:
            try:
                embedding = self.model.encode([text])[0]
                return embedding.tolist()
            except Exception as e:
                print(f"⚠️ Embedding error: {e}")
                return self._fallback_embedding(text)
        else:
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding based on word frequencies"""
        words = text.lower().split()
        embedding = [0.0] * self.dimension
        
        # Simple hash-based embedding
        for i, word in enumerate(words):
            hash_val = hash(word) % self.dimension
            embedding[hash_val] += 1.0
        
        # Normalize
        total = sum(embedding)
        if total > 0:
            embedding = [x / total for x in embedding]
        
        return embedding
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️ Similarity calculation error: {e}")
            # Simple fallback
            return 1.0 if vec1 == vec2 else 0.0

class FAISSIndex:
    """FAISS-based vector index (if available)"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.entries = []
        
        if FAISS_AVAILABLE:
            try:
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                print(f"🗃️ Initialized FAISS index (dim: {dimension})")
            except Exception as e:
                print(f"⚠️ FAISS initialization error: {e}")
    
    def add(self, vector: List[float], entry: MemoryEntry):
        """Add vector to index"""
        if self.index:
            try:
                import numpy as np
                vector_np = np.array(vector).reshape(1, -1)
                self.index.add(vector_np)
                self.entries.append(entry)
            except Exception as e:
                print(f"⚠️ FAISS add error: {e}")
    
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[float, MemoryEntry]]:
        """Search for similar vectors"""
        if self.index and self.entries:
            try:
                import numpy as np
                query_np = np.array(query_vector).reshape(1, -1)
                distances, indices = self.index.search(query_np, min(top_k, len(self.entries)))
                
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(self.entries):
                        entry = self.entries[idx]
                        results.append((float(distance), entry))
                
                return results
            except Exception as e:
                print(f"⚠️ FAISS search error: {e}")
        
        return []

class SimpleIndex:
    """Simple fallback index when FAISS is not available"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.entries = []
        self.embedding_generator = EmbeddingGenerator()
    
    def add(self, vector: List[float], entry: MemoryEntry):
        """Add entry to simple index"""
        self.entries.append((vector, entry))
    
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[float, MemoryEntry]]:
        """Search using simple similarity"""
        similarities = []
        
        for vector, entry in self.entries:
            similarity = self.embedding_generator.similarity(query_vector, vector)
            similarities.append((similarity, entry))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return similarities[:top_k]

class LongMemory:
    """Long-term memory system with vector storage and semantic search"""
    
    def __init__(self, memory_dir: str = "storage/memory", embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / "long_memory.jsonl"
        self.summary_dir = self.memory_dir / "summaries"
        
        # Ensure directories exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(embed_model)
        
        # Initialize index
        if FAISS_AVAILABLE:
            self.index = FAISSIndex(self.embedding_generator.dimension)
        else:
            self.index = SimpleIndex(self.embedding_generator.dimension)
        
        # Load existing memories
        self.memories = []
        self._load_memories()
        
        print(f"🧠 Long-term memory initialized")
        print(f"📂 Memory file: {self.memory_file}")
        print(f"📊 Loaded {len(self.memories)} memories")
    
    def add(self, text: str, session_id: str = "", turn_number: int = 0) -> str:
        """Add a new memory entry"""
        
        # Generate embedding
        vector = self.embedding_generator.encode(text)
        
        # Create memory entry
        memory_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=memory_id,
            text=text,
            vector=vector,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            turn_number=turn_number
        )
        
        # Add to index
        self.index.add(vector, entry)
        
        # Add to memory list
        self.memories.append(entry)
        
        # Save to file
        self._save_memory(entry)
        
        return memory_id
    
    def query(self, context: str, top_k: int = 3) -> List[str]:
        """Query for similar memories"""
        if not self.memories:
            return []
        
        # Generate embedding for query
        query_vector = self.embedding_generator.encode(context)
        
        # Search index
        results = self.index.search(query_vector, top_k)
        
        # Extract relevant memories
        relevant_memories = []
        for similarity, entry in results:
            if similarity > 0.1:  # Threshold for relevance
                relevant_memories.append(entry.text)
        
        return relevant_memories
    
    def query_by_session(self, session_id: str) -> List[MemoryEntry]:
        """Get all memories from a specific session"""
        return [entry for entry in self.memories if entry.session_id == session_id]
    
    def get_recent_memories(self, count: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories"""
        return sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:count]
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[MemoryEntry]:
        """Search memories by keywords"""
        matched_memories = []
        
        for entry in self.memories:
            text_lower = entry.text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                matched_memories.append(entry)
        
        return matched_memories[-top_k:]  # Return most recent matches
    
    def _load_memories(self):
        """Load memories from disk"""
        if not self.memory_file.exists():
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = MemoryEntry.from_dict(data)
                        self.memories.append(entry)
                        
                        # Rebuild index
                        self.index.add(entry.vector, entry)
                        
        except Exception as e:
            print(f"⚠️ Error loading memories: {e}")
    
    def _save_memory(self, entry: MemoryEntry):
        """Save a single memory entry to disk"""
        try:
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error saving memory: {e}")
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics"""
        return {
            "total_memories": len(self.memories),
            "unique_sessions": len(set(entry.session_id for entry in self.memories)),
            "average_text_length": sum(len(entry.text) for entry in self.memories) / max(len(self.memories), 1),
            "memory_file_size_mb": self.memory_file.stat().st_size / (1024 * 1024) if self.memory_file.exists() else 0,
            "embedding_model": self.embedding_generator.model_name,
            "faiss_enabled": FAISS_AVAILABLE
        }

# Global long-term memory instance
long_memory = None

def initialize_long_memory(memory_dir: str = "storage/memory") -> LongMemory:
    """Initialize the global long-term memory system"""
    global long_memory
    long_memory = LongMemory(memory_dir)
    return long_memory

def add_to_long_memory(text: str, session_id: str = "", turn_number: int = 0) -> str:
    """Add text to long-term memory"""
    if long_memory:
        return long_memory.add(text, session_id, turn_number)
    return ""

def query_long_memory(context: str, top_k: int = 3) -> List[str]:
    """Query long-term memory for similar contexts"""
    if long_memory:
        return long_memory.query(context, top_k)
    return []

if __name__ == "__main__":
    # Test the long-term memory system
    print("🧪 Testing Long-Term Memory System...")
    
    # Initialize
    memory = LongMemory("test_memory")
    
    # Add test memories
    memories = [
        ("I love programming in Python", "session_001", 1),
        ("The weather is nice today", "session_001", 2),
        ("I'm working on a machine learning project", "session_002", 1),
        ("Ada has a neural network brain", "session_002", 2),
        ("Phase 3 adds long-term memory", "session_003", 1)
    ]
    
    for text, session, turn in memories:
        memory_id = memory.add(text, session, turn)
        print(f"Added memory: {memory_id[:8]}... - {text}")
    
    # Test queries
    print("\n🔍 Testing memory queries...")
    queries = [
        "What do I like to code?",
        "What's the weather like?",
        "Tell me about Ada",
        "What phases have we implemented?"
    ]
    
    for query in queries:
        results = memory.query(query, top_k=2)
        print(f"\nQuery: '{query}'")
        print(f"Results: {results}")
    
    # Test statistics
    stats = memory.get_statistics()
    print(f"\n📊 Memory Statistics: {stats}")
    
    print("\n✅ Long-term memory system test completed!")