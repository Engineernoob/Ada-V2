"""
🗄️ SQLite Database Manager for Ada
Handles conversation persistence and storage
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

class ConversationDatabase:
    """SQLite database manager for Ada's conversations"""
    
    def __init__(self, db_path: str = "ada/storage/conversations.db"):
        self.db_path = db_path
        self._ensure_storage_directory()
        self._initialize_database()
    
    def _ensure_storage_directory(self):
        """Ensure the storage directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                ada_response TEXT NOT NULL,
                reward REAL DEFAULT 0.0,
                session_id TEXT,
                turn_number INTEGER DEFAULT 1,
                context TEXT,
                metadata TEXT
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON conversations(session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reward 
            ON conversations(reward)
        """)
        
        # Create embeddings table for future use
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE,
                embedding_vector TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"🗄️ Database initialized: {self.db_path}")
    
    def insert_conversation(self, user_input: str, ada_response: str, 
                          reward: float = 0.0, session_id: str = None, 
                          turn_number: int = 1, context: str = None,
                          metadata: Dict = None) -> int:
        """Insert a new conversation into the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO conversations 
            (timestamp, user_input, ada_response, reward, session_id, 
             turn_number, context, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, user_input, ada_response, reward, session_id, 
              turn_number, context, metadata_json))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Retrieve a specific conversation by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations WHERE id = ?
        """, (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "timestamp": row[1],
                "user_input": row[2],
                "ada_response": row[3],
                "reward": row[4],
                "session_id": row[5],
                "turn_number": row[6],
                "context": row[7],
                "metadata": json.loads(row[8]) if row[8] else None
            }
        return None
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get the most recent conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "timestamp": row[1],
                "user_input": row[2],
                "ada_response": row[3],
                "reward": row[4],
                "session_id": row[5],
                "turn_number": row[6],
                "context": row[7],
                "metadata": json.loads(row[8]) if row[8] else None
            })
        
        return conversations
    
    def get_conversations_by_session(self, session_id: str) -> List[Dict]:
        """Get all conversations from a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "timestamp": row[1],
                "user_input": row[2],
                "ada_response": row[3],
                "reward": row[4],
                "session_id": row[5],
                "turn_number": row[6],
                "context": row[7],
                "metadata": json.loads(row[8]) if row[8] else None
            })
        
        return conversations
    
    def get_average_reward(self, session_id: str = None) -> float:
        """Get average reward for conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT AVG(reward) FROM conversations 
                WHERE session_id = ?
            """, (session_id,))
        else:
            cursor.execute("SELECT AVG(reward) FROM conversations")
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] is not None else 0.0
    
    def get_conversation_stats(self) -> Dict:
        """Get overall conversation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_count = cursor.fetchone()[0]
        
        # Average reward
        cursor.execute("SELECT AVG(reward) FROM conversations")
        avg_reward = cursor.fetchone()[0] or 0.0
        
        # Session count
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
        session_count = cursor.fetchone()[0]
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE timestamp > datetime('now', '-1 day')
        """)
        recent_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_conversations": total_count,
            "total_sessions": session_count,
            "average_reward": avg_reward,
            "recent_conversations_24h": recent_count,
            "database_size_mb": self._get_database_size()
        }
    
    def _get_database_size(self) -> float:
        """Get database file size in MB"""
        if os.path.exists(self.db_path):
            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)  # Convert to MB
        return 0.0
    
    def delete_old_conversations(self, days_to_keep: int = 30) -> int:
        """Delete conversations older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversations 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days_to_keep))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Deleted {deleted_count} old conversations")
        return deleted_count
    
    def close(self):
        """Close database connection (placeholder for connection pooling)"""
        pass

def test_database():
    """Test the database functionality"""
    print("🧪 Testing Conversation Database...")
    
    # Create database instance
    db = ConversationDatabase("ada/storage/test_conversations.db")
    
    # Insert test conversations
    test_conversations = [
        ("Hello Ada", "Hello! Nice to meet you!", 0.8),
        ("How are you?", "I'm doing well, thank you!", 0.7),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!", 0.6),
        ("Goodbye", "Goodbye! Hope to talk again soon!", 0.9)
    ]
    
    session_id = "test_session_001"
    for i, (user_input, ada_response, reward) in enumerate(test_conversations):
        db.insert_conversation(
            user_input=user_input,
            ada_response=ada_response,
            reward=reward,
            session_id=session_id,
            turn_number=i+1,
            context="test_conversation"
        )
    
    # Test retrieval
    recent = db.get_recent_conversations(3)
    print(f"Retrieved {len(recent)} recent conversations")
    
    # Test session retrieval
    session_conversations = db.get_conversations_by_session(session_id)
    print(f"Session has {len(session_conversations)} conversations")
    
    # Test statistics
    stats = db.get_conversation_stats()
    print(f"Database stats: {stats}")
    
    # Test average reward
    avg_reward = db.get_average_reward(session_id)
    print(f"Average reward for session: {avg_reward:.3f}")
    
    print("✅ Database test completed!")

if __name__ == "__main__":
    test_database()