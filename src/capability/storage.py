"""Storage backend for capabilities"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CapabilityStorage:
    """SQLite storage for capabilities"""
    
    def __init__(self, db_path: str = "./data/capabilities.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    embedding BLOB,
                    examples TEXT,
                    performance_stats TEXT,
                    created_at REAL,
                    last_used REAL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON capabilities(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage ON capabilities(usage_count)")
            
    def save_capability(self, capability: Dict[str, Any]):
        """Save capability to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Convert numpy array to bytes
            embedding_bytes = capability["embedding"].tobytes() if "embedding" in capability else None
            
            conn.execute("""
                INSERT OR REPLACE INTO capabilities 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                capability["id"],
                capability["name"],
                capability.get("description", ""),
                embedding_bytes,
                json.dumps(capability.get("examples", [])),
                json.dumps(capability.get("performance_stats", {})),
                capability.get("created_at", 0),
                capability.get("last_used", 0),
                capability.get("usage_count", 0),
                capability.get("success_rate", 0.0)
            ))
            
    def load_capability(self, cap_id: str) -> Optional[Dict[str, Any]]:
        """Load capability by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM capabilities WHERE id = ?", (cap_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return self._row_to_capability(row)
