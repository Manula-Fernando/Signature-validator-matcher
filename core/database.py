import sqlite3
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import DATABASE_PATH

logger = logging.getLogger("SpecimenDatabase")

class SpecimenDatabase:
    """
    Local SQLite Database managing customer signature enrollment profiles,
    256-D neural embeddings, and verification audit trails.
    """

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Customers Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    account_number TEXT,
                    document_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Signature Specimens Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS specimens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    specimen_index INTEGER NOT NULL,
                    image_base64 TEXT NOT NULL,
                    embedding_bytes BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)

            # Verification Audit Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    document_type TEXT,
                    verdict TEXT NOT NULL,
                    is_match INTEGER NOT NULL,
                    similarity_percentage REAL NOT NULL,
                    distance REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def enroll_customer(
        self,
        customer_id: str,
        full_name: str,
        account_number: str = "",
        document_type: str = "NIC/Passport"
    ) -> bool:
        """Enrolls or updates a customer profile."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO customers (customer_id, full_name, account_number, document_type)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    account_number = excluded.account_number,
                    document_type = excluded.document_type
            """, (customer_id, full_name, account_number, document_type))
            conn.commit()
            return True

    def add_specimen(
        self,
        customer_id: str,
        image_base64: str,
        embedding: np.ndarray
    ) -> int:
        """
        Saves a reference specimen image and its 256-D float32 embedding vector for a customer.
        """
        emb_bytes = embedding.astype(np.float32).tobytes()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Determine next specimen index for this customer
            cursor.execute(
                "SELECT COALESCE(MAX(specimen_index), 0) + 1 FROM specimens WHERE customer_id = ?",
                (customer_id,)
            )
            next_idx = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO specimens (customer_id, specimen_index, image_base64, embedding_bytes)
                VALUES (?, ?, ?, ?)
            """, (customer_id, next_idx, image_base64, emb_bytes))
            conn.commit()
            return next_idx

    def get_customer_specimens(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all enrolled specimens and deserializes embeddings for a customer.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, specimen_index, image_base64, embedding_bytes, created_at
                FROM specimens
                WHERE customer_id = ?
                ORDER BY specimen_index ASC
            """, (customer_id,))
            rows = cursor.fetchall()
            
            specimens = []
            for r in rows:
                emb = np.frombuffer(r["embedding_bytes"], dtype=np.float32)
                specimens.append({
                    "id": r["id"],
                    "specimen_index": r["specimen_index"],
                    "image_base64": r["image_base64"],
                    "embedding": emb,
                    "created_at": r["created_at"]
                })
            return specimens

    def list_customers(self) -> List[Dict[str, Any]]:
        """Lists all registered customers with their enrolled specimen counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.customer_id, c.full_name, c.account_number, c.document_type, c.created_at,
                       COUNT(s.id) as specimen_count
                FROM customers c
                LEFT JOIN specimens s ON c.customer_id = s.customer_id
                GROUP BY c.customer_id
                ORDER BY c.created_at DESC
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def log_verification(
        self,
        customer_id: str,
        document_type: str,
        verdict: str,
        is_match: bool,
        similarity_percentage: float,
        distance: float
    ):
        """Records an audit trail entry for every verification operation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (customer_id, document_type, verdict, is_match, similarity_percentage, distance)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (customer_id, document_type, verdict, 1 if is_match else 0, similarity_percentage, distance))
            conn.commit()

    def get_recent_audits(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent verification audit records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, customer_id, document_type, verdict, is_match, similarity_percentage, distance, timestamp
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]
