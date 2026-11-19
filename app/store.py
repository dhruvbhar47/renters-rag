"""Postgres + pgvector storage, plus the two search primitives (vector and keyword)."""
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import settings

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    url        TEXT NOT NULL,
    title      TEXT NOT NULL,
    chunk_idx  INT  NOT NULL,
    content    TEXT NOT NULL,
    updated_at TEXT,
    embedding  vector({settings.embed_dim}),
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED,
    UNIQUE (url, chunk_idx)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_STOPWORDS = {
    "the", "and", "for", "are", "was", "what", "when", "how", "can", "does", "did",
    "with", "that", "this", "have", "has", "from", "your", "you", "about", "into",
    "will", "should", "would", "could", "who", "why", "where", "which", "there",
}


def keyword_query(text: str) -> str:
    """Build an OR-style tsquery string from free text (alphanumerics only, so it is injection-safe)."""
    terms = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]
    return " | ".join(dict.fromkeys(terms))


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or settings.database_url, autocommit=True, row_factory=dict_row)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


class PgStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def replace_document(
        self, url: str, title: str, updated_at: str | None, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunks WHERE url = %s", (url,))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (url, title, chunk_idx, content, updated_at, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (url, title, i, text, updated_at, np.array(vec, dtype=np.float32))
                        for i, (text, vec) in enumerate(zip(chunks, embeddings))
                    ],
                )

    def vector_search(self, query_vector: list[float], k: int) -> list[dict]:
        qv = np.array(query_vector, dtype=np.float32)
        return self.conn.execute(
            "SELECT id, url, title, content, updated_at, 1 - (embedding <=> %s) AS vector_score "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (qv, qv, k),
        ).fetchall()

    def keyword_search(self, query: str, k: int) -> list[dict]:
        q = keyword_query(query)
        if not q:
            return []
        return self.conn.execute(
            "SELECT id, url, title, content, updated_at, ts_rank_cd(tsv, tsq) AS keyword_score "
            "FROM chunks, to_tsquery('english', %s) tsq WHERE tsv @@ tsq "
            "ORDER BY keyword_score DESC LIMIT %s",
            (q, k),
        ).fetchall()

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
