from psycopg.types.json import Jsonb
from app.core.db import pool


def write_audit(thread_id: str, question: str, answer: str, source_used: str, trace: list[str]) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO query_audit (thread_id, question, answer, source_used, trace) VALUES (%s, %s, %s, %s, %s)",
            (thread_id, question, answer, source_used, Jsonb(trace)),
        )
