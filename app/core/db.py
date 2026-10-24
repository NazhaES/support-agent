from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from app.core.config import get_settings

settings = get_settings()

pool = ConnectionPool(
    conninfo=settings.database_url,
    max_size=10,
    open=True,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)

checkpointer = PostgresSaver(pool)


def init_db() -> None:
    checkpointer.setup()
    with pool.connection() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                thread_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace JSONB NOT NULL
            )"""
        )


init_db()
