# data_store.py
import sqlite3
import pandas as pd
import io

DB_NAME = "health_data.db"

# expected columns and their SQL types
EXPECTED_COLUMNS = {
    "user_id": "TEXT",
    "ts": "TEXT",
    "age": "INTEGER",
    "sex": "TEXT",
    "height_cm": "REAL",
    "weight_kg": "REAL",
    "bmi": "REAL",
    "sbp": "REAL",
    "dbp": "REAL",
    "hr": "REAL",
    "spo2": "REAL",
    "glucose": "REAL",
    "symptoms": "TEXT",
    "free_text": "TEXT",
    "risk_Type_2_Diabetes_Risk": "REAL",
    "risk_Hypertension_Risk": "REAL",
    "risk_Depression_Mood_Concern": "REAL",
    "risk_Migraine_Risk": "REAL",
    "risk_Sleep_Apnea_Risk": "REAL",
    "risk_Anemia_Risk": "REAL",
}


def init_db():
    """Initialize DB and ensure expected columns exist (create table or alter as needed)."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # If table doesn't exist create it with full schema
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            user_id TEXT,
            ts TEXT,
            age INTEGER,
            sex TEXT,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            sbp REAL,
            dbp REAL,
            hr REAL,
            spo2 REAL,
            glucose REAL,
            symptoms TEXT,
            free_text TEXT,
            risk_Type_2_Diabetes_Risk REAL,
            risk_Hypertension_Risk REAL,
            risk_Depression_Mood_Concern REAL,
            risk_Migraine_Risk REAL,
            risk_Sleep_Apnea_Risk REAL,
            risk_Anemia_Risk REAL
        )
        """
    )
    conn.commit()

    # Ensure any missing columns are added (handles older DBs)
    cur.execute("PRAGMA table_info(entries);")
    existing = [row[1] for row in cur.fetchall()]  # column names

    for col, coltype in EXPECTED_COLUMNS.items():
        if col not in existing:
            # Add the missing column
            alter_sql = f'ALTER TABLE entries ADD COLUMN "{col}" {coltype}'
            cur.execute(alter_sql)
            conn.commit()

    conn.close()


def upsert_entry(entry: dict):
    """Insert entry into SQLite DB (append)."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.DataFrame([entry])
    # Ensure DataFrame columns exist in DB by filling missing expected columns
    for col in EXPECTED_COLUMNS.keys():
        if col not in df.columns:
            df[col] = pd.NA
    # Keep only expected columns and in order (prevents unexpected-columns errors)
    df = df[[c for c in EXPECTED_COLUMNS.keys() if c in df.columns]]
    df.to_sql("entries", conn, if_exists="append", index=False)
    conn.close()


def get_user_history(user_id: str) -> pd.DataFrame:
    """Retrieve all entries for a user_id."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM entries WHERE user_id = ? ORDER BY ts ASC",
            conn,
            params=(user_id,),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return DataFrame as downloadable CSV (bytes)."""
    with io.StringIO() as buf:
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
