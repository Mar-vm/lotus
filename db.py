import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": "aws-0-us-west-2.pooler.supabase.com",
    "port": 5432,
    "database": "postgres",
    "user": "postgres.iiztllsahklrsnclmoup",
    "password": "LaCasaDePapel123"
}

def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            cursor_factory=RealDictCursor
        )
        print("Conectado a Supabase PostgreSQL")
        return connection
    except Exception as e:
        print("Error connecting to database:", e)
        return None


# prueba directa
if __name__ == "__main__":
    conn = get_db_connection()

    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()

        print("Hora del servidor:", result)

        cursor.close()
        conn.close()