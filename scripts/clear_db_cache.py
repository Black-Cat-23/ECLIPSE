import sqlite3

def clear_cache():
    conn = sqlite3.connect('eclipse.db')
    cursor = conn.cursor()
    # Delete where predicted_class is OTHER or period is NULL
    cursor.execute("DELETE FROM candidates WHERE predicted_class = 'OTHER' OR period IS NULL")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Cleared {deleted} bad cache entries from the database.")

if __name__ == "__main__":
    clear_cache()
