import sqlite3


def find_user(conn: sqlite3.Connection, username: str):
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE name = ?", (username,))
    return cur.fetchone()
