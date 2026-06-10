import sqlite3
import bcrypt
import os

DB_PATH = "data/rag.db"

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_admin():
    print("=== Create Admin Account ===")
    username = input("Enter admin username: ").strip()
    password = input("Enter admin password: ").strip()
    
    if not username or not password:
        print("Username and password cannot be empty.")
        return
        
    hashed_pw = hash_password(password)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Check if user exists
        cur.execute("SELECT role FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            # Update to admin
            cur.execute("UPDATE users SET role = 'admin', password_hash = ? WHERE username = ?", (hashed_pw, username))
            print(f"User '{username}' updated to admin role successfully.")
        else:
            # Create new admin
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
                (username, hashed_pw)
            )
            print(f"Admin account '{username}' created successfully.")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        print("Make sure you run this script from the root of the project.")
    else:
        create_admin()
