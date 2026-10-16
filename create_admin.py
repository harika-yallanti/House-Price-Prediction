
import os
import sqlite3
from getpass import getpass
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "auth.db")


def create_admin():
    if not os.path.exists(DATABASE_PATH):
        print("Database not found. Check the database path.")
        return

    name = input("Enter admin name: ").strip()
    email = input("Enter admin email: ").strip().lower()
    password = getpass("Enter admin password: ")
    confirm_password = getpass("Confirm admin password: ")

    if not name or not email or not password:
        print("All fields are required.")
        return

    if password != confirm_password:
        print("Passwords do not match.")
        return

    if len(password) < 12:
        print("Use a password with at least 12 characters.")
        return

    connection = sqlite3.connect(DATABASE_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        required_columns = {
            "name", "email", "password_hash", "role"
        }

        if not required_columns.issubset(set(columns)):
            print("The users table structure is different than expected.")
            print("No changes were made.")
            return

        cursor.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        )

        if cursor.fetchone():
            print("This email already exists. Choose another email.")
            return

        password_hash = generate_password_hash(password)

        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, password_hash, "admin")
        )

        connection.commit()
        print("Administrator account created successfully.")
        print("You can now log in using your admin email and password.")

    except sqlite3.Error as error:
        connection.rollback()
        print("Database error:", error)

    finally:
        connection.close()


if __name__ == "__main__":
    create_admin()
