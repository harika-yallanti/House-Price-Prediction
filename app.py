from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from email.message import EmailMessage
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

import pandas as pd
import joblib
import sqlite3
import os
import math
import hashlib
import hmac
import secrets
import smtplib

import firebase_admin
from firebase_admin import credentials, auth

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[]
)
FIREBASE_SERVICE_ACCOUNT_PATH = os.environ.get(
    "FIREBASE_SERVICE_ACCOUNT_PATH"
)

if not FIREBASE_SERVICE_ACCOUNT_PATH:
    raise RuntimeError(
        "Set the FIREBASE_SERVICE_ACCOUNT_PATH environment variable."
    )

if not firebase_admin._apps:
    credential = credentials.Certificate(
        FIREBASE_SERVICE_ACCOUNT_PATH
    )
    firebase_admin.initialize_app(credential)

app.secret_key = os.environ.get("SECRET_KEY")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get(
        "COOKIE_SECURE", "false"
    ).lower() == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_REFRESH_EACH_REQUEST=False
)

if not app.secret_key:
    raise RuntimeError(
        "Set the SECRET_KEY environment variable before starting Flask."
    )
csrf = CSRFProtect(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_final.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_house_data.csv")
DATASET_BACKUP_PATH = os.path.join(
    BASE_DIR, "dataset", "cleaned_house_data_backup.csv"
)
DATABASE_PATH = os.path.join(BASE_DIR, "auth.db")
# print("Using database:", os.path.abspath(DATABASE_PATH))
DATASET_NAME_PATH = os.path.join(
    BASE_DIR, "dataset", "current_dataset_name.txt"
)
ALLOWED_DATASET_COLUMNS = {
    "location",
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "price"
}


model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)
df["location"] = df["location"].astype(str).str.strip()

locations = sorted(
    location for location in df["location"].unique()
    if location and location.lower() != "nan"
)

price_insights = {
    "total_properties": len(df),
    "average_price": round(float(df["price"].mean()), 2),
    "minimum_price": round(float(df["price"].min()), 2),
    "maximum_price": round(float(df["price"].max()), 2)
}


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:

        # Create the users table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


        # Check whether firebase_uid already exists
        user_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        # Add firebase_uid if it is missing
        if "firebase_uid" not in user_columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN firebase_uid TEXT"
            )

        # Ensure each Firebase UID is unique
        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_users_firebase_uid
            ON users(firebase_uid)
            WHERE firebase_uid IS NOT NULL
        """)

        # Create the prediction history table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                location TEXT NOT NULL,
                total_sqft REAL NOT NULL,
                bhk INTEGER NOT NULL,
                bath INTEGER NOT NULL,
                balcony INTEGER NOT NULL,
                predicted_price REAL NOT NULL,
                price_per_sqft REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create the email OTP verification table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_login_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

initialize_database()


def send_login_code(email, code):
    message = EmailMessage()
    message["Subject"] = "Your House Price Prediction login code"
    message["From"] = os.environ["MAIL_FROM"]
    message["To"] = email
    message.set_content(
        f"Your login verification code is {code}.\n\n"
        "This code expires in 5 minutes. "
        "Do not share it with anyone."
    )

    with smtplib.SMTP(
        os.environ["SMTP_HOST"],
        int(os.environ.get("SMTP_PORT", "587"))
    ) as server:
        server.starttls()
        server.login(
            os.environ["SMTP_USERNAME"],
            os.environ["SMTP_PASSWORD"]
        )
        server.send_message(message)


def create_login_code(connection, user_id):
    code = f"{secrets.randbelow(1000000):06d}"
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(minutes=5)).isoformat()

    code_hash = hashlib.sha256(code.encode()).hexdigest()

    connection.execute(
        "DELETE FROM email_login_codes WHERE user_id = ?",
        (user_id,)
    )

    connection.execute(
        """
        INSERT INTO email_login_codes
        (user_id, code_hash, expires_at, attempts, created_at)
        VALUES (?, ?, ?, 0, ?)
        """,
        (user_id, code_hash, expires_at, now.isoformat())
    )

    return code


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the prediction system.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def admin_required(view_function):
    @wraps(view_function)
    @login_required
    def wrapped_view(*args, **kwargs):
        if session.get("user_role") != "admin":
            flash("Administrator access required.", "error")
            return redirect(url_for("home"))

        return view_function(*args, **kwargs)

    return wrapped_view


@app.route("/")
@login_required
def home():
    return render_template(
        "index.html",
        locations=locations,
        prediction=None,
        error=None,
        form_data={
            "location": "",
            "total_sqft": "",
            "bhk": "2",
            "bath": "2",
            "balcony": "1"
        },
        price_insights=price_insights,
        price_per_sqft=None
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    prediction = None
    error = None
    price_per_sqft = None

    form_data = {
        "location": request.form.get("location", "").strip(),
        "total_sqft": request.form.get("total_sqft", "").strip(),
        "bhk": request.form.get("bhk", "").strip(),
        "bath": request.form.get("bath", "").strip(),
        "balcony": request.form.get("balcony", "").strip()
    }

    try:
        location = form_data["location"]
        total_sqft = float(form_data["total_sqft"])
        bhk = int(form_data["bhk"])
        bath = int(form_data["bath"])
        balcony = int(form_data["balcony"])

        if not math.isfinite(total_sqft):
            raise ValueError("Please enter a valid area.")

        if location not in locations:
            raise ValueError("Please select a valid location.")

        if not 300 <= total_sqft <= 10000:
            raise ValueError(
                "Area must be between 300 and 10,000 square feet."
            )

        if not 1 <= bhk <= 10:
            raise ValueError("Bedrooms must be between 1 and 10.")

        if not 1 <= bath <= 10:
            raise ValueError("Bathrooms must be between 1 and 10.")

        if not 0 <= balcony <= 5:
            raise ValueError("Balconies must be between 0 and 5.")

        if total_sqft / bhk < 300:
            raise ValueError(
                "Please enter a larger area for the selected bedrooms."
            )

        house = pd.DataFrame([{
            "location": location,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
            "bhk": bhk
        }])

        predicted_price = float(model.predict(house)[0])

        if not math.isfinite(predicted_price):
            raise ValueError(
                "The model could not produce a valid estimate."
            )

        prediction = max(0, predicted_price)
        price_per_sqft = round(prediction * 100000 / total_sqft, 2)

        with get_db_connection() as connection:
            connection.execute("""
                INSERT INTO prediction_history (
                    user_id, location, total_sqft, bhk, bath,
                    balcony, predicted_price, price_per_sqft
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                location,
                total_sqft,
                bhk,
                bath,
                balcony,
                prediction,
                price_per_sqft
            ))

    except (ValueError, TypeError, OverflowError):
        error = "Please check the property details and enter valid values."

    return render_template(
        "index.html",
        locations=locations,
        prediction=prediction,
        error=error,
        form_data=form_data,
        price_insights=price_insights,
        price_per_sqft=price_per_sqft
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            flash("Please complete all fields.", "error")

        elif len(name) > 100 or len(email) > 254:
            flash("Name or email is too long.", "error")

        elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash("Please enter a valid email address.", "error")

        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")

        elif password != confirm_password:
            flash("Passwords do not match.", "error")

        else:
            try:
                with get_db_connection() as connection:
                    connection.execute(
                        """
                        INSERT INTO users (name, email, password_hash)
                        VALUES (?, ?, ?)
                        """,
                        (name, email, generate_password_hash(password))
                    )

                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))

            except sqlite3.IntegrityError:
                flash("An account with this email already exists.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if "user_id" in session:
        if session.get("user_role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                """
                SELECT id, name, email, password_hash, role
                FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            if not user or not check_password_hash(
                user["password_hash"], password
            ):
                flash("Invalid email or password.", "error")
                return render_template("login.html")

            code = create_login_code(connection, user["id"])

        try:
            send_login_code(user["email"], code)
        except Exception:
            app.logger.exception("Failed to send login verification code")
            with get_db_connection() as connection:
                connection.execute(
                    "DELETE FROM email_login_codes WHERE user_id = ?",
                    (user["id"],)
                )
            flash(
                "Unable to send the verification code. Please try again.",
                "error"
            )
            return render_template("login.html")

        session["pending_login_user_id"] = user["id"]
        flash("A verification code has been sent to your email.", "success")
        return redirect(url_for("verify_email_code"))

    return render_template("login.html")


@app.route("/verify-email-code", methods=["GET"])
def verify_email_code():
    if "user_id" in session:
        return redirect(url_for("home"))

    if "pending_login_user_id" not in session:
        flash("Please sign in first.", "error")
        return redirect(url_for("login"))

    return render_template("verify_email_code.html")


@app.route("/auth/verify-email-code", methods=["POST"])
@limiter.limit("10 per minute")
def verify_email_code_submit():
    if "user_id" in session:
        return jsonify({
            "success": True,
            "redirect": (
                url_for("admin_dashboard")
                if session.get("user_role") == "admin"
                else url_for("home")
            )
        })

    user_id = session.get("pending_login_user_id")
    data = request.get_json(silent=True) or request.form
    code = str(data.get("code", "")).strip()

    if not user_id:
        return jsonify({
            "error": "Your login session has expired. Please sign in again."
        }), 401

    if len(code) != 6 or not code.isdigit():
        return jsonify({
            "error": "Enter the six-digit verification code."
        }), 400

    now = datetime.now(timezone.utc)

    with get_db_connection() as connection:
        challenge = connection.execute(
            """
            SELECT id, code_hash, expires_at, attempts
            FROM email_login_codes
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

        user = connection.execute(
            """
            SELECT id, name, email, role
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        if not challenge or not user:
            session.pop("pending_login_user_id", None)
            return jsonify({
                "error": "No active verification code. Please sign in again."
            }), 401

        if datetime.fromisoformat(
            challenge["expires_at"]
        ) <= now:
            connection.execute(
                "DELETE FROM email_login_codes WHERE id = ?",
                (challenge["id"],)
            )
            session.pop("pending_login_user_id", None)
            return jsonify({
                "error": "The code has expired. Please sign in again."
            }), 401

        if challenge["attempts"] >= 5:
            connection.execute(
                "DELETE FROM email_login_codes WHERE id = ?",
                (challenge["id"],)
            )
            session.pop("pending_login_user_id", None)
            return jsonify({
                "error": "Too many attempts. Please sign in again."
            }), 429

        submitted_hash = hashlib.sha256(code.encode()).hexdigest()

        if not hmac.compare_digest(
            submitted_hash, challenge["code_hash"]
        ):
            connection.execute(
                """
                UPDATE email_login_codes
                SET attempts = attempts + 1
                WHERE id = ?
                """,
                (challenge["id"],)
            )
            return jsonify({
                "error": "Incorrect verification code."
            }), 400

        connection.execute(
            "DELETE FROM email_login_codes WHERE id = ?",
            (challenge["id"],)
        )

    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_role"] = user["role"]
    session.permanent = True

    destination = (
        url_for("admin_dashboard")
        if user["role"] == "admin"
        else url_for("home")
    )

    return jsonify({
        "success": True,
        "redirect": destination
    })

@app.route("/auth/google", methods=["POST"])
@limiter.limit("10 per minute")
def google_login():
    data = request.get_json(silent=True) or {}
    token = data.get("idToken")

    if not token:
        return jsonify({"error": "Missing Firebase ID token."}), 400

    try:
        decoded_token = auth.verify_id_token(token)

        firebase_uid = decoded_token.get("uid")
        email = decoded_token.get("email", "").strip().lower()
        name = decoded_token.get("name") or email.split("@")[0]

        if not firebase_uid or not email:
            return jsonify({
                "error": "Google account email is unavailable."
            }), 400

        if decoded_token.get("email_verified") is not True:
            return jsonify({
                "error": "Please verify your Google account email first."
            }), 403

        with get_db_connection() as connection:
            user = connection.execute(
                """
                SELECT id, name, email, role, firebase_uid
                FROM users
                WHERE firebase_uid = ?
                """,
                (firebase_uid,)
            ).fetchone()

            if user is None:
                existing_user = connection.execute(
                    """
                    SELECT id, name, email, role
                    FROM users
                    WHERE email = ?
                    """,
                    (email,)
                ).fetchone()

                if existing_user:
                    return jsonify({
                        "error": (
                            "An account already exists with this email. "
                            "Please use your existing password login."
                        )
                    }), 409

                connection.execute(
                    """
                    INSERT INTO users
                        (name, email, password_hash, role, firebase_uid)
                    VALUES (?, ?, ?, 'user', ?)
                    """,
                    (
                        name[:100],
                        email,
                        generate_password_hash(secrets.token_hex(32)),
                        firebase_uid
                    )
                )

                user = connection.execute(
                    """
                    SELECT id, name, email, role, firebase_uid
                    FROM users
                    WHERE firebase_uid = ?
                    """,
                    (firebase_uid,)
                ).fetchone()

            code = create_login_code(connection, user["id"])

        try:
            send_login_code(user["email"], code)
        except Exception:
            app.logger.exception(
                "Failed to send Google login verification code"
            )
            with get_db_connection() as connection:
                connection.execute(
                    "DELETE FROM email_login_codes WHERE user_id = ?",
                    (user["id"],)
                )
            return jsonify({
                "error": (
                    "Unable to send the verification code. "
                    "Please try again."
                )
            }), 503

        session.clear()
        session["pending_login_user_id"] = user["id"]

        return jsonify({
            "success": True,
            "challenge_required": True,
            "redirect": url_for("verify_email_code")
        })

    except (ValueError, auth.InvalidIdTokenError):
        return jsonify({
            "error": "Google sign-in could not be verified. Please try again."
        }), 401

    except Exception:
        app.logger.exception("Google sign-in failed")
        return jsonify({
            "error": "Sign-in is temporarily unavailable. Please try again."
        }), 500

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    comparisons = []
    error = None

    form_data = {
        "total_sqft": "1200",
        "bhk": "2",
        "bath": "2",
        "balcony": "1",
        "locations": []
    }

    if request.method == "POST":
        form_data["total_sqft"] = request.form.get(
            "total_sqft", ""
        ).strip()
        form_data["bhk"] = request.form.get("bhk", "").strip()
        form_data["bath"] = request.form.get("bath", "").strip()
        form_data["balcony"] = request.form.get("balcony", "").strip()
        form_data["locations"] = request.form.getlist("locations")

        try:
            total_sqft = float(form_data["total_sqft"])
            bhk = int(form_data["bhk"])
            bath = int(form_data["bath"])
            balcony = int(form_data["balcony"])
            selected_locations = form_data["locations"]

            if not math.isfinite(total_sqft) or not 300 <= total_sqft <= 10000:
                raise ValueError(
                    "Area must be between 300 and 10,000 square feet."
                )

            if not 1 <= bhk <= 10:
                raise ValueError("Bedrooms must be between 1 and 10.")

            if not 1 <= bath <= 10:
                raise ValueError("Bathrooms must be between 1 and 10.")

            if not 0 <= balcony <= 5:
                raise ValueError("Balconies must be between 0 and 5.")

            if total_sqft / bhk < 300:
                raise ValueError(
                    "Please enter a larger area for the selected bedrooms."
                )

            if len(selected_locations) < 2:
                raise ValueError("Please select at least two locations.")

            if len(selected_locations) != len(set(selected_locations)):
                raise ValueError("Please select each location only once.")

            if any(location not in locations for location in selected_locations):
                raise ValueError("Please select valid locations.")

            for location in selected_locations:
                house = pd.DataFrame([{
                    "location": location,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "balcony": balcony,
                    "bhk": bhk
                }])

                predicted_price = float(model.predict(house)[0])

                if not math.isfinite(predicted_price):
                    raise ValueError(
                        "A valid estimate could not be generated."
                    )

                predicted_price = max(0, predicted_price)

                comparisons.append({
                    "location": location,
                    "predicted_price": predicted_price,
                    "price_per_sqft": predicted_price * 100000 / total_sqft
                })

            comparisons.sort(
                key=lambda item: item["predicted_price"],
                reverse=True
            )

        except (ValueError, TypeError, OverflowError) as e:
            comparisons = []
            error = str(e)

    return render_template(
        "compare.html",
        locations=locations,
        comparisons=comparisons,
        error=error,
        form_data=form_data
    )


@app.route("/history")
@login_required
def history():
    with get_db_connection() as connection:
        records = connection.execute("""
            SELECT id, location, total_sqft, bhk, bath, balcony,
                   predicted_price, price_per_sqft, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
        """, (session["user_id"],)).fetchall()

    predictions = []
    india_timezone = ZoneInfo("Asia/Kolkata")

    for item in records:
        record = dict(item)

        try:
            created_at = datetime.strptime(
                record["created_at"], "%Y-%m-%d %H:%M:%S"
            )

            created_at = created_at.replace(tzinfo=ZoneInfo("UTC"))
            created_at = created_at.astimezone(india_timezone)

            record["display_time"] = created_at.strftime(
                "%d %b %Y, %I:%M:%S %p"
            )

        except (ValueError, TypeError):
            record["display_time"] = record.get("created_at", "")

        predictions.append(record)

    return render_template("history.html", predictions=predictions)


@app.route("/market-trends")
@login_required
def market_trends():
    average_prices = (
        df.groupby("location")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
    )

    property_counts = df["location"].value_counts().head(10)

    return render_template(
        "market_trends.html",
        price_locations=average_prices.index.tolist(),
        average_prices=average_prices.tolist(),
        count_locations=property_counts.index.tolist(),
        property_counts=property_counts.tolist()
    )


@app.route("/admin")
@admin_required
def admin_dashboard():
    with get_db_connection() as connection:
        total_users = connection.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

        total_predictions = connection.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]

        recent_predictions = connection.execute("""
            SELECT prediction_history.id,
                   users.name,
                   prediction_history.location,
                   prediction_history.predicted_price,
                   prediction_history.created_at
            FROM prediction_history
            JOIN users ON prediction_history.user_id = users.id
            ORDER BY prediction_history.created_at DESC,
                     prediction_history.id DESC
            LIMIT 5
        """).fetchall()

        users = connection.execute("""
            SELECT id, name, email, role, created_at
            FROM users
            ORDER BY id DESC
        """).fetchall()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        users=users,
        total_predictions=total_predictions,
        recent_predictions=recent_predictions
    )



@app.route("/admin/dataset", methods=["GET", "POST"])
@admin_required
def manage_dataset():
    global df, locations, price_insights

    message = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a CSV file."

        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are allowed."

        else:
            try:
                
                uploaded_df = pd.read_csv(uploaded_file)

                missing_columns = (
                    ALLOWED_DATASET_COLUMNS - set(uploaded_df.columns)
                )

                if missing_columns:
                    error = (
                        "Missing required columns: "
                        + ", ".join(sorted(missing_columns))
                    )

                elif uploaded_df.empty:
                    error = "The uploaded dataset is empty."

                elif uploaded_df["location"].isna().any():
                    error = "Some rows have missing location values."

                else:
                    uploaded_df["location"] = (
                        uploaded_df["location"].astype(str).str.strip()
                    )

                    if uploaded_df["location"].eq("").any():
                        error = "Some rows have empty location values."

                    numeric_columns = [
                        "total_sqft", "bath", "balcony", "bhk", "price"
                    ]

                    if error is None:
                        for column in numeric_columns:
                            uploaded_df[column] = pd.to_numeric(
                                uploaded_df[column],
                                errors="coerce"
                            )

                        if uploaded_df[numeric_columns].isna().any().any():
                            error = (
                                "The dataset contains missing or "
                                "non-numeric values."
                            )

                        elif (uploaded_df["total_sqft"] <= 0).any():
                            error = "Total square feet must be greater than zero."

                        elif (uploaded_df["bath"] <= 0).any():
                            error = "Bathrooms must be greater than zero."

                        elif (uploaded_df["balcony"] < 0).any():
                            error = "Balcony count cannot be negative."

                        elif (uploaded_df["balcony"] > 5).any():
                            error = "Balcony count cannot exceed 5."

                        elif (uploaded_df["bhk"] <= 0).any():
                            error = "BHK must be greater than zero."

                        elif (uploaded_df["price"] <= 0).any():
                            error = "Price must be greater than zero."

                    if error is None:
                        if not os.path.exists(DATASET_BACKUP_PATH):
                            import shutil
                            shutil.copy2(DATASET_PATH, DATASET_BACKUP_PATH)

                        uploaded_df.to_csv(DATASET_PATH, index=False)

                        uploaded_filename = secure_filename(uploaded_file.filename)

                        with open(DATASET_NAME_PATH, "w", encoding="utf-8") as file:
                            file.write(uploaded_filename)

                        df = uploaded_df

                        locations = sorted(
                            location
                            for location in df["location"].unique()
                            if location and location.lower() != "nan"
                        )

                        price_insights = {
                            "total_properties": len(df),
                            "average_price": round(
                                float(df["price"].mean()), 2
                            ),
                            "minimum_price": round(
                                float(df["price"].min()), 2
                            ),
                            "maximum_price": round(
                                float(df["price"].max()), 2
                            )
                        }

                        message = (
                            f"Dataset uploaded successfully. "
                            f"{len(df)} records loaded."
                        )

            except Exception as exc:
                error = f"Could not read the uploaded CSV: {exc}"

    return render_template(
        "dataset_management.html",
        message=message,
        error=error,
        dataset_name=(
            open(DATASET_NAME_PATH, encoding="utf-8").read().strip()
            if os.path.exists(DATASET_NAME_PATH)
            else os.path.basename(DATASET_PATH)
        ),
        total_records=len(df),
        columns=list(df.columns)
    )

@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    if user_id == session.get("user_id"):
        flash(
            "You cannot delete your own administrator account.",
            "error"
        )
        return redirect(url_for("admin_dashboard"))

    connection = get_db_connection()

    try:
        connection.execute("BEGIN")

        user = connection.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if user is None:
            connection.rollback()
            flash("User account not found.", "error")
            return redirect(url_for("admin_dashboard"))

        if user["role"] == "admin":
            connection.rollback()
            flash(
                "Administrator accounts cannot be deleted here.",
                "error"
            )
            return redirect(url_for("admin_dashboard"))

        connection.execute(
            "DELETE FROM prediction_history WHERE user_id = ?",
            (user_id,)
        )

        connection.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,)
        )

        connection.commit()
        flash(
            "User and associated prediction history deleted.",
            "success"
        )

    except sqlite3.Error:
        connection.rollback()
        flash("Could not delete the user. Please try again.", "error")

    finally:
        connection.close()

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
        app.run(debug=False)
