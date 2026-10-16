# House Price Prediction

## Project Overview

House Price Prediction is a machine learning web application that estimates house prices based on property details such as location, total square footage, number of bedrooms (BHK), bathrooms, and balconies.

The application provides an easy-to-use interface for entering property details and viewing estimated prices.

## Features

- House price prediction using a trained machine learning model
- Location-based price estimation
- User registration and login
- Email OTP verification for login
- Google login with Firebase authentication
- Prediction history
- Comparison of property predictions
- Market trend visualizations
- Admin dashboard for user management
- Dataset management features
- Model evaluation and prediction error analysis

## Technologies Used

- Python
- Flask
- SQLite
- Pandas and NumPy
- Scikit-learn
- HTML and CSS
- Firebase Authentication
- SMTP email verification
- Matplotlib

## Project Structure

```text
House-Price-Prediction/
├── dataset/
├── models/
├── results/
├── static/
├── templates/
├── app.py
├── requirements.txt
├── analyze_data.py
├── train_model.py
├── train_model_v2.py
├── train_model_v3.py
├── train_model_v4.py
├── train_model_v5.py
├── final_train.py
├── evaluate_models.py
├── check_locations.py
├── create_admin.py
├── test_prediction.py
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/harika-yallanti/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Before starting the application, configure the following environment variables:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Protects Flask sessions |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | Path to the Firebase service-account JSON file |
| `MAIL_FROM` | Sender email address |
| `SMTP_HOST` | SMTP server hostname |
| `SMTP_PORT` | SMTP server port; defaults to `587` |
| `SMTP_USERNAME` | SMTP login username |
| `SMTP_PASSWORD` | SMTP login password or app password |
| `COOKIE_SECURE` | Set to `true` when using HTTPS in production; use `false` for local HTTP development |

Keep secret values and the Firebase service-account JSON file outside the repository. Never commit them to GitHub.

### 5. Run the application

```bash
python app.py
```

Open the local address shown in the terminal in your browser.

## Machine Learning

The project includes scripts for data analysis, data preparation, model training, model comparison, and prediction evaluation. The trained model is stored in the `models` directory.

Predictions are estimates based on the available dataset and may differ from actual market prices.

## Security

- Authentication includes email OTP verification.
- Google authentication uses Firebase.
- Sensitive configuration should be stored outside source code.
- Local database files and credentials should not be uploaded to GitHub.

## Limitations

Prediction accuracy depends on the quality, coverage, and freshness of the dataset. Estimated prices should be used for informational purposes and not as a substitute for professional property valuation.

