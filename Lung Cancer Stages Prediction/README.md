# Lung Cancer Stages Prediction

An AI-powered web application for detecting and classifying lung cancer stages from CT scan images. Built with Flask, TensorFlow, and HuggingFace Transformers, featuring Explainable AI (XAI) to help users understand predictions with treatment suggestions and precautions.

---

## Features

### Authentication & Role-Based Access
- User registration and login with secure password hashing
- Role-based access control (Admin / User)
- Session management via Flask-Login

### User Dashboard
- Upload CT scan images and get instant predictions
- Choose between local CNN model or HuggingFace pretrained models
- View prediction history with confidence scores
- Detailed prediction results with XAI explanations
- Profile management (update email and password)

### Admin Dashboard
- Overview dashboard with stats and charts
- User management (activate, deactivate, promote, delete users)
- View all user prediction histories with filtering
- Dataset preprocessing and model training controls
- Training history visualization and confusion matrix
- Model performance evaluation
- HuggingFace model management (search, add, enable/disable)

### Explainable AI (XAI)
- Grad-CAM heatmap visualizations showing model attention areas
- Plain-language explanations of cancer stage classifications
- Risk level assessment with visual indicators
- Confidence analysis for each prediction

### Treatment Suggestions & Precautions
- Stage-specific treatment recommendations
- Precautions and lifestyle guidance per cancer stage
- Recommended next steps for patients
- Medical disclaimer included

### HuggingFace Integration
- Search and browse pretrained medical imaging models
- Curated list of lung cancer detection models
- Enable/disable models for prediction
- Use pretrained models alongside the local CNN

## Cancer Stages Classified

| Stage | Description |
|-------|-------------|
| **Normal** | No signs of lung cancer detected |
| **Stage 1** | Early-stage, localized cancer |
| **Stage 2** | Cancer has grown but remains regional |
| **Stage 3** | Advanced cancer, may have spread to nearby structures |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask, Flask-Login, Flask-SQLAlchemy |
| Database | SQLite (via SQLAlchemy ORM) |
| ML/DL | TensorFlow/Keras (CNN), HuggingFace Transformers |
| XAI | Grad-CAM, custom attention heatmaps |
| Frontend | Bootstrap 5, Chart.js, Font Awesome |
| Deployment | Docker, Waitress (WSGI) |

---

## Project Structure

```
├── app.py                  # Main Flask application
├── auth.py                 # Authentication blueprint (login/register/logout)
├── database.py             # SQLAlchemy models (User, PredictionHistory, HuggingFaceModel)
├── user_dashboard.py       # User routes (dashboard, predict, history, profile)
├── admin_dashboard.py      # Admin routes (dashboard, users, predictions, models)
├── model_utils.py          # CNN model training, prediction, preprocessing utilities
├── hf_models.py            # HuggingFace model search, download, and prediction
├── xai_engine.py           # Explainable AI engine (Grad-CAM, treatment info)
├── wsgi.py                 # Production WSGI entry point (Waitress)
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Python dependencies
├── Dataset/                # Training dataset
│   ├── normal/             # Normal lung CT images
│   ├── Stage1/             # Stage 1 cancer images
│   ├── Stage2/             # Stage 2 cancer images
│   └── Stage3/             # Stage 3 cancer images
├── model/                  # Saved model weights and training data
│   ├── model_weights.weights.h5
│   ├── history.pckl
│   ├── X.txt.npy
│   └── Y.txt.npy
├── templates/
│   ├── dashboard_base.html # Base layout with sidebar navigation
│   ├── auth/               # Login and register pages
│   ├── user/               # User dashboard templates
│   ├── admin/              # Admin dashboard templates
│   ├── preprocess.html     # Dataset preprocessing page
│   ├── train.html          # Model training page
│   ├── history.html        # Training history page
│   ├── confusion_matrix.html
│   └── model_info.html     # Model evaluation page
├── static/                 # Static assets (images)
├── uploads/                # User-uploaded CT scan images
└── testImages/             # Sample test images
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Lung Cancer Stages Prediction"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

### Default Admin Credentials
| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin123` |

> Change the admin password after first login.

---

## Docker Deployment

```bash
# Build the image
docker build -t lung-cancer-prediction .

# Run the container
docker run -p 8000:8000 lung-cancer-prediction
```

The production server runs on **port 8000** using Waitress.

---

## Usage Guide

### For Users
1. Register a new account at `/register`
2. Log in and access the **User Dashboard**
3. Navigate to **Predict** and upload a lung CT scan image
4. Select a prediction model (Local CNN or HuggingFace)
5. View results with XAI explanations, treatment suggestions, and precautions
6. Check **History** to review past predictions

### For Admins
1. Log in with admin credentials
2. Access the **Admin Dashboard** for an overview of system activity
3. Manage users under **Users** (activate/deactivate, assign roles, delete)
4. View all prediction histories under **Predictions**
5. Use **Preprocess** to prepare the dataset, then **Train** to train the CNN model
6. Browse and manage HuggingFace models under **Models**

---

## API Endpoints

### Authentication
| Method | Route | Description |
|--------|-------|-------------|
| GET/POST | `/login` | User login |
| GET/POST | `/register` | User registration |
| GET | `/logout` | Logout |

### User Routes (`/user`)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/user/dashboard` | User dashboard |
| GET/POST | `/user/predict` | Upload and predict |
| GET | `/user/history` | Prediction history |
| GET | `/user/prediction/<id>` | Prediction detail with XAI |
| GET/POST | `/user/profile` | Profile management |

### Admin Routes (`/admin`)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/admin/dashboard` | Admin overview |
| GET | `/admin/users` | User management |
| GET | `/admin/predictions` | All predictions |
| GET | `/admin/models` | HuggingFace model management |
| POST | `/admin/models/add` | Add a HuggingFace model |

### Legacy Admin Routes
| Method | Route | Description |
|--------|-------|-------------|
| GET/POST | `/preprocess` | Dataset preprocessing |
| GET/POST | `/train` | Model training |
| GET | `/train_status` | Training progress (JSON) |
| GET | `/history` | Training history |
| GET | `/confusion_matrix` | Confusion matrix |
| GET | `/model_info` | Model evaluation metrics |

---

## Disclaimer

This application is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.
