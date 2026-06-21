"""
Lung Cancer Stages Prediction — Main Application
Features:
  - User authentication (login/register)
  - User dashboard with prediction & history
  - Admin dashboard with user management & model management
  - HuggingFace pretrained model integration
  - XAI (Explainable AI) with treatment suggestions
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_login import LoginManager, login_required, current_user
import os
import numpy as np
from werkzeug.utils import secure_filename

# Local modules
import model_utils
from database import db, init_db, User
from auth import auth_bp
from user_dashboard import user_bp
from admin_dashboard import admin_bp

# ─── App Setup ───────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lung-cancer-ai-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lungcancer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Initialize Extensions ──────────────────────────────────────────────────────

init_db(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─── Register Blueprints ────────────────────────────────────────────────────────

app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(admin_bp)

# ─── Model Configuration ────────────────────────────────────────────────────────

try:
    _, Y = model_utils.load_dataset()
    num_classes = Y.shape[1] if Y.ndim == 2 else len(np.unique(Y))
except:
    num_classes = 4

ALL_LABELS = ['Normal', 'Stage1', 'Stage2', 'Stage3']
LABELS = ALL_LABELS[:num_classes]


# ─── Root Route ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Redirect based on auth status."""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('user.dashboard'))
    return redirect(url_for('auth.login'))


# ─── Legacy Admin Routes (Preprocess/Train/History) ─────────────────────────────

@app.route('/preprocess', methods=['GET', 'POST'])
@login_required
def preprocess_page():
    if not current_user.is_admin:
        return redirect(url_for('user.dashboard'))
    if request.method == 'POST':
        try:
            count = model_utils.preprocess_from_folder('Dataset')
            return jsonify({'success': True, 'count': count})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    exists = os.path.exists('model/X.txt.npy') and os.path.exists('model/Y.txt.npy')
    return render_template('preprocess.html', exists=exists)


@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    if not current_user.is_admin:
        return redirect(url_for('user.dashboard'))
    if request.method == 'POST':
        epochs = int(request.form.get('epochs', 20))
        batch = int(request.form.get('batch', 32))
        started = model_utils.start_training(epochs=epochs, batch_size=batch)
        return jsonify({'started': started})
    status = model_utils.training_status()
    return render_template('train.html', status=status)


@app.route('/train_status')
@login_required
def train_status():
    return jsonify(model_utils.training_status())


@app.route('/history')
@login_required
def history():
    if not current_user.is_admin:
        return redirect(url_for('user.dashboard'))
    history_data = model_utils.load_history()
    plot_b64 = model_utils.plot_history_as_base64()
    return render_template('history.html', history=history_data, plot_b64=plot_b64)


@app.route('/confusion_matrix')
@login_required
def confusion_matrix():
    if not current_user.is_admin:
        return redirect(url_for('user.dashboard'))
    cm_b64 = model_utils.get_confusion_matrix_base64()
    return render_template('confusion_matrix.html', cm_b64=cm_b64)


@app.route('/model_info')
@login_required
def model_info():
    if not current_user.is_admin:
        return redirect(url_for('user.dashboard'))
    try:
        loss, acc = model_utils.evaluate_on_test_split()
        return render_template('model_info.html', loss=loss, acc=acc, labels=LABELS)
    except Exception as e:
        return render_template('model_info.html', error=str(e), labels=LABELS)


# ─── File Serving ────────────────────────────────────────────────────────────────

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return redirect(url_for('auth.login'))


# ─── Run ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)
