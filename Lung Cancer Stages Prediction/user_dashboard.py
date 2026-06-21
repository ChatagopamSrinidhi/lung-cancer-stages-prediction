"""
User dashboard blueprint.
Handles: dashboard overview, image upload & prediction, prediction history, XAI explanations.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from database import db, PredictionHistory, HuggingFaceModel
from xai_engine import get_stage_explanation, generate_gradcam_visualization, get_risk_summary
from hf_models import predict_with_local_model, predict_with_hf_model, LABELS
import os
import json
import numpy as np
from datetime import datetime

user_bp = Blueprint('user', __name__, url_prefix='/user', template_folder='templates')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@user_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with overview stats."""
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc()).all()

    total_predictions = len(predictions)
    recent_predictions = predictions[:5]

    # Stats
    stage_counts = {'Normal': 0, 'Stage1': 0, 'Stage2': 0, 'Stage3': 0}
    for p in predictions:
        if p.predicted_label in stage_counts:
            stage_counts[p.predicted_label] += 1

    # Available models
    enabled_models = HuggingFaceModel.query.filter_by(is_enabled=True).all()

    return render_template('user/dashboard.html',
                           predictions=recent_predictions,
                           total_predictions=total_predictions,
                           stage_counts=stage_counts,
                           enabled_models=enabled_models)


@user_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Upload image and get prediction with XAI explanation."""
    # Get available models for selection
    enabled_models = HuggingFaceModel.query.filter_by(is_enabled=True).all()

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload JPG, PNG, or BMP.', 'danger')
            return redirect(request.url)

        # Save file
        filename = secure_filename(file.filename)
        # Add user id and timestamp to avoid collisions
        unique_name = f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        # Read image bytes
        with open(filepath, 'rb') as f:
            img_bytes = f.read()

        # Choose model
        model_choice = request.form.get('model_choice', 'local_cnn')

        try:
            if model_choice == 'local_cnn':
                label, confidence, probs_dict, raw = predict_with_local_model(img_bytes)
                model_used = 'Local CNN'
            else:
                label, confidence, probs_dict, raw = predict_with_hf_model(model_choice, img_bytes)
                model_used = model_choice

            # Generate XAI explanation
            explanation = get_stage_explanation(label, confidence, probs_dict)

            # Generate Grad-CAM visualization
            gradcam_b64 = None
            if model_choice == 'local_cnn':
                gradcam_b64 = generate_gradcam_visualization(img_bytes)

            # Save to database
            prediction_record = PredictionHistory(
                user_id=current_user.id,
                image_filename=unique_name,
                predicted_label=label,
                confidence=confidence * 100 if confidence <= 1 else confidence,
                probabilities=json.dumps(probs_dict),
                model_used=model_used,
                explanation=json.dumps(explanation),
                treatment_suggestions=json.dumps(explanation.get('treatment_suggestions', []))
            )
            db.session.add(prediction_record)
            db.session.commit()

            return render_template('user/prediction_result.html',
                                   prediction=prediction_record,
                                   explanation=explanation,
                                   gradcam_b64=gradcam_b64,
                                   probs_dict=probs_dict,
                                   filename=unique_name,
                                   enabled_models=enabled_models)

        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
            return render_template('user/predict.html', enabled_models=enabled_models, error=str(e))

    return render_template('user/predict.html', enabled_models=enabled_models)


@user_bp.route('/history')
@login_required
def history():
    """View prediction history."""
    page = request.args.get('page', 1, type=int)
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=10, error_out=False)

    return render_template('user/history.html', predictions=predictions)


@user_bp.route('/prediction/<int:prediction_id>')
@login_required
def prediction_detail(prediction_id):
    """View detailed prediction with XAI explanation."""
    prediction = PredictionHistory.query.get_or_404(prediction_id)

    # Ensure user can only view their own predictions
    if prediction.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('user.history'))

    # Load explanation
    explanation = None
    if prediction.explanation:
        try:
            explanation = json.loads(prediction.explanation)
        except:
            explanation = get_stage_explanation(
                prediction.predicted_label,
                prediction.confidence / 100,
                json.loads(prediction.probabilities) if prediction.probabilities else {}
            )

    # Load probabilities
    probs_dict = {}
    if prediction.probabilities:
        try:
            probs_dict = json.loads(prediction.probabilities)
        except:
            pass

    return render_template('user/prediction_detail.html',
                           prediction=prediction,
                           explanation=explanation,
                           probs_dict=probs_dict)


@user_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')

        if email and email != current_user.email:
            from database import User
            if User.query.filter_by(email=email).first():
                flash('Email already in use.', 'danger')
            else:
                current_user.email = email
                flash('Email updated.', 'success')

        if current_password and new_password:
            if current_user.check_password(current_password):
                if len(new_password) >= 6:
                    current_user.set_password(new_password)
                    flash('Password updated.', 'success')
                else:
                    flash('New password must be at least 6 characters.', 'danger')
            else:
                flash('Current password is incorrect.', 'danger')

        db.session.commit()

    return render_template('user/profile.html')
