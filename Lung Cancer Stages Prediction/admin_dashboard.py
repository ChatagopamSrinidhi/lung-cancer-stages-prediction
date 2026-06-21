"""
Admin dashboard blueprint.
Handles: admin dashboard, user management, prediction history, HuggingFace model management.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from functools import wraps
from database import db, User, PredictionHistory, HuggingFaceModel
from hf_models import search_hf_models, get_curated_models, get_model_info
from xai_engine import get_risk_summary
import json
from datetime import datetime, timedelta

admin_bp = Blueprint('admin', __name__, url_prefix='/admin', template_folder='templates')


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required.', 'danger')
            return redirect(url_for('user.dashboard'))
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """Admin dashboard overview."""
    total_users = User.query.filter_by(role='user').count()
    total_admins = User.query.filter_by(role='admin').count()
    total_predictions = PredictionHistory.query.count()
    active_models = HuggingFaceModel.query.filter_by(is_enabled=True).count()

    # Recent predictions
    recent_predictions = PredictionHistory.query\
        .order_by(PredictionHistory.created_at.desc())\
        .limit(10).all()

    # Recent users
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()

    # Stage distribution
    stage_counts = {'Normal': 0, 'Stage1': 0, 'Stage2': 0, 'Stage3': 0}
    all_preds = PredictionHistory.query.all()
    for p in all_preds:
        if p.predicted_label in stage_counts:
            stage_counts[p.predicted_label] += 1

    # Predictions this week
    week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_predictions = PredictionHistory.query.filter(
        PredictionHistory.created_at >= week_ago
    ).count()

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           total_admins=total_admins,
                           total_predictions=total_predictions,
                           active_models=active_models,
                           recent_predictions=recent_predictions,
                           recent_users=recent_users,
                           stage_counts=stage_counts,
                           weekly_predictions=weekly_predictions)


# ─── User Management ────────────────────────────────────────────────────────────

@admin_bp.route('/users')
@admin_required
def users():
    """List all users with management options."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')

    query = User.query
    if search:
        query = query.filter(
            db.or_(
                User.username.contains(search),
                User.email.contains(search)
            )
        )

    users_paginated = query.order_by(User.created_at.desc())\
        .paginate(page=page, per_page=15, error_out=False)

    return render_template('admin/users.html', users=users_paginated, search=search)


@admin_bp.route('/users/<int:user_id>/toggle_status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Activate/deactivate a user."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'danger')
        return redirect(url_for('admin.users'))

    user.is_active_user = not user.is_active_user
    db.session.commit()

    status = 'activated' if user.is_active_user else 'deactivated'
    flash(f'User {user.username} has been {status}.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/make_admin', methods=['POST'])
@admin_required
def make_admin(user_id):
    """Promote a user to admin."""
    user = User.query.get_or_404(user_id)
    user.role = 'admin'
    db.session.commit()
    flash(f'User {user.username} is now an admin.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/remove_admin', methods=['POST'])
@admin_required
def remove_admin(user_id):
    """Demote admin to regular user."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot remove your own admin status.', 'danger')
        return redirect(url_for('admin.users'))
    user.role = 'user'
    db.session.commit()
    flash(f'User {user.username} is no longer an admin.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Delete a user and their predictions."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.users'))

    # Delete user's predictions
    PredictionHistory.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()

    flash(f'User {user.username} and their data have been deleted.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/users/<int:user_id>/predictions')
@admin_required
def user_predictions(user_id):
    """View a specific user's prediction history."""
    user = User.query.get_or_404(user_id)
    page = request.args.get('page', 1, type=int)

    predictions = PredictionHistory.query.filter_by(user_id=user_id)\
        .order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=15, error_out=False)

    return render_template('admin/user_predictions.html',
                           target_user=user,
                           predictions=predictions)


# ─── Prediction History (All Users) ─────────────────────────────────────────────

@admin_bp.route('/predictions')
@admin_required
def all_predictions():
    """View all predictions from all users."""
    page = request.args.get('page', 1, type=int)
    stage_filter = request.args.get('stage', '')
    user_filter = request.args.get('user_id', '', type=str)

    query = PredictionHistory.query

    if stage_filter:
        query = query.filter_by(predicted_label=stage_filter)
    if user_filter:
        query = query.filter_by(user_id=int(user_filter))

    predictions = query.order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=20, error_out=False)

    users_list = User.query.all()

    return render_template('admin/predictions.html',
                           predictions=predictions,
                           stage_filter=stage_filter,
                           user_filter=user_filter,
                           users_list=users_list)


@admin_bp.route('/predictions/<int:prediction_id>')
@admin_required
def prediction_detail(prediction_id):
    """View detailed prediction with XAI explanation (admin view)."""
    prediction = PredictionHistory.query.get_or_404(prediction_id)

    explanation = None
    if prediction.explanation:
        try:
            explanation = json.loads(prediction.explanation)
        except:
            from xai_engine import get_stage_explanation
            explanation = get_stage_explanation(
                prediction.predicted_label,
                prediction.confidence / 100,
                json.loads(prediction.probabilities) if prediction.probabilities else {}
            )

    probs_dict = {}
    if prediction.probabilities:
        try:
            probs_dict = json.loads(prediction.probabilities)
        except:
            pass

    return render_template('admin/prediction_detail.html',
                           prediction=prediction,
                           explanation=explanation,
                           probs_dict=probs_dict)


# ─── HuggingFace Model Management ───────────────────────────────────────────────

@admin_bp.route('/models')
@admin_required
def models():
    """View and manage HuggingFace models."""
    saved_models = HuggingFaceModel.query.order_by(HuggingFaceModel.added_at.desc()).all()
    curated = get_curated_models()

    return render_template('admin/models.html',
                           saved_models=saved_models,
                           curated_models=curated)


@admin_bp.route('/models/search', methods=['POST'])
@admin_required
def search_models():
    """Search HuggingFace for models."""
    query = request.form.get('query', 'lung cancer')
    results = search_hf_models(query=query, limit=15)
    saved_models = HuggingFaceModel.query.order_by(HuggingFaceModel.added_at.desc()).all()
    curated = get_curated_models()

    return render_template('admin/models.html',
                           saved_models=saved_models,
                           curated_models=curated,
                           search_results=results,
                           search_query=query)


@admin_bp.route('/models/add', methods=['POST'])
@admin_required
def add_model():
    """Add a HuggingFace model to the system."""
    model_id = request.form.get('model_id', '').strip()
    model_name = request.form.get('model_name', model_id)
    description = request.form.get('description', '')

    if not model_id:
        flash('Model ID is required.', 'danger')
        return redirect(url_for('admin.models'))

    # Check if already exists
    existing = HuggingFaceModel.query.filter_by(model_id=model_id).first()
    if existing:
        flash(f'Model {model_id} already exists.', 'warning')
        return redirect(url_for('admin.models'))

    hf_model = HuggingFaceModel(
        model_id=model_id,
        model_name=model_name,
        description=description,
        pipeline_tag='image-classification',
        is_enabled=False
    )
    db.session.add(hf_model)
    db.session.commit()

    flash(f'Model {model_name} added successfully.', 'success')
    return redirect(url_for('admin.models'))


@admin_bp.route('/models/<int:model_db_id>/toggle', methods=['POST'])
@admin_required
def toggle_model(model_db_id):
    """Enable/disable a model for user predictions."""
    model = HuggingFaceModel.query.get_or_404(model_db_id)
    model.is_enabled = not model.is_enabled
    db.session.commit()

    status = 'enabled' if model.is_enabled else 'disabled'
    flash(f'Model {model.model_name} has been {status}.', 'success')
    return redirect(url_for('admin.models'))


@admin_bp.route('/models/<int:model_db_id>/delete', methods=['POST'])
@admin_required
def delete_model(model_db_id):
    """Remove a model from the system."""
    model = HuggingFaceModel.query.get_or_404(model_db_id)
    db.session.delete(model)
    db.session.commit()

    flash(f'Model {model.model_name} has been removed.', 'success')
    return redirect(url_for('admin.models'))


@admin_bp.route('/models/<int:model_db_id>/info')
@admin_required
def model_detail(model_db_id):
    """Get detailed info about a specific model from HuggingFace."""
    model = HuggingFaceModel.query.get_or_404(model_db_id)
    hf_info = get_model_info(model.model_id)

    return render_template('admin/model_detail.html',
                           model=model,
                           hf_info=hf_info)


# ─── Admin API Endpoints ────────────────────────────────────────────────────────

@admin_bp.route('/api/stats')
@admin_required
def api_stats():
    """API endpoint for dashboard statistics."""
    total_users = User.query.filter_by(role='user').count()
    total_predictions = PredictionHistory.query.count()
    active_models = HuggingFaceModel.query.filter_by(is_enabled=True).count()

    stage_counts = {}
    for label in ['Normal', 'Stage1', 'Stage2', 'Stage3']:
        stage_counts[label] = PredictionHistory.query.filter_by(predicted_label=label).count()

    return jsonify({
        'total_users': total_users,
        'total_predictions': total_predictions,
        'active_models': active_models,
        'stage_counts': stage_counts,
    })
