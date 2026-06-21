"""
Database models for Lung Cancer Prediction System.
Uses Flask-SQLAlchemy with SQLite backend.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model with role-based access (admin/user)."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'admin' or 'user'
    is_active_user = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True,
                                  order_by='PredictionHistory.created_at.desc()')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'

    def __repr__(self):
        return f'<User {self.username}>'


class PredictionHistory(db.Model):
    """Stores each prediction made by users."""
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(256), nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.Text, nullable=True)  # JSON string
    model_used = db.Column(db.String(100), default='local_cnn')
    explanation = db.Column(db.Text, nullable=True)  # XAI explanation
    treatment_suggestions = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_label}>'


class HuggingFaceModel(db.Model):
    """Tracks HuggingFace models available for prediction."""
    __tablename__ = 'hf_models'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(256), unique=True, nullable=False)
    model_name = db.Column(db.String(256), nullable=False)
    description = db.Column(db.Text, nullable=True)
    downloads = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)
    pipeline_tag = db.Column(db.String(100), nullable=True)
    is_enabled = db.Column(db.Boolean, default=False)
    is_downloaded = db.Column(db.Boolean, default=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<HFModel {self.model_id}>'


def init_db(app):
    """Initialize database and create default admin user."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@lungcancer.ai',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("[DB] Default admin created: admin / admin123")
