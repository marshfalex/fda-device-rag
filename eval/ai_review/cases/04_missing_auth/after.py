from flask import Blueprint, abort, jsonify
from flask_login import current_user, login_required

from .models import User, db

admin = Blueprint("admin", __name__)


@admin.get("/admin/users")
@login_required
def list_users():
    if not current_user.is_admin:
        abort(403)
    return jsonify([u.to_dict() for u in User.query.all()])


@admin.delete("/admin/users/<int:user_id>")
def delete_user(user_id: int):
    db.session.delete(User.query.get_or_404(user_id))
    db.session.commit()
    return "", 204
