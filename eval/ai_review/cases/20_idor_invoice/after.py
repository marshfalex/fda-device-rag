from flask import abort, jsonify
from flask_login import current_user, login_required

from .models import Invoice


@login_required
def get_invoice(invoice_id: int):
    invoice = Invoice.query.get(invoice_id)
    if invoice is None:
        abort(404)
    return jsonify(invoice.to_dict())
