from flask import Blueprint, jsonify, request
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

main = Blueprint('main', __name__)

@main.route('/api/static-response', methods=['GET'])
def static_response():
    logging.info('Static response endpoint called.')

    try:
        # Return static JSON
        response = {
            "status": "success",
            "message": "This is a static JSON response"
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

# Optional: Handle 404 errors
@main.app_errorhandler(404)
def not_found_error(error):
    logging.warning(f"404 error: {request.path} not found.")
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404
