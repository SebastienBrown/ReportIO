from flask import Flask, request, jsonify
import sys
import os
from flask_cors import CORS
import threading
# app.py - Flask backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64
import io
import os
from datetime import datetime
import logging
import requests

# Make sure we can import from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.multimodal_orchestrator import run_multimodal_pipeline
from pdf_utils import generate_pdf_from_data

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Log buffer (global) ----
log_lines = []
last_result = None 

def log(msg):
    print(msg)  # still prints to console
    log_lines.append(msg)
    

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    log_lines.clear()

    # Run your orchestration in a background thread
    def run():
        global last_result
        result = run_multimodal_pipeline(query, logger=log)
        last_result = result
        log("[✅] Final answer ready.")

    threading.Thread(target=run).start()

    return jsonify({"status": "started"}), 202

@app.route("/api/last_result", methods=["GET"])
def get_last_result():
    return jsonify(last_result or {})

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": log_lines})

# Email configuration - UPDATE THESE WITH YOUR CREDENTIALS
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',  # Gmail SMTP
    'smtp_port': 587,
    'sender_email': 'sebastienbrown1@gmail.com',  # Your email
    'sender_password': 'tgil iuum ullq rgqj',   # App password (not regular password)
    'sender_name': 'PDF Export Service'
}

@app.route('/api/send-pdf', methods=['POST'])
def send_pdf():
    try:
        print("in call")
        data = request.get_json()
        recipient_email = data.get('email')
        subject = data.get('subject', 'Your PDF Export')
        
        pdf_data = data.get('pdf_data')  # Base64 encoded PDF (legacy/optional)
        
        if pdf_data:
            # Decode base64 PDF data
            pdf_bytes = base64.b64decode(pdf_data.split(',')[1]) if ',' in pdf_data else base64.b64decode(pdf_data)
        else:
            # Generate PDF from raw data
            query = data.get('query')
            answer = data.get('answer')
            snippets = data.get('snippets', [])
            videos = data.get('videos', [])
            
            if not query or not answer:
                return jsonify({'error': 'Query and answer are required if pdf_data is not provided'}), 400
            
            pdf_bytes = generate_pdf_from_data(query, answer, snippets, videos)
        
        if not recipient_email or not pdf_bytes:
            return jsonify({'error': 'Email and PDF content are required'}), 400
            
        # Send email with PDF attachment
        success = send_email_with_pdf(recipient_email, pdf_bytes, subject)
        
        if success:
            logger.info("PDF sent successfully")
            return jsonify({'message': 'PDF sent successfully!'})
        else:
            logger.error("Failed to send email")
            return jsonify({'error': 'Failed to send email'}), 500
            
    except Exception as e:
        logger.error(f"Error sending PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

def send_email_with_pdf(recipient_email, pdf_bytes, subject):
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Email body
        body = f"""
        Hello,
        
        Please find your PDF export attached.
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Best regards,
        PDF Export Service
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_attachment = MIMEBase('application', 'pdf')
        pdf_attachment.set_payload(pdf_bytes)
        encoders.encode_base64(pdf_attachment)
        
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}'
        )
        
        msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"PDF sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(debug=True)
