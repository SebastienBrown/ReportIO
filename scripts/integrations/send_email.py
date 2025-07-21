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

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Email configuration - UPDATE THESE WITH YOUR CREDENTIALS
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',  # Gmail SMTP
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',  # Your email
    'sender_password': 'your-app-password',   # App password (not regular password)
    'sender_name': 'PDF Export Service'
}

@app.route('/send-pdf', methods=['POST'])
def send_pdf():
    try:
        data = request.get_json()
        recipient_email = data.get('email')
        pdf_data = data.get('pdf_data')  # Base64 encoded PDF
        subject = data.get('subject', 'Your PDF Export')
        
        if not recipient_email or not pdf_data:
            return jsonify({'error': 'Email and PDF data are required'}), 400
            
        # Decode base64 PDF data
        pdf_bytes = base64.b64decode(pdf_data.split(',')[1])  # Remove data:application/pdf;base64, prefix
        
        # Send email with PDF attachment
        success = send_email_with_pdf(recipient_email, pdf_bytes, subject)
        
        if success:
            return jsonify({'message': 'PDF sent successfully!'})
        else:
            return jsonify({'error': 'Failed to send email'}), 500
            
    except Exception as e:
        logger.error(f"Error sending PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

def send_email_with_pdf(recipient_email, pdf_bytes, subject):
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = "sebastienbrown@gmail.com"
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

if __name__ == '__main__':
    print("Starting PDF Email Server...")
    print("Make sure to update EMAIL_CONFIG with your credentials!")
    app.run(debug=True, host='0.0.0.0', port=5000)