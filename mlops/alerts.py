"""
alerts.py
=========
Alert system for MLOps pipeline.
Sends notifications via Email + Slack when training/validation fails.

Setup:
1. Gmail alerts:
   - Enable 2FA: https://accounts.google.com/security
   - Create app password: https://myaccount.google.com/apppasswords
   - Set GMAIL_SENDER, GMAIL_PASSWORD below

2. Slack alerts:
   - Create webhook: https://api.slack.com/messaging/webhooks
   - Set SLACK_WEBHOOK below

Usage:
   alert_on_success('johor', 'Model trained with 94.64% accuracy')
   alert_on_failure('johor', 'HTTP Error 404 loading data')
"""

import smtplib
import requests
import json
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

# ── Configuration ──────────────────────────────────────────

GMAIL_SENDER = "your-email@gmail.com"          # Your Gmail address
GMAIL_PASSWORD = "xxxx-xxxx-xxxx-xxxx"         # Your app password (NOT Gmail password)

SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Email Alerts ───────────────────────────────────────────

def send_email_alert(subject: str, message: str, alert_type: str = "INFO"):
    """
    Send alert via Gmail.
    
    Args:
        subject: Email subject line
        message: Email body
        alert_type: "SUCCESS", "FAILURE", "WARNING"
    
    Returns:
        bool: True if sent successfully
    """
    try:
        # Build email
        msg = MIMEText(message)
        msg['Subject'] = f"[{alert_type}] {subject}"
        msg['From'] = GMAIL_SENDER
        msg['To'] = GMAIL_SENDER
        
        # Send via Gmail SMTP
        # Note: This requires an app-specific password, not your Gmail password
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, GMAIL_SENDER, msg.as_string())
        
        print(f"✅ Email alert sent: {subject}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print(f"❌ Email failed: Authentication error. Check GMAIL_PASSWORD")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ Email failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ── Slack Alerts ───────────────────────────────────────────

def send_slack_alert(status: str, state: str, message: str, details: dict = None):
    """
    Send alert to Slack webhook.
    
    Args:
        status: "SUCCESS", "FAILED", "WARNING"
        state: State name (e.g., "johor", "selangor")
        message: Main message
        details: Optional dict with extra info (accuracy, error_type, etc.)
    
    Returns:
        bool: True if sent successfully
    """
    try:
        # Color-code by status
        color_map = {
            "SUCCESS": "good",      # Green
            "FAILED": "danger",     # Red
            "WARNING": "#ff9900",   # Orange
        }
        color = color_map.get(status, "good")
        
        # Build Slack message
        text_field = message
        if details:
            for key, value in details.items():
                text_field += f"\n• {key}: {value}"
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{status}: {state.upper()} Training",
                "text": text_field,
                "footer": "GE-Insights MLOps",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        # Post to Slack
        response = requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Slack alert sent: {state} {status}")
            return True
        else:
            print(f"❌ Slack alert failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Slack alert failed: Timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Slack alert failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Slack alert failed: {e}")
        return False


# ── File Logging ───────────────────────────────────────────

def log_event(state: str, event_type: str, details: dict):
    """
    Log event to JSON file (audit trail).
    
    Args:
        state: State name
        event_type: "training_success", "training_failure", "validation_failure"
        details: Dict with event info
    """
    try:
        log_file = LOG_DIR / f"{state}_events.jsonl"
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "event_type": event_type,
            **details
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        print(f"✅ Event logged: {state} {event_type}")
        
    except Exception as e:
        print(f"❌ Logging failed: {e}")


# ── High-level Handlers ────────────────────────────────────

def alert_on_success(state: str, accuracy: float, model_version: str = "v1"):
    """
    Alert when training succeeds.
    
    Args:
        state: State name
        accuracy: Model accuracy (0.0-1.0)
        model_version: Version tag (e.g., "v1_2026_09_15")
    """
    message = f"""
Training SUCCESSFUL for {state}!

Accuracy: {accuracy:.2%}
Model: {model_version}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model saved to: backend/models/{state}/{model_version}/
    """
    
    # Email + Slack
    send_email_alert(
        subject=f"Training {state} succeeded",
        message=message,
        alert_type="SUCCESS"
    )
    
    send_slack_alert(
        status="SUCCESS",
        state=state,
        message=f"✅ Model trained with {accuracy:.2%} accuracy",
        details={
            "accuracy": f"{accuracy:.2%}",
            "version": model_version,
        }
    )
    
    # Log event
    log_event(state, "training_success", {
        "accuracy": accuracy,
        "model_version": model_version,
    })


def alert_on_failure(state: str, error: str, error_type: str = "unknown"):
    """
    Alert when training/validation fails.
    
    Args:
        state: State name
        error: Error message
        error_type: "data_load_error", "training_error", "validation_error"
    """
    message = f"""
⚠️  Training FAILED for {state}!

Error Type: {error_type}
Error Message: {error}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Action Required:
1. Check backend/logs/{state}_events.jsonl for details
2. Verify data sources (electiondata.my accessible?)
3. Check API credentials (Groq, yfinance)
4. Manual retry: python scripts/train_models.py {state}
    """
    
    # Email + Slack
    send_email_alert(
        subject=f"Training {state} FAILED",
        message=message,
        alert_type="FAILURE"
    )
    
    send_slack_alert(
        status="FAILED",
        state=state,
        message=f"❌ {error_type}",
        details={
            "error": error[:100],  # Truncate for readability
            "logs": f"backend/logs/{state}_events.jsonl",
        }
    )
    
    # Log event
    log_event(state, "training_failure", {
        "error": error,
        "error_type": error_type,
    })


# ── Test Alerts ────────────────────────────────────────────

def test_alerts():
    """
    Test email + Slack alerts (safe to run).
    Sends a test message to both channels.
    """
    print("\n" + "="*60)
    print("  TESTING ALERTS")
    print("="*60)
    
    test_state = "TEST"
    
    print("\n1. Testing Email Alert...")
    email_ok = send_email_alert(
        subject="Test email from MLOps",
        message="This is a test email from your MLOps pipeline.",
        alert_type="INFO"
    )
    
    print("\n2. Testing Slack Alert...")
    slack_ok = send_slack_alert(
        status="SUCCESS",
        state=test_state,
        message="This is a test Slack message from MLOps",
        details={"test": "true", "time": datetime.now().isoformat()}
    )
    
    print("\n3. Testing Event Logging...")
    log_event(test_state, "test_event", {"message": "Test log entry"})
    
    print("\n" + "="*60)
    if email_ok and slack_ok:
        print("✅ All alerts working!")
    else:
        print("⚠️  Some alerts failed. Check configuration above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run test when script is called directly
    test_alerts()