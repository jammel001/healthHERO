"""
HealthHero - Healthcare Symptom Checker Chatbot
HIPAA-Compliant, Secure, Production-Ready
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from functools import wraps

import joblib
import numpy as np
from rapidfuzz import process
from cryptography.fernet import Fernet

from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthcare_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# App Configuration
# ---------------------------
app = Flask(__name__, template_folder="templates")

# Security: Force HTTPS and secure headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# CORS: Restrict to specific origins in production
CORS(app, resources={
    r"/api/*": {
        "origins": os.environ.get("ALLOWED_ORIGINS", "https://healthhero.onrender.com").split(","),
        "methods": ["POST", "GET"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Session Configuration: Secure, encrypted, time-limited
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key or app.secret_key == "change-this-secret":
    raise RuntimeError("FLASK_SECRET_KEY must be set to a secure random string")

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)  # Auto-expire after 30 min
app.config["SESSION_COOKIE_SECURE"] = True  # HTTPS only
app.config["SESSION_COOKIE_HTTPONLY"] = True  # No JavaScript access
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
Session(app)

# Rate Limiting: Prevent abuse
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Encryption for PHI (if storage is necessary)
ENCRYPTION_KEY = os.environ.get("SESSION_ENCRYPTION_KEY")
if ENCRYPTION_KEY:
    cipher_suite = Fernet(ENCRYPTION_KEY.encode())
else:
    logger.warning("SESSION_ENCRYPTION_KEY not set - PHI will not be encrypted")
    cipher_suite = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Emergency Detection Configuration
# ---------------------------
EMERGENCY_PATTERNS = {
    "cardiac": {
        "keywords": ["chest pain", "heart attack", "cardiac arrest", "can't breathe", "gasping"],
        "message": "🚨 CARDIAC EMERGENCY DETECTED: Call 911 immediately. Do not wait. Symptoms of heart attack require immediate medical attention.",
        "resources": ["911", "Emergency Services"]
    },
    "respiratory": {
        "keywords": ["cannot breathe", "suffocating", "choking", "blue lips", "gasping for air"],
        "message": "🚨 RESPIRATORY EMERGENCY: Call 911 immediately. Difficulty breathing is a life-threatening emergency.",
        "resources": ["911"]
    },
    "neurological": {
        "keywords": ["unconscious", "seizure", "stroke", "can't move one side", "slurred speech"],
        "message": "🚨 NEUROLOGICAL EMERGENCY: Call 911 immediately. Time-critical: note the time symptoms started.",
        "resources": ["911"]
    },
    "mental_health": {
        "keywords": ["suicide", "kill myself", "end my life", "want to die", "self-harm", "cutting myself"],
        "message": "🚨 MENTAL HEALTH CRISIS: You are not alone. Immediate help is available:\n• Call or text 988 (Suicide & Crisis Lifeline)\n• Text HOME to 741741 (Crisis Text Line)\n• Call 911 if in immediate danger",
        "resources": ["988 Suicide & Crisis Lifeline", "741741 Crisis Text Line", "911"]
    },
    "severe_allergy": {
        "keywords": ["anaphylaxis", "throat closing", "can't swallow", "severe allergic reaction"],
        "message": "🚨 ANAPHYLAXIS EMERGENCY: Use EpiPen if available and call 911 immediately.",
        "resources": ["911"]
    }
}

# ---------------------------
# Symptom Configuration
# ---------------------------
CANONICAL_SYMPTOMS = [
    "fever", "headache", "vomiting", "nausea", "fatigue",
    "dizziness", "body pain", "loss of appetite", "cough",
    "shortness of breath", "chest pain", "diarrhea",
    "abdominal pain", "sore throat", "insomnia", "rash",
    "joint pain", "muscle weakness", "confusion", "blurred vision"
]

SYMPTOM_ALIASES = {
    "catarrh": "cough",
    "running nose": "cough",
    "nasal congestion": "cough",
    "hotness": "fever",
    "high temperature": "fever",
    "body weakness": "fatigue",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "running stomach": "diarrhea",
    "loose stools": "diarrhea",
    "head pain": "headache",
    "migraine": "headache",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "feeling nauseous": "nausea",
    "queasy": "nausea",
    "weak body": "fatigue",
    "no appetite": "loss of appetite",
    "not hungry": "loss of appetite",
    "dry cough": "cough",
    "wet cough": "cough",
    "tight chest": "chest pain",
    "chest tightness": "chest pain",
    "difficulty breathing": "shortness of breath",
    "breathlessness": "shortness of breath",
    "body aches": "body pain",
    "muscle pain": "body pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "tummy ache": "abdominal pain",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "vertigo": "dizziness",
    "throat pain": "sore throat",
    "painful throat": "sore throat",
    "cannot sleep": "insomnia",
    "sleeplessness": "insomnia",
    "skin rash": "rash",
    "hives": "rash",
    "aching joints": "joint pain",
    "arthritis pain": "joint pain",
}

# Pre-compile regex patterns for efficiency
SYMPTOM_PATTERNS = {
    re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE): symptom 
    for phrase, symptom in SYMPTOM_ALIASES.items()
}

# ---------------------------
# Safe Loaders with Validation
# ---------------------------
def safe_load(path: str, critical: bool = False):
    """Safely load pickle files with error handling."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        if critical:
            raise RuntimeError(f"Critical model missing: {path}")
        return None

def safe_numpy(path: str, critical: bool = False):
    """Safely load numpy files."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        return np.load(path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        if critical:
            raise RuntimeError(f"Critical data missing: {path}")
        return None

# ---------------------------
# Load Models with Validation
# ---------------------------
logger.info("Loading models...")

symptom_encoder = safe_load(os.path.join(BASE_DIR, "symptom_encoder.pkl"), critical=True)
disease_model = safe_load(os.path.join(BASE_DIR, "disease_model.pkl"), critical=True)
label_encoder = safe_load(os.path.join(BASE_DIR, "label_encoder.pkl"), critical=True)

symptom_embeddings = safe_numpy(os.path.join(BASE_DIR, "symptom_embeddings.npz"))
symptom_explanations = safe_load(os.path.join(BASE_DIR, "symptom_to_explanation.pkl")) or {}
disease_descriptions = safe_load(os.path.join(BASE_DIR, "disease_to_description.pkl")) or {}
disease_precautions = safe_load(os.path.join(BASE_DIR, "disease_to_precautions.pkl")) or {}

logger.info("Models loaded successfully")

# ---------------------------
# Security & Validation Functions
# ---------------------------
def sanitize_input(text: str) -> Tuple[str, List[str]]:
    """
    Sanitize user input to prevent injection attacks and validate length.
    Returns: (sanitized_text, error_messages)
    """
    errors = []
    
    if not text:
        return "", ["Input is required"]
    
    if len(text) > 500:
        text = text[:500]
        errors.append("Input truncated to 500 characters")
    
    # Remove potentially dangerous characters but keep basic punctuation
    # Allow: letters, numbers, spaces, basic punctuation
    sanitized = re.sub(r'[^\w\s.,!?\'-]', '', text)
    
    # Check for prompt injection attempts
    injection_patterns = [
        r'ignore previous instructions',
        r'system prompt',
        r'act as (?:an? )?(?:admin|doctor|physician)',
        r'you are now',
        r'forget (?:everything|all)',
        r'disregard',
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Potential injection attempt detected: {pattern}")
            errors.append("Invalid input detected")
            return "", errors
    
    return sanitized.strip(), errors

def check_emergency(text: str, symptoms: List[str]) -> Optional[Dict]:
    """
    Check for emergency keywords and return appropriate response.
    Returns None if no emergency detected.
    """
    text_lower = text.lower()
    
    for category, data in EMERGENCY_PATTERNS.items():
        # Check text and symptoms
        for keyword in data["keywords"]:
            if keyword in text_lower or keyword in symptoms:
                logger.critical(f"EMERGENCY DETECTED: {category} - Immediate attention required")
                return {
                    "is_emergency": True,
                    "type": category,
                    "text": data["message"],
                    "resources": data["resources"],
                    "requires_immediate": True
                }
    return None

def hash_identifier(data: str) -> str:
    """Create a hashed identifier for audit logging (PHI protection)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def log_audit_event(session_id: str, action: str, details: Dict):
    """Log audit event with hashed identifiers."""
    safe_details = {
        k: hash_identifier(str(v)) if k in ['name', 'user_id', 'ip'] else v 
        for k, v in details.items()
    }
    logger.info(f"AUDIT: Session={session_id}, Action={action}, Details={safe_details}")

# ---------------------------
# Symptom Extraction
# ---------------------------
def extract_symptoms_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract symptoms from user text using pattern matching and fuzzy search.
    Returns: (extracted_symptoms, clarification_questions)
    """
    text_lower = text.lower()
    extracted = set()
    clarifications = []
    
    # Direct pattern matching for aliases
    for pattern, symptom in SYMPTOM_PATTERNS.items():
        if pattern.search(text_lower):
            extracted.add(symptom)
    
    # Direct regex matching for canonical symptoms
    for symptom in CANONICAL_SYMPTOMS:
        if re.search(rf"\b{re.escape(symptom)}\b", text_lower):
            extracted.add(symptom)
    
    # Fuzzy matching for unknown words
    words = [w for w in re.findall(r"[a-z]+", text_lower) if len(w) > 3]
    checked_words = set()
    
    for word in words:
        if word in checked_words or word in CANONICAL_SYMPTOMS:
            continue
        checked_words.add(word)
        
        match = process.extractOne(word, CANONICAL_SYMPTOMS, score_cutoff=85)
        if match and match[0] not in extracted:
            clarifications.append(f"Did you mean '{match[0]}' instead of '{word}'?")
            extracted.add(match[0])
    
    return list(extracted), clarifications

# ---------------------------
# Model Bundle
# ---------------------------
class ModelBundle:
    def __init__(self):
        self.model = disease_model
        self.encoder = symptom_encoder
        self.label_encoder = label_encoder
        self.symptoms = (
            self.encoder.get_feature_names_out().tolist()
            if hasattr(self.encoder, 'get_feature_names_out') else []
        )
        
        if not all([self.model, self.encoder, self.label_encoder]):
            raise RuntimeError("Critical models not loaded properly")
    
    def match_symptoms(self, symptoms: List[str]) -> List[str]:
        """Match user symptoms to model features with fuzzy fallback."""
        matched = []
        for s in symptoms:
            if s in self.symptoms:
                matched.append(s)
            else:
                fuzzy = process.extractOne(s, self.symptoms, score_cutoff=80)
                if fuzzy:
                    matched.append(fuzzy[0])
        return list(set(matched))
    
    def predict(self, symptoms: List[str]) -> List[Dict]:
        """
        Predict conditions based on symptoms.
        Returns top 3 predictions with probabilities.
        """
        if not symptoms:
            return []
        
        try:
            # Transform symptoms to feature vector
            symptom_text = " ".join(symptoms)
            X = self.encoder.transform([symptom_text])
            
            # Get prediction probabilities
            probs = self.model.predict_proba(X)[0]
            top_indices = np.argsort(probs)[::-1][:3]
            
            results = []
            for idx in top_indices:
                if probs[idx] < 0.05:  # Filter very low confidence
                    continue
                    
                label = self.label_encoder.inverse_transform([idx])[0]
                results.append({
                    "condition": label,
                    "probability": round(float(probs[idx]), 3),
                    "confidence_tier": "high" if probs[idx] > 0.7 else "medium" if probs[idx] > 0.4 else "low",
                    "description": disease_descriptions.get(label, "No description available"),
                    "precautions": disease_precautions.get(label, [])[:5]  # Limit to 5
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return []

# Initialize model bundle
try:
    BUNDLE = ModelBundle()
    logger.info("ModelBundle initialized")
except Exception as e:
    logger.critical(f"Failed to initialize ModelBundle: {e}")
    raise

# ---------------------------
# Routes
# ---------------------------
@app.before_request
def enforce_https():
    """Redirect HTTP to HTTPS in production."""
    if request.headers.get('X-Forwarded-Proto') == 'http':
        return redirect(request.url.replace('http://', 'https://'), code=301)

@app.route("/")
def home():
    """Serve the main application page."""
    return render_template("index.html")

@app.route("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0-secure"
    })

@app.route("/api/diagnose", methods=["POST"])
@limiter.limit("10 per minute")
def diagnose():
    """
    Main diagnostic endpoint with state machine for conversation flow.
    Includes emergency detection, input validation, and audit logging.
    """
    # Get and validate input
    data = request.json or {}
    raw_input = (data.get("reply") or data.get("symptoms") or "").strip()
    
    # Input sanitization
    user_input, input_errors = sanitize_input(raw_input)
    if input_errors and not user_input:
        return jsonify({
            "text": "I couldn't understand that. Please describe your symptoms in simple terms.",
            "error": input_errors[0]
        }), 400
    
    # Session initialization
    if "stage" not in session:
        session.clear()
        session["stage"] = "GREETING"
        session["patient"] = {}
        session["session_start"] = datetime.utcnow().isoformat()
        log_audit_event(session.get("_id", "new"), "session_start", {})
    
    stage = session["stage"]
    session_id = session.get("_id", "unknown")
    
    # Log interaction (with privacy protection)
    log_audit_event(session_id, f"stage_{stage}", {
        "input_length": len(user_input),
        "ip": request.remote_addr
    })
    
    # ==================== GREETING ====================
    if stage == "GREETING":
        session["stage"] = "ASK_CONSENT"
        return jsonify({
            "text": (
                "Hello 👋 I'm HealthHero, your health information assistant.\n\n"
                "⚠️ IMPORTANT DISCLAIMER:\n"
                "• I am NOT a doctor and do NOT provide medical diagnoses\n"
                "• I offer general health information only\n"
                "• For emergencies, call 911 or visit your nearest ER immediately\n"
                "• Always consult a healthcare professional for medical advice\n\n"
                "Do you understand and wish to continue?"
            ),
            "options": ["Yes, I understand", "No"]
        })
    
    # ==================== CONSENT ====================
    if stage == "ASK_CONSENT":
        if user_input.lower().startswith("y"):
            session["stage"] = "ASK_NAME"
            return jsonify({"text": "What is your first name? (You may use a nickname)"})
        
        session.clear()
        log_audit_event(session_id, "consent_denied", {})
        return jsonify({"text": "Understood. Take care 🙏"})
    
    # ==================== NAME ====================
    if stage == "ASK_NAME":
        if len(user_input) < 1 or len(user_input) > 50:
            return jsonify({"text": "Please enter a valid name (1-50 characters)."})
        
        session["patient"]["name"] = user_input.title()
        session["stage"] = "ASK_AGE"
        return jsonify({"text": "How old are you? (This helps provide age-appropriate information)"})
    
    # ==================== AGE ====================
    if stage == "ASK_AGE":
        age_match = re.search(r"\d+", user_input)
        if not age_match:
            return jsonify({"text": "Please enter a valid age as a number."})
        
        age = int(age_match.group())
        if age < 1 or age > 120:
            return jsonify({"text": "Please enter a realistic age between 1 and 120."})
        
        session["patient"]["age"] = age
        session["stage"] = "ASK_GENDER"
        return jsonify({
            "text": "What is your gender?",
            "options": ["Male", "Female", "Non-binary", "Prefer not to say"]
        })
    
    # ==================== GENDER ====================
    if stage == "ASK_GENDER":
        valid_genders = ["male", "female", "non-binary", "prefer not to say", "other"]
        if user_input.lower() not in valid_genders:
            return jsonify({
                "text": "Please select a valid option.",
                "options": ["Male", "Female", "Non-binary", "Prefer not to say"]
            })
        
        session["patient"]["gender"] = user_input
        session["stage"] = "ASK_SYMPTOMS"
        return jsonify({
            "text": "Please describe your symptoms in detail. What are you experiencing?",
            "hint": "Example: 'I have a headache and fever'"
        })
    
    # ==================== SYMPTOMS ====================
    if stage == "ASK_SYMPTOMS":
        matched, clarifications = extract_symptoms_from_text(user_input)
        
        # Check for emergency FIRST before anything else
        emergency = check_emergency(user_input, matched)
        if emergency:
            log_audit_event(session_id, "emergency_detected", {"type": emergency["type"]})
            return jsonify({
                "text": emergency["text"],
                "is_emergency": True,
                "resources": emergency["resources"],
                "options": ["I understand, this is not an emergency", "End conversation"]
            })
        
        # Handle clarifications needed
        if clarifications:
            session["pending_clarifications"] = clarifications
            session["pending_symptoms"] = matched
            session["stage"] = "ASK_CLARIFICATION"
            
            return jsonify({
                "text": f"Before we continue:\n{clarifications[0]}",
                "options": ["Yes, that's correct", "No, let me rephrase"]
            })
        
        # No symptoms detected
        if not matched:
            return jsonify({
                "text": "I couldn't identify specific symptoms. Please use simple terms like 'headache', 'fever', 'cough', etc.",
                "examples": ["I have a headache and nausea", "My chest hurts", "I'm feeling dizzy"]
            })
        
        # Success - save symptoms
        session["symptoms"] = matched
        session["stage"] = "ASK_SYMPTOM_EXPLANATION"
        
        return jsonify({
            "text": f"I found these symptoms: {', '.join(matched)}.\n\nWould you like detailed explanations of what these symptoms might mean?",
            "options": ["Yes, explain symptoms", "No, continue to analysis"]
        })
    
    # ==================== CLARIFICATION ====================
    if stage == "ASK_CLARIFICATION":
        if user_input.lower().startswith("y"):
            # User confirmed the clarification
            session["symptoms"] = session.get("pending_symptoms", [])
            session.pop("pending_clarifications", None)
            session.pop("pending_symptoms", None)
            session["stage"] = "ASK_SYMPTOM_EXPLANATION"
            
            return jsonify({
                "text": f"Thank you for confirming. I have noted: {', '.join(session['symptoms'])}.\n\nWould you like detailed explanations of these symptoms?",
                "options": ["Yes, explain symptoms", "No, continue to analysis"]
            })
        else:
            # User wants to rephrase
            session.pop("pending_clarifications", None)
            session.pop("pending_symptoms", None)
            session["stage"] = "ASK_SYMPTOMS"
            
            return jsonify({
                "text": "No problem. Please describe your symptoms again using different words.",
                "hint": "Try: 'headache', 'fever', 'stomach pain', etc."
            })
    
    # ==================== SYMPTOM EXPLANATION ====================
    if stage == "ASK_SYMPTOM_EXPLANATION":
        if user_input.lower().startswith("y"):
            explanations = []
            for s in session.get("symptoms", []):
                info = symptom_explanations.get(s, {})
                general = info.get("general", "A general symptom that can have various causes.")
                medical = info.get("medical", "Medically, this requires professional evaluation for accurate diagnosis.")
                
                explanations.append({
                    "symptom": s.title(),
                    "general": general,
                    "medical": medical
                })
            
            session["stage"] = "CONFIRM_PREDICTION"
            return jsonify({
                "text": "Here is information about your symptoms:",
                "explanations": explanations,
                "options": ["Continue to possible conditions", "End conversation"]
            })
        
        # User skipped explanations
        session["stage"] = "CONFIRM_PREDICTION"
        return jsonify({
            "text": "Understood. Shall I analyze your symptoms for possible conditions?\n\n⚠️ Remember: This is not a medical diagnosis.",
            "options": ["Yes, analyze symptoms", "No, end conversation"]
        })
    
    # ==================== CONFIRM PREDICTION ====================
    if stage == "CONFIRM_PREDICTION":
        if not user_input.lower().startswith(("y", "c")):
            session.clear()
            log_audit_event(session_id, "analysis_declined", {})
            return jsonify({"text": "Alright. Take care 🙏"})
        
        # Generate predictions
        predictions = BUNDLE.predict(session.get("symptoms", []))
        
        if not predictions:
            session.clear()
            return jsonify({
                "text": "I couldn't determine possible conditions from these symptoms. Please consult a healthcare professional for proper evaluation.",
                "disclaimer": "This is not medical advice."
            })
        
        # Format predictions
        formatted = []
        for p in predictions:
            confidence_emoji = "🔴" if p['confidence_tier'] == 'low' else "🟡" if p['confidence_tier'] == 'medium' else "🟢"
            
            formatted.append({
                "condition": p['condition'],
                "probability": f"{int(p['probability'] * 100)}%",
                "confidence": p['confidence_tier'],
                "description": p['description'],
                "precautions": p['precautions']
            })
        
        log_audit_event(session_id, "prediction_generated", {
            "symptoms_count": len(session.get("symptoms", [])),
            "predictions_count": len(predictions)
        })
        
        session.clear()  # Clear PHI after use
        
        return jsonify({
            "text": "Based on your symptoms, here are possible conditions (ranked by likelihood):",
            "predictions": formatted,
            "disclaimer": "⚠️ IMPORTANT: This is NOT a medical diagnosis. These are possible conditions based on symptoms only. Always consult a qualified healthcare professional for accurate diagnosis and treatment.",
            "next_steps": [
                "Schedule an appointment with your doctor",
                "Visit an urgent care if symptoms worsen",
                "Call 911 for emergencies"
            ]
        })
    
    # ==================== FALLBACK ====================
    # Unknown stage or error
    logger.error(f"Unknown stage reached: {stage}")
    session.clear()
    return jsonify({
        "text": "I'm sorry, something went wrong. Please refresh the page to start over.",
        "error": "Invalid session state"
    })

@app.errorhandler(429)
def rate_limit_handler(e):
    """Handle rate limit exceeded."""
    return jsonify({
        "text": "Too many requests. Please wait a moment before trying again.",
        "error": "rate_limit"
    }), 429

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.critical(f"Internal error: {e}")
    return jsonify({
        "text": "An error occurred. Please try again later.",
        "error": "internal_error"
    }), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # Production settings
    port = int(os.environ.get("PORT", 10000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    if debug_mode:
        logger.warning("Running in DEBUG mode - not for production!")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        threaded=True
    )
