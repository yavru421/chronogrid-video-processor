"""Chronogrid licensing and freemium management."""
import base64
import hmac
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# License secret (base64 encoded 32-byte key)
LICENSE_SECRET = base64.b64decode("jIMw+tRbQKw36ZsENwsNR6nGvDJs/gXxBkXZR0XCNFE=")

class LicenseError(Exception):
    """Raised when license validation fails."""
    pass

class FreemiumLimitError(Exception):
    """Raised when free tier limit is exceeded."""
    pass

def get_config_dir() -> Path:
    """Get the user config directory for Chronogrid."""
    if os.name == 'nt':  # Windows
        config_dir = Path(os.environ.get('APPDATA', '')) / 'Chronogrid'
    else:  # macOS/Linux
        config_dir = Path.home() / '.config' / 'chronogrid'

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_license_file() -> Path:
    """Get the license file path."""
    return get_config_dir() / 'license.key'

def get_usage_file() -> Path:
    """Get the usage tracking file path."""
    return get_config_dir() / 'usage.json'

def save_license_key(license_key: str) -> None:
    """Save license key to config file."""
    license_file = get_license_file()
    license_file.write_text(license_key.strip(), encoding='utf-8')

def load_license_key() -> Optional[str]:
    """Load license key from config file."""
    license_file = get_license_file()
    if license_file.exists():
        return license_file.read_text(encoding='utf-8').strip()
    return None

def validate_license_key(license_key: str) -> str:
    """
    Validate a license key and return the associated email.

    Raises LicenseError if invalid.
    """
    try:
        # Decode license key
        license_data = base64.b64decode(license_key).decode()
        payload, signature_b64 = license_data.rsplit('|', 1)

        email, expiry_timestamp_str = payload.split('|', 1)
        expiry_timestamp = int(expiry_timestamp_str)

        # Verify expiry
        if datetime.now().timestamp() > expiry_timestamp:
            raise LicenseError("License has expired")

        # Verify signature
        expected_signature = hmac.new(LICENSE_SECRET, payload.encode(), hashlib.sha256).digest()
        provided_signature = base64.b64decode(signature_b64)

        if not hmac.compare_digest(expected_signature, provided_signature):
            raise LicenseError("Invalid license key")

        return email

    except Exception as e:
        if isinstance(e, LicenseError):
            raise
        raise LicenseError("Invalid license key format")

def has_valid_license() -> bool:
    """Check if user has a valid license."""
    license_key = load_license_key()
    if not license_key:
        return False

    try:
        validate_license_key(license_key)
        return True
    except LicenseError:
        return False

def get_current_month_key() -> str:
    """Get the key for current month usage tracking."""
    now = datetime.now()
    return f"{now.year}-{now.month:02d}"

def get_usage_count() -> int:
    """Get current month's AI analysis usage count."""
    usage_file = get_usage_file()
    month_key = get_current_month_key()

    if usage_file.exists():
        try:
            data = json.loads(usage_file.read_text(encoding='utf-8'))
            return data.get(month_key, 0)
        except (json.JSONDecodeError, KeyError):
            pass

    return 0

def increment_usage_count() -> None:
    """Increment the usage count for current month."""
    usage_file = get_usage_file()
    month_key = get_current_month_key()

    # Load existing data
    data = {}
    if usage_file.exists():
        try:
            data = json.loads(usage_file.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            data = {}

    # Increment count
    data[month_key] = data.get(month_key, 0) + 1

    # Save back
    usage_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

def check_ai_analysis_allowed() -> None:
    """
    Check if AI analysis is allowed.

    Raises FreemiumLimitError if free limit exceeded and no valid license.
    """
    if has_valid_license():
        return  # Premium user, unlimited

    usage_count = get_usage_count()
    if usage_count >= 5:
        raise FreemiumLimitError(
            f"Free tier limit reached ({usage_count}/5 analyses this month). "
            "Please upgrade to Chronogrid Premium for unlimited AI analysis."
        )

def record_ai_analysis() -> None:
    """Record that an AI analysis was performed."""
    increment_usage_count()

def get_upgrade_message() -> str:
    """Get appropriate upgrade message based on current state."""
    if has_valid_license():
        return ""

    usage_count = get_usage_count()
    remaining = max(0, 5 - usage_count)

    if remaining == 0:
        return (
            "You've reached the free tier limit of 5 AI analyses per month.\n\n"
            "Upgrade to Chronogrid Premium for:\n"
            "• Unlimited AI analysis\n"
            "• Priority support\n"
            "• All future features\n\n"
            "Visit: [Your Stripe Payment Link] to upgrade for $49 (one-time)"
        )
    else:
        return f"You have {remaining} free AI analyses remaining this month."

def activate_license(license_key: str) -> str:
    """
    Activate a license key.

    Returns the associated email if successful.
    Raises LicenseError if invalid.
    """
    email = validate_license_key(license_key)
    save_license_key(license_key)
    return email