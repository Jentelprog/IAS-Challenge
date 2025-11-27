# security/authentication.py

SECRET_KEY = "DT_IDS_SECRET_2025"  # Change this before final submission


def check_key(provided_key: str) -> bool:
    """Return True if the provided key matches the expected secret."""
    return provided_key == SECRET_KEY
