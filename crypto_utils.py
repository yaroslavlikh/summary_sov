from cryptography.fernet import Fernet

from config import get_encryption_key

_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is None:
        key = get_encryption_key()
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def encrypt(text):
    if text is None:
        return None
    return _get_fernet().encrypt(text.encode('utf-8')).decode('utf-8')


def decrypt(token):
    if token is None:
        return None
    return _get_fernet().decrypt(token.encode('utf-8')).decode('utf-8')
