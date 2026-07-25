"""Shared rate limiter instance.

Kept in its own module (rather than defined in main.py) so routes.py can
import and decorate endpoints with it without a circular import.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
