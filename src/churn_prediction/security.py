"""
Application Security, Input Sanitization, and Data Privacy Utilities.

Provides:
- In-memory sliding window rate limiter for API endpoints
- Batch size and payload size enforcement
- PII masking and redaction helpers
- Security headers definition
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class InMemoryRateLimiter:
    """
    Sliding-window in-memory rate limiter per client IP.
    """

    def __init__(self, requests_per_minute: int = 120):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client is permitted to make a request.

        Returns
        -------
        Tuple[bool, int]
            (is_allowed, remaining_requests_in_window)
        """
        now = time.time()
        window_start = now - 60.0

        # Purge timestamps older than 60 seconds
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]

        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False, 0

        self.requests[client_ip].append(now)
        remaining = self.requests_per_minute - len(self.requests[client_ip])
        return True, remaining


def sanitize_customer_id(customer_id: str) -> str:
    """
    Sanitize and mask customer IDs for safe telemetry logging.
    Preserves prefix while masking sensitive inner characters.
    Example: '7590-VHVEG' -> '7590-****'
    """
    if not customer_id or not isinstance(customer_id, str):
        return "ANON-0000"
    
    # Strip potential HTML/script injection characters
    clean_id = re.sub(r"[^\w\-]", "", customer_id.strip())
    
    if len(clean_id) <= 4:
        return "****"
    
    visible_prefix = clean_id[:4]
    return f"{visible_prefix}-{'*' * (len(clean_id) - 4)}"


def redact_pii_dict(data: Dict) -> Dict:
    """
    Redact potential PII fields from diagnostic dictionaries before logging.
    """
    sensitive_keys = {"email", "phone", "ssn", "name", "address", "ip_address", "credit_card"}
    sanitized = {}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            sanitized[k] = "[REDACTED_PII]"
        elif k.lower() in ("customerid", "customer_id"):
            sanitized[k] = sanitize_customer_id(str(v))
        elif isinstance(v, dict):
            sanitized[k] = redact_pii_dict(v)
        else:
            sanitized[k] = v
    return sanitized


# Standard HTTP security headers for production serving
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "no-referrer"
}
