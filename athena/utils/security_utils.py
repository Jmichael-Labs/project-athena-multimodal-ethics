"""
Security Utilities for Project Athena

Comprehensive security management including encryption, access control,
audit logging, and secure content handling.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import hashlib
import hmac
import secrets
import base64
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# Security imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import jwt
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
except ImportError as e:
    logging.warning(f"Some security dependencies not available: {e}")

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """Access levels for content and operations."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class SecurityCredentials:
    """Security credentials for access control."""
    user_id: str
    access_level: AccessLevel
    permissions: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    issued_at: datetime = field(default_factory=datetime.now)

@dataclass
class AuditEvent:
    """Security audit event."""
    event_id: str
    event_type: str
    user_id: str
    resource: str
    action: str
    result: str  # success, failure, denied
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None

class EncryptionHandler:
    """
    Advanced encryption and cryptographic operations.
    
    Handles symmetric and asymmetric encryption, key management,
    and secure data protection for ethics evaluation pipeline.
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption handler.
        
        Args:
            master_key: Master encryption key (generated if None)
        """
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        
        # Initialize password context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto"
        )
        
        logger.info("Encryption Handler initialized")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        try:
            return self.fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt string and return base64 encoded result."""
        encrypted_bytes = self.encrypt_data(text.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('ascii')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt base64 encoded string."""
        encrypted_bytes = base64.b64decode(encrypted_text.encode('ascii'))
        decrypted_bytes = self.decrypt_data(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(password, hashed_password)
    
    def generate_salt(self, length: int = 32) -> bytes:
        """Generate cryptographically secure salt."""
        return secrets.token_bytes(length)
    
    def derive_key(self, password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(password.encode('utf-8'))
    
    def generate_hmac(self, data: bytes, key: Optional[bytes] = None) -> str:
        """Generate HMAC signature for data integrity."""
        if key is None:
            key = self.master_key
        
        signature = hmac.new(key, data, hashlib.sha256).hexdigest()
        return signature
    
    def verify_hmac(self, data: bytes, signature: str, key: Optional[bytes] = None) -> bool:
        """Verify HMAC signature."""
        if key is None:
            key = self.master_key
        
        expected_signature = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_signature)
    
    @staticmethod
    def generate_rsa_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA public/private key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current encryption setup."""
        return {
            "algorithm": "Fernet (AES 128)",
            "key_generated": True,
            "password_hashing": "bcrypt",
            "hmac_algorithm": "SHA256"
        }

class AccessController:
    """
    Role-based access control system.
    
    Manages user permissions, access levels, and authorization
    for ethics evaluation operations and sensitive content.
    """
    
    def __init__(self, config):
        """Initialize access controller."""
        self.config = config
        
        # User database (in production would be external)
        self.users = {}
        self.roles = {}
        self.permissions = {}
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = timedelta(hours=8)
        
        # JWT settings
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        
        logger.info("Access Controller initialized")
    
    def _initialize_default_roles(self) -> None:
        """Initialize default roles and permissions."""
        
        # Define permissions
        permissions = [
            "content.evaluate",
            "content.view",
            "content.modify",
            "content.delete",
            "system.monitor",
            "system.configure",
            "admin.users",
            "admin.audit"
        ]
        
        for perm in permissions:
            self.permissions[perm] = {"description": f"Permission for {perm}"}
        
        # Define roles
        self.roles = {
            "viewer": {
                "permissions": {"content.view", "system.monitor"},
                "access_level": AccessLevel.PUBLIC,
                "description": "Read-only access to evaluations"
            },
            "evaluator": {
                "permissions": {"content.evaluate", "content.view", "system.monitor"},
                "access_level": AccessLevel.RESTRICTED,
                "description": "Can perform ethics evaluations"
            },
            "moderator": {
                "permissions": {"content.evaluate", "content.view", "content.modify", "system.monitor"},
                "access_level": AccessLevel.CONFIDENTIAL,
                "description": "Can evaluate and moderate content"
            },
            "admin": {
                "permissions": set(permissions),
                "access_level": AccessLevel.SECRET,
                "description": "Full system access"
            }
        }
    
    def create_user(
        self, 
        user_id: str, 
        password: str, 
        role: str,
        additional_permissions: Optional[Set[str]] = None
    ) -> bool:
        """Create new user with specified role."""
        
        if user_id in self.users:
            logger.warning(f"User already exists: {user_id}")
            return False
        
        if role not in self.roles:
            logger.error(f"Invalid role: {role}")
            return False
        
        # Hash password
        from .security_utils import EncryptionHandler
        encryption_handler = EncryptionHandler()
        password_hash = encryption_handler.hash_password(password)
        
        # Create user
        user_permissions = self.roles[role]["permissions"].copy()
        if additional_permissions:
            user_permissions.update(additional_permissions)
        
        self.users[user_id] = {
            "password_hash": password_hash,
            "role": role,
            "permissions": user_permissions,
            "access_level": self.roles[role]["access_level"],
            "created_at": datetime.now(),
            "last_login": None,
            "active": True
        }
        
        logger.info(f"User created: {user_id} with role {role}")
        return True
    
    def authenticate_user(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        
        if user_id not in self.users:
            logger.warning(f"Authentication failed: user not found: {user_id}")
            return None
        
        user = self.users[user_id]
        
        if not user["active"]:
            logger.warning(f"Authentication failed: user inactive: {user_id}")
            return None
        
        # Verify password
        from .security_utils import EncryptionHandler
        encryption_handler = EncryptionHandler()
        
        if not encryption_handler.verify_password(password, user["password_hash"]):
            logger.warning(f"Authentication failed: invalid password: {user_id}")
            return None
        
        # Create session token
        session_token = self._create_session_token(user_id)
        
        # Update user login time
        user["last_login"] = datetime.now()
        
        logger.info(f"User authenticated: {user_id}")
        return session_token
    
    def _create_session_token(self, user_id: str) -> str:
        """Create JWT session token."""
        
        user = self.users[user_id]
        
        payload = {
            "user_id": user_id,
            "role": user["role"],
            "permissions": list(user["permissions"]),
            "access_level": user["access_level"].value,
            "iat": datetime.now(),
            "exp": datetime.now() + self.session_timeout
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store session
        self.active_sessions[token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + self.session_timeout
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[SecurityCredentials]:
        """Validate session token and return credentials."""
        
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if session exists and is valid
            if token not in self.active_sessions:
                return None
            
            session = self.active_sessions[token]
            if datetime.now() > session["expires_at"]:
                del self.active_sessions[token]
                return None
            
            # Create credentials
            credentials = SecurityCredentials(
                user_id=payload["user_id"],
                access_level=AccessLevel(payload["access_level"]),
                permissions=set(payload["permissions"]),
                expires_at=session["expires_at"],
                issued_at=session["created_at"]
            )
            
            return credentials
        
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def check_permission(self, credentials: SecurityCredentials, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in credentials.permissions
    
    def check_access_level(
        self, 
        credentials: SecurityCredentials, 
        required_level: AccessLevel
    ) -> bool:
        """Check if user has required access level."""
        
        level_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.RESTRICTED: 1,
            AccessLevel.CONFIDENTIAL: 2,
            AccessLevel.SECRET: 3
        }
        
        user_level = level_hierarchy.get(credentials.access_level, 0)
        required_level_num = level_hierarchy.get(required_level, 0)
        
        return user_level >= required_level_num
    
    def revoke_token(self, token: str) -> bool:
        """Revoke session token."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_tokens = [
            token for token, session in self.active_sessions.items()
            if current_time > session["expires_at"]
        ]
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        return len(expired_tokens)
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)."""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        return {
            "user_id": user_id,
            "role": user["role"],
            "permissions": list(user["permissions"]),
            "access_level": user["access_level"].value,
            "created_at": user["created_at"].isoformat(),
            "last_login": user["last_login"].isoformat() if user["last_login"] else None,
            "active": user["active"]
        }
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        return {
            "total_users": len(self.users),
            "active_users": sum(1 for user in self.users.values() if user["active"]),
            "active_sessions": len(self.active_sessions),
            "total_roles": len(self.roles),
            "total_permissions": len(self.permissions)
        }

class SecurityManager:
    """
    Comprehensive security management system.
    
    Integrates encryption, access control, audit logging,
    and security monitoring for the ethics evaluation system.
    """
    
    def __init__(self, config):
        """Initialize security manager."""
        self.config = config
        
        # Initialize components
        self.encryption = EncryptionHandler()
        self.access_control = AccessController(config)
        
        # Audit logging
        self.audit_events = []
        self.max_audit_events = 10000
        
        # Security monitoring
        self.security_alerts = []
        self.failed_attempts = {}
        self.rate_limits = {}
        
        # Security settings
        self.security_settings = {
            "max_failed_attempts": 5,
            "lockout_duration": timedelta(minutes=30),
            "rate_limit_window": timedelta(minutes=1),
            "rate_limit_max_requests": 100,
            "audit_retention_days": 90
        }
        
        logger.info("Security Manager initialized")
    
    async def secure_content_evaluation(
        self, 
        content: Any,
        credentials: SecurityCredentials,
        operation: str = "evaluate"
    ) -> Tuple[bool, Optional[str]]:
        """
        Secure wrapper for content evaluation operations.
        
        Args:
            content: Content to evaluate
            credentials: User credentials
            operation: Type of operation
        
        Returns:
            Tuple of (authorized, error_message)
        """
        
        # Check permissions
        required_permission = f"content.{operation}"
        if not self.access_control.check_permission(credentials, required_permission):
            await self._log_audit_event(
                event_type="access_denied",
                user_id=credentials.user_id,
                resource="content",
                action=operation,
                result="denied",
                details={"reason": "insufficient_permissions"}
            )
            return False, "Insufficient permissions"
        
        # Check access level for sensitive content
        if self._is_sensitive_content(content):
            if not self.access_control.check_access_level(credentials, AccessLevel.RESTRICTED):
                await self._log_audit_event(
                    event_type="access_denied",
                    user_id=credentials.user_id,
                    resource="sensitive_content",
                    action=operation,
                    result="denied",
                    details={"reason": "insufficient_access_level"}
                )
                return False, "Insufficient access level for sensitive content"
        
        # Rate limiting
        if not self._check_rate_limit(credentials.user_id):
            await self._log_audit_event(
                event_type="rate_limit_exceeded",
                user_id=credentials.user_id,
                resource="content",
                action=operation,
                result="denied",
                details={"reason": "rate_limit_exceeded"}
            )
            return False, "Rate limit exceeded"
        
        # Log successful access
        await self._log_audit_event(
            event_type="content_access",
            user_id=credentials.user_id,
            resource="content",
            action=operation,
            result="success"
        )
        
        return True, None
    
    def _is_sensitive_content(self, content: Any) -> bool:
        """Determine if content is sensitive and requires higher access level."""
        # Placeholder logic - would implement actual sensitivity detection
        # Could check for PII, explicit content, etc.
        return False
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        current_time = datetime.now()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Remove old requests outside the window
        window_start = current_time - self.security_settings["rate_limit_window"]
        self.rate_limits[user_id] = [
            request_time for request_time in self.rate_limits[user_id]
            if request_time > window_start
        ]
        
        # Check if under limit
        if len(self.rate_limits[user_id]) >= self.security_settings["rate_limit_max_requests"]:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
    
    async def _log_audit_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Log security audit event."""
        
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            ip_address=ip_address
        )
        
        # Add to audit log
        self.audit_events.append(event)
        
        # Limit audit log size
        if len(self.audit_events) > self.max_audit_events:
            self.audit_events = self.audit_events[-self.max_audit_events//2:]
        
        # Check for security alerts
        await self._check_security_alerts(event)
        
        logger.info(f"Audit event: {event_type} by {user_id} on {resource} - {result}")
    
    async def _check_security_alerts(self, event: AuditEvent) -> None:
        """Check if event should trigger security alerts."""
        
        # Track failed authentication attempts
        if event.event_type == "authentication" and event.result == "failure":
            user_id = event.user_id
            
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []
            
            self.failed_attempts[user_id].append(event.timestamp)
            
            # Check if exceeds threshold
            recent_failures = [
                attempt for attempt in self.failed_attempts[user_id]
                if (event.timestamp - attempt).total_seconds() < 3600  # 1 hour window
            ]
            
            if len(recent_failures) >= self.security_settings["max_failed_attempts"]:
                await self._generate_security_alert(
                    alert_type="multiple_failed_logins",
                    user_id=user_id,
                    details={"failure_count": len(recent_failures)}
                )
        
        # Check for suspicious access patterns
        if event.result == "denied":
            # Multiple access denials might indicate probing
            recent_denials = [
                e for e in self.audit_events[-100:]  # Last 100 events
                if e.user_id == event.user_id and e.result == "denied"
                and (event.timestamp - e.timestamp).total_seconds() < 300  # 5 minutes
            ]
            
            if len(recent_denials) >= 10:
                await self._generate_security_alert(
                    alert_type="suspicious_access_pattern",
                    user_id=event.user_id,
                    details={"denial_count": len(recent_denials)}
                )
    
    async def _generate_security_alert(
        self,
        alert_type: str,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Generate security alert."""
        
        alert = {
            "alert_id": secrets.token_urlsafe(16),
            "alert_type": alert_type,
            "user_id": user_id,
            "timestamp": datetime.now(),
            "details": details,
            "severity": "high" if alert_type == "multiple_failed_logins" else "medium"
        }
        
        self.security_alerts.append(alert)
        
        logger.warning(f"Security alert: {alert_type} for user {user_id}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage."""
        return self.encryption.encrypt_string(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption.decrypt_string(encrypted_data)
    
    def get_audit_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get filtered audit events."""
        
        filtered_events = self.audit_events
        
        # Apply filters
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    def get_security_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security alerts."""
        return sorted(
            self.security_alerts,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        
        recent_time = datetime.now() - timedelta(hours=24)
        recent_events = [e for e in self.audit_events if e.timestamp >= recent_time]
        
        return {
            "total_audit_events": len(self.audit_events),
            "recent_events_24h": len(recent_events),
            "security_alerts": len(self.security_alerts),
            "active_sessions": len(self.access_control.active_sessions),
            "failed_attempts_24h": sum(
                len([a for a in attempts if (datetime.now() - a).total_seconds() < 86400])
                for attempts in self.failed_attempts.values()
            ),
            "encryption_enabled": True,
            "access_control_enabled": True,
            "audit_logging_enabled": True
        }
    
    async def cleanup_security_data(self) -> None:
        """Clean up old security data."""
        
        # Clean up old audit events
        retention_cutoff = datetime.now() - timedelta(days=self.security_settings["audit_retention_days"])
        self.audit_events = [e for e in self.audit_events if e.timestamp >= retention_cutoff]
        
        # Clean up old failed attempts
        cleanup_cutoff = datetime.now() - timedelta(hours=24)
        for user_id in list(self.failed_attempts.keys()):
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if (datetime.now() - attempt).total_seconds() < 86400
            ]
            
            if not self.failed_attempts[user_id]:
                del self.failed_attempts[user_id]
        
        # Clean up expired sessions
        self.access_control.cleanup_expired_sessions()
        
        logger.info("Security data cleanup completed")