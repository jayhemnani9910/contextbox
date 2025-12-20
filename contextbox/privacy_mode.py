"""
Privacy Mode for ContextBox

This module provides data encryption and redaction features for protecting
sensitive information in captured contexts.
"""

import logging
import hashlib
import re
import json
import base64
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class PrivacyError(Exception):
    """Custom exception for privacy mode operations."""
    pass


class DataEncryption:
    """
    Handles data encryption and decryption for ContextBox.
    """
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption system.
        
        Args:
            password: Master password for encryption (if None, generates temporary)
        """
        self.logger = logging.getLogger(__name__)
        self._fernet = None
        self._password_provided = password is not None
        
        if password:
            self._setup_encryption_with_password(password)
        else:
            self._setup_encryption_temporary()
    
    def _setup_encryption_with_password(self, password: str) -> None:
        """Setup encryption using provided password."""
        try:
            # Generate salt for key derivation
            salt = os.urandom(16)
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
            # Create Fernet instance
            self._fernet = Fernet(key)
            
            # Store salt for decryption (would need persistent storage in production)
            self._salt = salt
            self.logger.info("Encryption initialized with provided password")
            
        except Exception as e:
            self.logger.error(f"Failed to setup encryption: {e}")
            raise PrivacyError(f"Encryption setup failed: {e}")
    
    def _setup_encryption_temporary(self) -> None:
        """Setup encryption with temporary key (for session only)."""
        try:
            self._fernet = Fernet(Fernet.generate_key())
            self._salt = None
            self.logger.info("Encryption initialized with temporary key")
        except Exception as e:
            self.logger.error(f"Failed to setup temporary encryption: {e}")
            raise PrivacyError(f"Temporary encryption setup failed: {e}")
    
    def encrypt_text(self, text: str) -> str:
        """Encrypt text data."""
        if not self._fernet:
            raise PrivacyError("Encryption not initialized")
        
        try:
            encrypted_data = self._fernet.encrypt(text.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Text encryption failed: {e}")
            raise PrivacyError(f"Text encryption failed: {e}")
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text data."""
        if not self._fernet:
            raise PrivacyError("Encryption not initialized")
        
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_text.encode())
            decrypted_data = self._fernet.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Text decryption failed: {e}")
            raise PrivacyError(f"Text decryption failed: {e}")
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data."""
        return self.encrypt_text(json.dumps(data))
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        decrypted_text = self.decrypt_text(encrypted_data)
        return json.loads(decrypted_text)
    
    def is_encrypted(self, data: str) -> bool:
        """Check if data appears to be encrypted."""
        try:
            base64.urlsafe_b64decode(data.encode())
            return True
        except Exception:
            return False
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get encryption system information."""
        return {
            'initialized': self._fernet is not None,
            'password_protected': self._password_provided,
            'has_salt': self._salt is not None
        }


class PIIRedactor:
    """
    Handles PII (Personally Identifiable Information) detection and redaction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PII redaction system.
        
        Args:
            config: Configuration for redaction rules
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # PII detection patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'mac_address': re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'),
            'postal_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b')
        }
        
        # Configuration
        self.auto_detect_enabled = self.config.get('auto_detect', True)
        self.redaction_pattern = self.config.get('pattern', '[REDACTED]')
        self.preserve_length = self.config.get('preserve_length', False)
        
        # Using regex-based PII detection (comprehensive and reliable)
        self.logger.info("PII redaction initialized with regex-based detection")
    

    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected PII items with type and position
        """
        detected_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            for match in pattern.finditer(text):
                pii_item = {
                    'type': pii_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'replacement': self._generate_replacement(pii_type, match.group())
                }
                detected_pii.append(pii_item)
        
        # Sort by position
        detected_pii.sort(key=lambda x: x['start'])
        
        return detected_pii
    
    def redact_pii(self, text: str, pii_items: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            pii_items: Pre-detected PII items (optional)
            
        Returns:
            Tuple of (redacted_text, detection_report)
        """
        if pii_items is None:
            pii_items = self.detect_pii(text)
        
        if not pii_items:
            return text, {'detected': 0, 'redacted': 0, 'types': {}}
        
        # Apply redactions in reverse order to maintain positions
        redacted_text = text
        redactions_applied = []
        
        for pii_item in reversed(pii_items):
            start = pii_item['start']
            end = pii_item['end']
            replacement = pii_item['replacement']
            
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            redactions_applied.append(pii_item.copy())
        
        # Generate report
        report = {
            'detected': len(pii_items),
            'redacted': len(redactions_applied),
            'types': {}
        }
        
        for pii_item in pii_items:
            pii_type = pii_item['type']
            if pii_type not in report['types']:
                report['types'][pii_type] = 0
            report['types'][pii_type] += 1
        
        return redacted_text, report
    
    def _generate_replacement(self, pii_type: str, original_text: str) -> str:
        """Generate replacement text for PII."""
        if self.preserve_length:
            return '*' * len(original_text)
        else:
            return self.redaction_pattern
    
    def get_detection_stats(self, text: str) -> Dict[str, Any]:
        """Get PII detection statistics for text."""
        detected_pii = self.detect_pii(text)
        
        stats = {
            'total_detected': len(detected_pii),
            'types_found': {},
            'redaction_needed': len(detected_pii) > 0
        }
        
        for pii_item in detected_pii:
            pii_type = pii_item['type']
            if pii_type not in stats['types_found']:
                stats['types_found'][pii_type] = 0
            stats['types_found'][pii_type] += 1
        
        return stats


class PrivacyMode:
    """
    Main privacy mode handler that combines encryption and redaction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize privacy mode.
        
        Args:
            config: Configuration dictionary
                - enable_encryption: Enable data encryption
                - enable_redaction: Enable PII redaction
                - master_password: Password for encryption
                - redaction_config: Configuration for PII redaction
                - auto_protect: Automatically protect sensitive data
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encryption = None
        self.redactor = None
        
        # Initialize encryption
        if self.config.get('enable_encryption', True):
            try:
                password = self.config.get('master_password')
                self.encryption = DataEncryption(password)
                self.logger.info("Encryption enabled")
            except PrivacyError as e:
                self.logger.error(f"Encryption initialization failed: {e}")
                self.encryption = None
        
        # Initialize PII redaction
        if self.config.get('enable_redaction', True):
            try:
                redaction_config = self.config.get('redaction_config', {})
                self.redactor = PIIRedactor(redaction_config)
                self.logger.info("PII redaction enabled")
            except Exception as e:
                self.logger.error(f"PII redaction initialization failed: {e}")
                self.redactor = None
        
        # Auto-protect settings
        self.auto_protect = self.config.get('auto_protect', True)
        
        self.logger.info("Privacy mode initialized")
    
    def protect_capture(self, capture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply privacy protection to capture data.
        
        Args:
            capture_data: Raw capture data
            
        Returns:
            Protected capture data
        """
        protected_data = capture_data.copy()
        protection_log = {
            'encryption_applied': False,
            'redaction_applied': False,
            'protection_time': datetime.now().isoformat(),
            'pii_detected': {},
            'sensitive_fields_encrypted': []
        }
        
        try:
            # Encrypt sensitive text fields
            if self.encryption and self._should_encrypt_field('clipboard_text'):
                if 'clipboard_text' in protected_data and protected_data['clipboard_text']:
                    protected_data['clipboard_text'] = self.encryption.encrypt_text(
                        protected_data['clipboard_text']
                    )
                    protection_log['sensitive_fields_encrypted'].append('clipboard_text')
                    protection_log['encryption_applied'] = True
            
            # Redact PII from text fields
            if self.redactor and self._should_redact_field('clipboard_text'):
                if 'clipboard_text' in protected_data and protected_data['clipboard_text']:
                    text = protected_data['clipboard_text']
                    redacted_text, redaction_report = self.redactor.redact_pii(text)
                    protected_data['clipboard_text'] = redacted_text
                    protection_log['redaction_applied'] = True
                    protection_log['pii_detected']['clipboard_text'] = redaction_report
            
            # Redact PII from notes
            if self.redactor and self._should_redact_field('notes'):
                if 'notes' in protected_data and protected_data['notes']:
                    text = protected_data['notes']
                    redacted_text, redaction_report = self.redactor.redact_pii(text)
                    protected_data['notes'] = redacted_text
                    protection_log['redaction_applied'] = True
                    protection_log['pii_detected']['notes'] = redaction_report
            
            # Redact PII from window titles
            if self.redactor and self._should_redact_field('source_window'):
                if 'source_window' in protected_data and protected_data['source_window']:
                    text = protected_data['source_window']
                    redacted_text, redaction_report = self.redactor.redact_pii(text)
                    protected_data['source_window'] = redacted_text
                    protection_log['redaction_applied'] = True
                    protection_log['pii_detected']['source_window'] = redaction_report
            
            # Add protection metadata
            if 'privacy_protection' not in protected_data:
                protected_data['privacy_protection'] = {}
            protected_data['privacy_protection'].update(protection_log)
            
            self.logger.info("Capture data protected successfully")
            
        except Exception as e:
            self.logger.error(f"Privacy protection failed: {e}")
            protection_log['protection_error'] = str(e)
            protected_data['privacy_protection'] = protection_log
        
        return protected_data
    
    def unprotect_capture(self, protected_data: Dict[str, Any], password: Optional[str] = None) -> Dict[str, Any]:
        """
        Remove privacy protection from capture data.
        
        Args:
            protected_data: Protected capture data
            password: Password for decryption (if required)
            
        Returns:
            Unprotected capture data
        """
        unprotected_data = protected_data.copy()
        
        try:
            # Decrypt encrypted fields
            if self.encryption:
                for field in ['clipboard_text']:
                    if field in unprotected_data and unprotected_data[field]:
                        encrypted_text = unprotected_data[field]
                        if self.encryption.is_encrypted(encrypted_text):
                            try:
                                decrypted_text = self.encryption.decrypt_text(encrypted_text)
                                unprotected_data[field] = decrypted_text
                            except Exception as e:
                                self.logger.error(f"Failed to decrypt {field}: {e}")
            
            # Remove protection metadata
            if 'privacy_protection' in unprotected_data:
                del unprotected_data['privacy_protection']
            
            self.logger.info("Capture data unprotected successfully")
            
        except Exception as e:
            self.logger.error(f"Privacy removal failed: {e}")
        
        return unprotected_data
    
    def analyze_data_sensitivity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data for sensitivity and PII.
        
        Args:
            data: Data to analyze
            
        Returns:
            Sensitivity analysis report
        """
        analysis = {
            'sensitivity_score': 0,
            'pii_detected': {},
            'sensitive_fields': [],
            'recommendations': []
        }
        
        # Fields to analyze
        text_fields = ['clipboard_text', 'notes', 'source_window']
        
        for field in text_fields:
            if field in data and data[field]:
                # Check for PII
                if self.redactor:
                    pii_stats = self.redactor.get_detection_stats(data[field])
                    analysis['pii_detected'][field] = pii_stats
                    
                    if pii_stats['redaction_needed']:
                        analysis['sensitive_fields'].append(field)
                        analysis['sensitivity_score'] += pii_stats['total_detected']
                
                # Check for encrypted data
                if self.encryption and self.encryption.is_encrypted(str(data[field])):
                    analysis['recommendations'].append(f"{field} is already encrypted")
                elif field in analysis['sensitive_fields']:
                    analysis['recommendations'].append(f"Consider encrypting {field}")
        
        return analysis
    
    def _should_encrypt_field(self, field_name: str) -> bool:
        """Check if field should be encrypted."""
        # Encrypt sensitive text fields
        return field_name in ['clipboard_text', 'notes'] and self.encryption
    
    def _should_redact_field(self, field_name: str) -> bool:
        """Check if field should be redacted."""
        # Redact PII from all text fields
        return field_name in ['clipboard_text', 'notes', 'source_window'] and self.redactor
    
    def batch_protect(self, captures_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply protection to multiple captures.
        
        Args:
            captures_data: List of capture data
            
        Returns:
            Tuple of (protected_data, batch_report)
        """
        protected_data = []
        batch_report = {
            'total_processed': len(captures_data),
            'successful': 0,
            'failed': 0,
            'encryption_applied': 0,
            'redaction_applied': 0,
            'errors': []
        }
        
        for i, capture_data in enumerate(captures_data):
            try:
                protected = self.protect_capture(capture_data)
                protected_data.append(protected)
                batch_report['successful'] += 1
                
                # Count protection types applied
                if 'privacy_protection' in protected:
                    protection = protected['privacy_protection']
                    if protection.get('encryption_applied'):
                        batch_report['encryption_applied'] += 1
                    if protection.get('redaction_applied'):
                        batch_report['redaction_applied'] += 1
                
            except Exception as e:
                batch_report['failed'] += 1
                batch_report['errors'].append(f"Failed to protect capture {i}: {e}")
                protected_data.append(capture_data)  # Keep original if protection fails
        
        return protected_data, batch_report
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get current privacy configuration."""
        return {
            'encryption_enabled': self.encryption is not None,
            'redaction_enabled': self.redactor is not None,
            'auto_protect': self.auto_protect,
            'encryption_info': self.encryption.get_encryption_info() if self.encryption else {},
            'redaction_config': self.redactor.config if self.redactor else {}
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update privacy configuration."""
        self.config.update(config_updates)
        self.logger.info("Privacy configuration updated")


def create_privacy_mode(config: Optional[Dict[str, Any]] = None) -> PrivacyMode:
    """
    Factory function to create privacy mode instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PrivacyMode instance
    """
    return PrivacyMode(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create privacy mode
    config = {
        'enable_encryption': True,
        'enable_redaction': True,
        'auto_protect': True,
        'redaction_config': {
            'pattern': '[REDACTED]',
            'preserve_length': False
        }
    }
    
    privacy_mode = create_privacy_mode(config)
    
    # Test data with PII
    test_capture = {
        'source_window': 'Email - john.doe@company.com',
        'clipboard_text': 'My phone number is (555) 123-4567 and email is john@email.com',
        'notes': 'Meeting with SSN: 123-45-6789 on 12/25/2023',
        'screenshot_path': '/path/to/screenshot.png'
    }
    
    print("Original data:")
    print(json.dumps(test_capture, indent=2))
    
    # Protect data
    protected = privacy_mode.protect_capture(test_capture)
    print("\nProtected data:")
    print(json.dumps(protected, indent=2))
    
    # Analyze sensitivity
    analysis = privacy_mode.analyze_data_sensitivity(test_capture)
    print("\nSensitivity analysis:")
    print(json.dumps(analysis, indent=2))