#!/usr/bin/env python3
"""Data integrity validator"""
import os
import re
import json

def validate_session_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    required = [
        '## Session Metadata',
        '## AI Response - Visualization Description',
        '## AI Response - Self-Reflection'
    ]
    
    for section in required:
        if section not in content:
            return False, f"Missing: {section}"
    
    return True, "Valid"

print("✅ Validation script ready")
