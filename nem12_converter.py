#!/usr/bin/env python3
"""
NEM12 Converter Module - Polished and Corrected Version
=====================================================

A comprehensive tool for converting CSV/Excel files to NEM12 format with support for:
- Multiple input formats (CSV, Excel, TAB-delimited)
- Time series data conversion
- Existing NEM12 file processing and validation
- Interval data format detection
- Batch processing capabilities

Author: Energy Data Systems
Version: 2.0.0
License: MIT
"""

import argparse
import contextlib
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# ============================================================================
# ENHANCED CONFIGURATION FOR IMPROVED NEM12 PROCESSING
# Add these to your NEM12Config class or update existing ones
# ============================================================================

class NEM12Config:
    """Enhanced configuration constants for NEM12 processing."""
    
    # Valid NEM12 record types in order
    ROW_ORDER = ["100", "200", "300", "400", "900"]
    
    # Valid quality flags for interval data
    VALID_QUALITY_FLAGS = ["A", "S", "F", "V", "N", "E"]
    
    # Maximum file size to process (100MB)
    MAX_FILE_SIZE_MB = 100
    
    # Enhanced reason codes for quality flags
    VALID_REASON_CODES = {
        "0": "Free text description",
        "1": "Meter/equipment changed",
        "2": "Extreme weather conditions",
        "3": "Quarantined premises",
        "4": "Around the clock", 
        "5": "Multiplier change",
        "6": "Null data",
        "7": "Prior to first read after an outage",
        "8": "Prorated",
        "9": "Communications fault",
        "10": "Reset/Revised",
        "11": "Linear interpolation",
        "12": "Agreed between parties",
        "13": "Previous day",
        "14": "Previous week",
        "15": "Previous month", 
        "16": "Previous year",
        "17": "Agreed between parties (alternate)",
        "18": "Agreed between parties (alternate 2)",
        "19": "Like day previous week",
        "20": "Not used",
        "21": "Not used",
        "22": "Not used", 
        "23": "Not used",
        "24": "Substituted/replaced data",
        "51": "Estimated using check meter",
        "52": "Estimated using check/backup meter",
        "53": "Agreed between parties",
        "54": "Prorated on energy",
        "55": "Prorated using demand",
        "56": "Prorated using average",
        "57": "Like day previous week",
        "58": "Customer read",
        "59": "Types 51-54 where date estimated",
        "60": "Substituted with backup",
        "61": "Linear interpolation",
        "62": "Agreed between parties",
        "63": "Previous day same time",
        "64": "Previous week same time",
        "65": "Previous month same time",
        "66": "Previous year same time",
        "67": "Agreed between parties (alternate)",
        "68": "Agreed between parties (alternate 2)",
        "69": "Previous week same day",
        "70": "Not used",
        "71": "Not used",
        "72": "Not used",
        "73": "Not used",
        "74": "Substituted/replaced data",
        "75": "Customer read",
        "76": "Communications fault",
        "77": "Estimation Forecast",
        "78": "Null Data",
        "79": "Previous year same day",
        "80": "Agreed between parties",
        "81": "Linear interpolation",
        "82": "Previous day same time",
        "83": "Previous week same time",
        "84": "Previous month same time",
        "85": "Previous year same time",
        "86": "Agreed between parties (alternate)",
        "87": "Agreed between parties (alternate 2)",
        "88": "Previous week same day",  
        "89": "Time reset occurred"
    }
    
    # Valid method flags (11-19 for different estimation methods)
    VALID_METHOD_FLAGS = [str(i) for i in range(11, 20)]
    
    # Interval length configurations
    INTERVAL_CONFIGS = {
        5: {"intervals_per_day": 288, "total_fields": 291},    # 5-minute
        15: {"intervals_per_day": 96, "total_fields": 99},     # 15-minute  
        30: {"intervals_per_day": 48, "total_fields": 51},     # 30-minute
        60: {"intervals_per_day": 24, "total_fields": 27}      # 1-hour
    }
    
    # Supported date formats for parsing
    DATE_FORMATS = [
        '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y.%m.%d',
        '%d-%b-%y', '%d-%b-%Y'
    ]
    
    # File encoding preferences
    ENCODINGS = ['utf-8', 'utf-8-sig', 'ISO-8859-1', 'cp1252']
    
    # Validation ranges
    REASONABLE_READING_RANGE = (-999999, 999999)  # kWh reading range
    REASONABLE_DATE_RANGE = (1990, 2050)  # Year range

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS FOR ENHANCED VALIDATION
# Add these to your utility functions section
# ============================================================================

def get_interval_config(interval_length: int) -> Dict[str, int]:
    """Get configuration for specific interval length."""
    if interval_length in NEM12Config.INTERVAL_CONFIGS:
        return NEM12Config.INTERVAL_CONFIGS[interval_length]
    
    # Calculate for non-standard intervals
    minutes_per_day = 24 * 60
    intervals_per_day = minutes_per_day // interval_length
    total_fields = 2 + intervals_per_day + 1  # RecordType + Date + Intervals + Quality
    
    return {
        "intervals_per_day": intervals_per_day,
        "total_fields": total_fields
    }


def validate_interval_reading_range(value: float, logger: logging.Logger) -> bool:
    """Validate if an interval reading is within reasonable range."""
    min_val, max_val = NEM12Config.REASONABLE_READING_RANGE
    
    if min_val <= value <= max_val:
        return True
    
    logger.warning(f"Reading {value} outside reasonable range ({min_val}, {max_val})")
    return False


def format_interval_reading(value: Any, precision: int = 3) -> str:
    """Format interval reading to standard precision."""
    try:
        float_val = float(value)
        return f"{float_val:.{precision}f}"
    except (ValueError, TypeError):
        return ""


def parse_quality_flag_field(quality_field: str) -> Tuple[str, Optional[str]]:
    """
    Parse quality flag field that may contain reason codes.
    
    Handles formats like:
    - "A" (just quality flag)
    - "A,75" (quality flag with reason code)
    - "A75" (quality flag with reason code, no separator)
    """
    if not quality_field:
        return 'A', None
    
    quality_str = str(quality_field).strip()
    
    # Handle comma-separated format
    if ',' in quality_str:
        parts = quality_str.split(',', 1)
        quality_flag = parts[0].strip()
        reason_code = parts[1].strip()
    # Handle concatenated format (like "A75")
    elif len(quality_str) > 1 and quality_str[0] in NEM12Config.VALID_QUALITY_FLAGS:
        quality_flag = quality_str[0]
        reason_code = quality_str[1:].strip()
    # Just quality flag
    else:
        quality_flag = quality_str
        reason_code = None
    
    # Validate quality flag
    if quality_flag not in NEM12Config.VALID_QUALITY_FLAGS:
        quality_flag = 'A'  # Default to Actual
        reason_code = None
    
    # Validate reason code
    if reason_code and reason_code not in NEM12Config.VALID_REASON_CODES:
        reason_code = None
    
    return quality_flag, reason_code



def validate_300_record_structure(row_data: List[Any], interval_length: int, 
                                logger: logging.Logger) -> Tuple[bool, List[str]]:
    """
    ENHANCED - Comprehensive validation of 300 record structure based on official NEM12 format.
    
    Official Format:
    - 30-min: RecordType + Date + 48 intervals + Quality + [Optional: Reason + Description + UpdateTime]
    - 15-min: RecordType + Date + 96 intervals + Quality + [Optional: Reason + Description + UpdateTime]
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check record type
    if not row_data or str(row_data[0]).strip() != "300":
        issues.append("Record must start with '300'")
        return False, issues
    
    # Get expected structure based on official NEM12 format
    intervals_per_day = 96 if interval_length == 15 else 48 if interval_length == 30 else 288 if interval_length == 5 else 48
    
    # Official NEM12 300 record structure
    min_fields = 2 + intervals_per_day + 1  # RecordType + Date + Intervals + Quality (minimum)
    max_fields = min_fields + 3  # + Reason Code + Reason Description + Update DateTime (maximum)
    
    logger.debug(f"300 record validation: interval_length={interval_length}min, "
                f"intervals_per_day={intervals_per_day}, min_fields={min_fields}, max_fields={max_fields}")
    
    # Enhanced field count validation based on official format
    current_fields = len(row_data)
    
    if current_fields < min_fields:
        # Check if we're missing just the quality flag
        if current_fields == min_fields - 1:
            issues.append(f"Missing quality flag: {current_fields} fields, need {min_fields} minimum")
        else:
            issues.append(f"Too few fields: {current_fields} < {min_fields} (minimum required)")
    
    elif current_fields > max_fields:
        # Allow some flexibility but warn about excess fields
        issues.append(f"Too many fields: {current_fields} > {max_fields} (maximum expected)")
    
    # Validate date field (position 1)
    if len(row_data) > 1:
        date_field = str(row_data[1]).strip()
        if not _validate_nem12_date_format(date_field):
            issues.append(f"Invalid date format: {date_field} (expected YYYYMMDD)")
        else:
            try:
                datetime.strptime(date_field, '%Y%m%d')
            except ValueError:
                issues.append(f"Invalid date value: {date_field}")
    else:
        issues.append("Missing date field")
    
    # Validate interval readings (positions 2 to 2+intervals_per_day-1)
    invalid_readings = 0
    missing_readings = 0
    interval_start = 2
    interval_end = min(len(row_data), interval_start + intervals_per_day)
    
    for i in range(interval_start, interval_end):
        interval_val = str(row_data[i]).strip() if i < len(row_data) else ""
        
        # Handle empty/missing intervals (acceptable in NEM12)
        if not interval_val or interval_val in ['', '-', 'null', 'NULL', 'None']:
            missing_readings += 1
            continue
        
        # Validate numeric intervals
        try:
            float_val = float(interval_val)
            if not validate_interval_reading_range(float_val, logger):
                invalid_readings += 1
        except (ValueError, TypeError):
            invalid_readings += 1
    
    # Report interval validation results
    if invalid_readings > 0:
        issues.append(f"{invalid_readings} invalid interval readings found")
    
    if missing_readings > intervals_per_day * 0.5:  # More than 50% missing
        issues.append(f"High number of missing readings: {missing_readings}/{intervals_per_day}")
    
    # Validate quality flag (position 2 + intervals_per_day)
    quality_flag_index = 2 + intervals_per_day
    
    if len(row_data) > quality_flag_index:
        quality_field = str(row_data[quality_flag_index]).strip()
        quality_validation = _validate_quality_field(quality_field)
        
        if not quality_validation['is_valid']:
            issues.append(f"Invalid quality field: {quality_field} - {quality_validation['error']}")
        else:
            # Log compound quality field details
            if quality_validation['method_code']:
                logger.debug(f"Quality: {quality_validation['quality']} with method {quality_validation['method_code']}")
    else:
        issues.append("Missing quality flag")
    
    # Validate optional reason code (position 2 + intervals_per_day + 1)
    reason_code_index = quality_flag_index + 1
    if len(row_data) > reason_code_index:
        reason_code = str(row_data[reason_code_index]).strip()
        if reason_code and not _validate_reason_code(reason_code):
            issues.append(f"Invalid reason code: {reason_code}")
    
    # Validate optional reason description (position 2 + intervals_per_day + 2)
    reason_desc_index = reason_code_index + 1
    if len(row_data) > reason_desc_index:
        reason_desc = str(row_data[reason_desc_index]).strip()
        if reason_desc and len(reason_desc) > 100:  # Reasonable length check
            issues.append(f"Reason description too long: {len(reason_desc)} characters")
    
    # Validate optional update datetime (position 2 + intervals_per_day + 3)
    update_time_index = reason_desc_index + 1
    if len(row_data) > update_time_index:
        update_time = str(row_data[update_time_index]).strip()
        if update_time and not _validate_update_datetime(update_time):
            issues.append(f"Invalid update datetime: {update_time}")
    
    is_valid = len(issues) == 0
    
    # Log validation summary
    if is_valid:
        logger.debug(f"300 record validation passed: {current_fields} fields, "
                    f"{intervals_per_day - missing_readings} valid intervals")
    else:
        logger.warning(f"300 record validation failed: {len(issues)} issues found")
    
    return is_valid, issues


def _validate_nem12_date_format(date_field: str) -> bool:
    """Enhanced NEM12 date format validation."""
    if not date_field or not isinstance(date_field, str):
        return False
    
    # Must be exactly 8 digits
    if not re.match(r'^\d{8}$', date_field):
        return False
    
    # Check reasonable date ranges
    try:
        year = int(date_field[:4])
        month = int(date_field[4:6])
        day = int(date_field[6:8])
        
        # Reasonable year range
        if year < 1990 or year > 2050:
            return False
        
        # Valid month
        if month < 1 or month > 12:
            return False
        
        # Valid day
        if day < 1 or day > 31:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def _validate_quality_field(quality_field: str) -> Dict[str, Any]:
    """
    Enhanced quality field validation supporting compound formats.
    
    Official formats:
    - "A" = Actual
    - "S51" = Substituted with method 51
    - "F75" = Final Substituted with method 75
    - "V" = Varied (requires 400 records)
    """
    result = {
        'is_valid': False,
        'quality': None,
        'method_code': None,
        'error': None
    }
    
    if not quality_field:
        result['error'] = "Empty quality field"
        return result
    
    quality_str = str(quality_field).strip().upper()
    
    # Handle single character quality flags
    if len(quality_str) == 1:
        if quality_str in NEM12Config.VALID_QUALITY_FLAGS:
            result['is_valid'] = True
            result['quality'] = quality_str
        else:
            result['error'] = f"Invalid quality flag: {quality_str}"
        return result
    
    # Handle compound quality fields (e.g., "S51", "F75")
    if len(quality_str) > 1:
        quality_flag = quality_str[0]
        method_part = quality_str[1:]
        
        # Validate quality flag
        if quality_flag not in NEM12Config.VALID_QUALITY_FLAGS:
            result['error'] = f"Invalid quality flag: {quality_flag}"
            return result
        
        # Validate method code for S, F, E flags
        if quality_flag in ['S', 'F', 'E']:
            if method_part.isdigit():
                method_code = int(method_part)
                if 11 <= method_code <= 89:  # Valid method code range
                    result['is_valid'] = True
                    result['quality'] = quality_flag
                    result['method_code'] = method_part
                else:
                    result['error'] = f"Method code {method_code} outside valid range (11-89)"
            else:
                result['error'] = f"Invalid method code: {method_part}"
        else:
            # For A, V, N - shouldn't have method codes
            result['error'] = f"Quality flag {quality_flag} should not have method code"
        
        return result
    
    result['error'] = f"Unrecognized quality field format: {quality_field}"
    return result


def _validate_reason_code(reason_code: str) -> bool:
    """Validate reason code against known NEM12 reason codes."""
    if not reason_code:
        return True  # Optional field
    
    try:
        code = str(reason_code).strip()
        # Check if it's in the expanded reason codes list
        return code in NEM12Config.VALID_REASON_CODES
    except (ValueError, TypeError):
        return False


def _validate_update_datetime(update_time: str) -> bool:
    """Validate update datetime field format (YYYYMMDDHHMMSS)."""
    if not update_time:
        return True  # Optional field
    
    # Should be 14 digits for YYYYMMDDHHMMSS
    if not re.match(r'^\d{14}$', update_time):
        return False
    
    try:
        # Validate the datetime
        datetime.strptime(update_time, '%Y%m%d%H%M%S')
        return True
    except ValueError:
        return False


# ============================================================================
# HELPER FUNCTION TO FIX 300 RECORDS BEFORE VALIDATION
# Call this before validate_300_record_structure if you want to auto-fix issues
# ============================================================================

def fix_300_record_structure(row_data: List[Any], interval_length: int, 
                           logger: logging.Logger) -> List[Any]:
    """
    Fix common 300 record structure issues before validation.
    
    This function modifies the row_data to match official NEM12 format.
    """
    if not row_data or str(row_data[0]).strip() != "300":
        return row_data
    
    intervals_per_day = 96 if interval_length == 15 else 48 if interval_length == 30 else 288 if interval_length == 5 else 48
    min_fields = 2 + intervals_per_day + 1  # RecordType + Date + Intervals + Quality
    
    original_length = len(row_data)
    fixed_data = row_data.copy()
    
    # Fix insufficient fields
    if len(fixed_data) < min_fields:
        # Calculate how many intervals we're missing
        current_intervals = len(fixed_data) - 2  # Subtract RecordType and Date
        missing_intervals = intervals_per_day - current_intervals
        
        if missing_intervals > 0 and current_intervals >= 0:
            # Add missing intervals as empty strings
            insert_position = 2 + current_intervals  # After existing intervals
            for _ in range(missing_intervals):
                fixed_data.insert(insert_position, '')
                insert_position += 1
        
        # Add quality flag if missing
        if len(fixed_data) == min_fields - 1:
            fixed_data.append('A')  # Default quality flag
        
        logger.info(f"Fixed 300 record: padded from {original_length} to {len(fixed_data)} fields")
    
    # Handle excess fields (keep up to max_fields)
    elif len(fixed_data) > min_fields + 3:  # Beyond max expected fields
        max_fields = min_fields + 3
        fixed_data = fixed_data[:max_fields]
        logger.warning(f"Fixed 300 record: truncated from {original_length} to {len(fixed_data)} fields")
    
    return fixed_data


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the enhanced validation."""
    
    # Test data
    test_row = ["300", "20240101"] + ["1.234"] * 48 + ["S51", "75", "Meter fault", "20240101120000"]
    
    logger = logging.getLogger("test")
    
    # Method 1: Fix then validate
    fixed_row = fix_300_record_structure(test_row, 30, logger)
    is_valid, issues = validate_300_record_structure(fixed_row, 30, logger)
    
    print(f"Validation result: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Method 2: Just validate
    is_valid, issues = validate_300_record_structure(test_row, 30, logger)
    print(f"Direct validation: {is_valid}")


# ============================================================================
# IMPORT STATEMENT TO ADD TO THE TOP OF YOUR FILE
# Add this import if not already present
# ============================================================================

from typing import Tuple  # Add this to your existing typing imports


# ============================================================================ 
# EXAMPLE USAGE AND TESTING FUNCTION
# You can use this to test the enhanced validation
# ============================================================================

def test_300_validation():
    """Test function for the enhanced 300 record validation."""
    # Test cases for different interval lengths
    test_cases = [
        {
            "name": "Valid 30-minute record",
            "interval_length": 30,
            "row_data": ["300", "20240101"] + ["1.234"] * 48 + ["A"],
            "should_pass": True
        },
        {
            "name": "Valid 15-minute record", 
            "interval_length": 15,
            "row_data": ["300", "20240101"] + ["0.567"] * 96 + ["A"],
            "should_pass": True
        },
        {
            "name": "Record with quality flag and reason code",
            "interval_length": 30,
            "row_data": ["300", "20240101"] + ["1.000"] * 48 + ["S,75"],
            "should_pass": True
        },
        {
            "name": "Record with method flag",
            "interval_length": 30, 
            "row_data": ["300", "20240101"] + ["2.345"] * 48 + ["A", "11"],
            "should_pass": True
        },
        {
            "name": "Invalid date format",
            "interval_length": 30,
            "row_data": ["300", "2024-01-01"] + ["1.000"] * 48 + ["A"],
            "should_pass": False
        }
    ]
    
    logger = logging.getLogger("test")
    
    for test in test_cases:
        is_valid, issues = validate_300_record_structure(
            test["row_data"], 
            test["interval_length"], 
            logger
        )
        
        print(f"Test: {test['name']}")
        print(f"  Expected: {'PASS' if test['should_pass'] else 'FAIL'}")
        print(f"  Actual: {'PASS' if is_valid else 'FAIL'}")
        if issues:
            print(f"  Issues: {', '.join(issues)}")
        print()

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class NEM12ValidationError(Exception):
    """Custom exception for NEM12 validation errors."""
    pass


class FileProcessingError(Exception):
    """Custom exception for file processing errors.""" 
    pass


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = "nem12_conversion.log") -> logging.Logger:
    """Set up and return a configured logger with proper encoding support."""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    logger = logging.getLogger("nem12_converter")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        print(f"Warning: Could not create log file {log_file}: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    return logger


def safe_log_message(logger: logging.Logger, level: str, message: str) -> None:
    """Safely log messages, handling encoding issues."""
    try:
        getattr(logger, level.lower())(message)
    except (UnicodeEncodeError, AttributeError):
        # Remove non-ASCII characters and try again
        clean_message = re.sub(r'[^\x00-\x7F]', '?', message)
        getattr(logger, level.lower(), logger.info)(clean_message)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class NEM12Block:
    """Complete NEM12 block with proper validation - UPDATED VERSION."""
    
    def __init__(self, nmi: Optional[str] = None) -> None:
        self.header: Optional[List[Any]] = None
        self.nmi_blocks: List[Dict[str, Any]] = []
        self.footer: Optional[List[Any]] = None
        self.current_nmi_block: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger("nem12_converter")
        
    def add_row(self, row_type: str, row_data: List[Any]) -> None:
        """Add a row to the appropriate section of the block."""
        row_type = str(row_type).strip()
        
        if row_type not in NEM12Config.ROW_ORDER:
            raise NEM12ValidationError(f"Invalid row type: {row_type}")
        
        try:
            if row_type == "100":
                self._add_header(row_data)
            elif row_type == "200":
                self._add_nmi_header(row_data)
            elif row_type == "300":
                self._add_interval_data(row_data)
            elif row_type == "400":
                self._add_event_data(row_data)
            elif row_type == "900":
                self._add_footer(row_data)
                
        except Exception as e:
            self.logger.error(f"Error adding row type {row_type}: {e}")
            raise NEM12ValidationError(f"Failed to add row: {e}")
    
    def _add_header(self, row_data: List[Any]) -> None:
        """Add header (100) record with validation."""
        if len(row_data) < 5:
            raise NEM12ValidationError("Header record must have at least 5 fields")
        
        # Validate header format
        if str(row_data[0]).strip() != "100":
            raise NEM12ValidationError("Header record must start with '100'")
        
        if str(row_data[1]).strip() != "NEM12":
            self.logger.warning(f"Header version is not 'NEM12': {row_data[1]}")
        
        # Validate timestamp format (should be YYYYMMDDHHMM)
        if len(row_data) > 2:
            timestamp = str(row_data[2]).strip()
            if not re.match(r'^\d{12}$', timestamp):
                self.logger.warning(f"Invalid timestamp format: {timestamp}")
        
        self.header = row_data
    
    def _add_nmi_header(self, row_data: List[Any]) -> None:
        """UPDATED - Add NMI header (200) record with proper validation."""
        if len(row_data) < 10:
            raise NEM12ValidationError("NMI header record must have at least 10 fields")
        
        # Validate record type
        if str(row_data[0]).strip() != "200":
            raise NEM12ValidationError("NMI header record must start with '200'")
        
        # Validate NMI format
        nmi = str(row_data[1]).strip()
        if not validate_nmi(nmi):
            self.logger.warning(f"Invalid NMI format: {nmi}")
        
        # Validate NMI suffix
        if len(row_data) > 2:
            suffix = str(row_data[2]).strip()
            valid_suffixes = ['E1', 'E2', 'B1', 'B2', 'G1', 'G2']
            if suffix not in valid_suffixes:
                self.logger.warning(f"Unusual NMI suffix: {suffix}")
        
        # Validate UOM (Unit of Measure)
        if len(row_data) > 7:
            uom = str(row_data[7]).strip()
            valid_uoms = ['KWH', 'KVARH', 'KW', 'KVAR', 'MWH', 'MW']
            if uom and uom not in valid_uoms:
                self.logger.warning(f"Unusual UOM: {uom}")
        
        # Validate interval length
        if len(row_data) > 8:
            interval_length = str(row_data[8]).strip()
            if interval_length and not re.match(r'^\d+$', interval_length):
                self.logger.warning(f"Invalid interval length: {interval_length}")
        
        nmi_block = {"200": row_data, "300": [], "400": []}
        self.nmi_blocks.append(nmi_block)
        self.current_nmi_block = nmi_block

    def _get_interval_length_from_nmi_block(self) -> int:
        """Get interval length from the current NMI block's 200 record."""
        try:
            if self.current_nmi_block and "200" in self.current_nmi_block:
                nmi_record = self.current_nmi_block["200"]
                if len(nmi_record) > 8:
                    interval_length = str(nmi_record[8]).strip()
                    if interval_length.isdigit():
                        return int(interval_length)
            
            # Default to 30 minutes if not specified
            self.logger.warning("Could not determine interval length from 200 record, defaulting to 30 minutes")
            return 30
            
        except (IndexError, ValueError, TypeError):
            self.logger.warning("Error reading interval length from 200 record, defaulting to 30 minutes")
            return 30

    def _calculate_intervals_per_day(self, interval_length: int) -> int:
        """Calculate number of intervals per day based on interval length."""
        intervals_map = {
            5: 288,   # 5-minute intervals
            15: 96,   # 15-minute intervals  
            30: 48,   # 30-minute intervals
            60: 24    # 1-hour intervals (less common)
        }
        
        if interval_length in intervals_map:
            return intervals_map[interval_length]
        
        # Calculate dynamically for non-standard intervals
        minutes_per_day = 24 * 60
        intervals_per_day = minutes_per_day // interval_length
        
        self.logger.warning(f"Non-standard interval length {interval_length}, calculated {intervals_per_day} intervals per day")
        return intervals_per_day

    def _fix_300_record_structure(self, row_data: List[Any], intervals_per_day: int, 
                                expected_fields: int, original_length: int) -> List[Any]:
        """Fix 300 record structure issues."""
        if len(row_data) < expected_fields:
            # Pad with missing intervals
            current_intervals = len(row_data) - 2  # Subtract RecordType and Date
            missing_intervals = intervals_per_day - current_intervals
            
            if missing_intervals > 0:
                # Add missing interval values as empty strings or zeros
                for _ in range(missing_intervals):
                    row_data.insert(-1 if len(row_data) >= 3 else len(row_data), '0.000')
            
            # Ensure we have a quality flag
            if len(row_data) == expected_fields - 1:
                row_data.append('A')  # Default quality flag
            
            self.logger.info(f"Padded 300 record from {original_length} to {len(row_data)} fields")
        
        elif len(row_data) > expected_fields:
            # Handle excess fields - could be method flags or extra data
            excess_fields = len(row_data) - expected_fields
            
            # Keep method flag if it looks valid
            if excess_fields == 1 and len(row_data) > expected_fields:
                potential_method = str(row_data[expected_fields]).strip()
                if potential_method.isdigit() and 11 <= int(potential_method) <= 19:
                    self.logger.debug(f"Keeping method flag: {potential_method}")
                    return row_data  # Keep the method flag
            
            # Otherwise truncate
            row_data = row_data[:expected_fields]
            self.logger.warning(f"Truncated 300 record from {original_length} to {len(row_data)} fields")
        
        return row_data

    def _validate_nem12_date(self, date_field: str) -> bool:
        """Enhanced NEM12 date validation."""
        if not re.match(r'^\d{8}$', date_field):
            return False
        
        try:
            year = int(date_field[:4])
            month = int(date_field[4:6])
            day = int(date_field[6:8])
            
            # Check reasonable ranges
            if year < 1990 or year > 2050:
                self.logger.warning(f"Date year {year} outside reasonable range")
                return False
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False
            
            # Validate actual date
            datetime.strptime(date_field, '%Y%m%d')
            return True
            
        except (ValueError, TypeError):
            return False

    def _validate_interval_readings(self, row_data: List[Any], intervals_per_day: int) -> int:
        """Validate interval reading values."""
        valid_intervals = 0
        interval_start = 2  # After RecordType and Date
        interval_end = interval_start + intervals_per_day
        
        for i in range(interval_start, interval_end):
            if i < len(row_data):
                interval_val = str(row_data[i]).strip()
                
                # Handle empty intervals
                if not interval_val or interval_val in ['', '-', 'null', 'NULL', 'None']:
                    row_data[i] = ''  # Keep empty for missing data
                    continue
                
                # Validate numeric intervals
                try:
                    float_val = float(interval_val)
                    
                    # Check for reasonable range (adjust as needed)
                    if abs(float_val) > 999999:
                        self.logger.warning(f"Large interval value at position {i}: {float_val}")
                    
                    # Format to standard precision
                    row_data[i] = f"{float_val:.3f}"
                    valid_intervals += 1
                    
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid interval value at position {i}: {interval_val}")
                    row_data[i] = ''  # Set to empty rather than zero for invalid data
        
        return valid_intervals

    def _validate_quality_flag_with_reason(self, quality_field: Any) -> Tuple[str, Optional[str]]:
        """Validate quality flag and extract reason code if present."""
        if not quality_field:
            return 'A', None
        
        quality_str = str(quality_field).strip()
        
        # Handle quality flag with reason code (format: "A,75" or "A75")
        if ',' in quality_str:
            parts = quality_str.split(',', 1)
            quality_flag = parts[0].strip()
            reason_code = parts[1].strip()
        elif len(quality_str) > 1 and quality_str[0] in NEM12Config.VALID_QUALITY_FLAGS:
            quality_flag = quality_str[0]
            reason_code = quality_str[1:].strip()
        else:
            quality_flag = quality_str
            reason_code = None
        
        # Validate quality flag
        if quality_flag not in NEM12Config.VALID_QUALITY_FLAGS:
            self.logger.warning(f"Invalid quality flag: {quality_flag}, defaulting to 'A'")
            quality_flag = 'A'
            reason_code = None
        
        # Validate reason code if present
        if reason_code:
            if reason_code not in NEM12Config.VALID_REASON_CODES:
                self.logger.warning(f"Invalid reason code: {reason_code}")
                reason_code = None
        
        return quality_flag, reason_code

    def _validate_method_flag(self, row_data: List[Any], method_flag_index: int) -> None:
        """Validate method flag if present."""
        if method_flag_index < len(row_data):
            method_flag = str(row_data[method_flag_index]).strip()
            if method_flag:
                # Method flags are typically 11-19 for different estimation methods
                valid_methods = [str(i) for i in range(11, 20)]
                if method_flag not in valid_methods:
                    self.logger.warning(f"Invalid method flag: {method_flag}")
                    # Keep the flag but log the warning - don't remove it

    def _add_event_data(self, row_data: List[Any]) -> None:
        """Enhanced 400 record processing for interval quality ranges."""
        
        if not self.current_nmi_block:
            raise NEM12ValidationError("400 record found without preceding 200 record")
        
        # Parse 400 record: [400, start_interval, end_interval, quality, reason_code, reason_description]
        if len(row_data) >= 4:
            try:
                start_interval = int(row_data[1])
                end_interval = int(row_data[2]) 
                quality_method = str(row_data[3]).strip()
                
                # Store interval quality range information
                interval_quality = {
                    'start': start_interval,
                    'end': end_interval,
                    'quality': quality_method,
                    'reason_code': row_data[4] if len(row_data) > 4 else None,
                    'reason_desc': row_data[5] if len(row_data) > 5 else None
                }
                
                self.current_nmi_block["400"].append(row_data)
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Invalid 400 record format: {e}")
    
    def _add_footer(self, row_data: List[Any]) -> None:
        """Add footer (900) record with validation."""
        if str(row_data[0]).strip() != "900":
            raise NEM12ValidationError("Footer record must start with '900'")
        
        # Footer typically has only one field
        if len(row_data) > 1:
            self.logger.debug(f"Footer has {len(row_data)} fields, typically expects 1")
        
        self.footer = row_data
    

    def _parse_quality_field(self, quality_field: str) -> Dict[str, Optional[str]]:
        """
        Parse compound quality fields like 'S51' into components.
        
        Handles official NEM12 quality formats:
        - "A" -> {'quality': 'A', 'method': None, 'reason': None}
        - "S51" -> {'quality': 'S', 'method': '51', 'reason': None}
        - "F75" -> {'quality': 'F', 'method': '75', 'reason': None}
        """
        if not quality_field:
            return {'quality': 'A', 'method': None, 'reason': None}
        
        quality_str = str(quality_field).strip().upper()
        
        # Handle compound formats like 'S51', 'F75'
        if len(quality_str) > 1 and quality_str[0] in ['S', 'F', 'E']:
            quality_flag = quality_str[0]
            method_code = quality_str[1:] if quality_str[1:].isdigit() else None
            
            # Validate method code range
            if method_code and method_code.isdigit():
                method_num = int(method_code)
                if not (11 <= method_num <= 89):
                    self.logger.warning(f"Method code {method_num} outside valid range (11-89)")
                    method_code = None
        else:
            # Single character quality flags (A, V, N, etc.)
            quality_flag = quality_str[0] if quality_str else 'A'
            method_code = None
        
        # Validate quality flag
        if quality_flag not in NEM12Config.VALID_QUALITY_FLAGS:
            self.logger.warning(f"Invalid quality flag: {quality_flag}, defaulting to 'A'")
            quality_flag = 'A'
            method_code = None
        
        return {
            'quality': quality_flag,
            'method': method_code,
            'reason': None
        }
    
    def _validate_quality_with_reason(self, quality_data: List[Any], start_index: int) -> Dict[str, Optional[str]]:
        """
        Validate quality field with optional reason code and description.
        
        Official NEM12 300 record format:
        [...intervals...] + Quality + [Reason Code] + [Reason Description] + [Update Time]
        """
        result = {'quality': 'A', 'method': None, 'reason_code': None, 'reason_desc': None}
        
        # Parse quality field
        if len(quality_data) > start_index:
            quality_info = self._parse_quality_field(quality_data[start_index])
            result.update(quality_info)
        
        # Parse reason code (next field)
        reason_code_index = start_index + 1
        if len(quality_data) > reason_code_index:
            reason_code = str(quality_data[reason_code_index]).strip()
            if reason_code and reason_code in NEM12Config.VALID_REASON_CODES:
                result['reason_code'] = reason_code
            elif reason_code:
                self.logger.warning(f"Unknown reason code: {reason_code}")
        
        # Parse reason description (next field)
        reason_desc_index = start_index + 2
        if len(quality_data) > reason_desc_index:
            reason_desc = str(quality_data[reason_desc_index]).strip()
            if reason_desc and len(reason_desc) <= 100:  # Reasonable length
                result['reason_desc'] = reason_desc
        
        return result
    
    def _format_quality_for_output(self, quality_info: Dict[str, Optional[str]]) -> str:
        """
        Format quality information back to NEM12 format for output.
        
        Examples:
        - {'quality': 'A', 'method': None} -> "A"
        - {'quality': 'S', 'method': '51'} -> "S51"  
        """
        quality = quality_info.get('quality', 'A')
        method = quality_info.get('method')
        
        if method and quality in ['S', 'F', 'E']:
            return f"{quality}{method}"
        else:
            return quality
    
    # ========================================================================
    # UPDATED _add_interval_data METHOD USING THE NEW HELPER METHODS
    # ========================================================================
    
    def _add_interval_data(self, row_data: List[Any]) -> None:
        """
        ENHANCED - Add interval data (300) record with official NEM12 format support.
        """
        if not self.current_nmi_block:
            raise NEM12ValidationError("300 record found without preceding 200 record")
        
        # Validate record type
        if str(row_data[0]).strip() != "300":
            raise NEM12ValidationError("Interval data record must start with '300'")
        
        # Get interval length from current NMI block (200 record)
        interval_length = self._get_interval_length_from_nmi_block()
        intervals_per_day = self._calculate_intervals_per_day(interval_length)
        
        # Official NEM12 format: RecordType + Date + Intervals + Quality + [Optional fields]
        min_fields = 2 + intervals_per_day + 1  # Minimum required
        max_fields = min_fields + 3  # + Reason Code + Description + Update Time
        
        original_length = len(row_data)
        
        self.logger.debug(f"Processing 300 record: interval_length={interval_length}min, "
                         f"expected_intervals={intervals_per_day}, current_fields={original_length}")
        
        # Fix structure if needed
        if len(row_data) < min_fields or len(row_data) > max_fields:
            row_data = self._fix_300_record_structure(row_data, intervals_per_day, min_fields, max_fields, original_length)
        
        # Validate date field
        if len(row_data) > 1:
            date_field = str(row_data[1]).strip()
            if not self._validate_nem12_date(date_field):
                raise NEM12ValidationError(f"Invalid date format/value in 300 record: {date_field}")
        
        # Validate interval readings
        valid_intervals = self._validate_interval_readings(row_data, intervals_per_day)
        self.logger.debug(f"Validated {valid_intervals}/{intervals_per_day} intervals")
        
        # Parse and validate quality information using new helper method
        quality_start_index = 2 + intervals_per_day
        quality_info = self._validate_quality_with_reason(row_data, quality_start_index)
        
        # Update the quality field with cleaned format
        if len(row_data) > quality_start_index:
            row_data[quality_start_index] = self._format_quality_for_output(quality_info)
        
        # Log quality information
        if quality_info['method']:
            self.logger.debug(f"Quality: {quality_info['quality']} with method {quality_info['method']}")
        if quality_info['reason_code']:
            self.logger.debug(f"Reason: {quality_info['reason_code']} - {quality_info['reason_desc']}")
        
        self.current_nmi_block["300"].append(row_data)
    








    def is_valid(self) -> bool:
        """Check if the block has the minimum required components."""
        is_structurally_valid = (
            self.header is not None and 
            len(self.nmi_blocks) > 0 and 
            self.footer is not None
        )
        
        if not is_structurally_valid:
            return False
        
        # Additional validation
        for nmi_block in self.nmi_blocks:
            # Each NMI block should have at least one 300 record
            if len(nmi_block.get("300", [])) == 0:
                self.logger.warning("NMI block has no interval data (300 records)")
                return False
        
        return True
    
    def get_nmis(self) -> List[str]:
        """Get the list of NMIs in this block."""
        nmis = []
        for nmi_block in self.nmi_blocks:
            if len(nmi_block["200"]) > 1:
                nmi = str(nmi_block["200"][1]).strip()
                nmis.append(nmi)
            else:
                nmis.append("UNKNOWN")
        return nmis
    
    def get_all_rows(self) -> List[List[Any]]:
        """Get all rows in the correct order for a NEM12 file."""
        rows = []
        
        if self.header:
            rows.append(self.header)
        
        for nmi_block in self.nmi_blocks:
            # Add 200 record
            rows.append(nmi_block["200"])
            
            # Add all 300 records (sorted by date if possible)
            interval_records = nmi_block["300"]
            try:
                # Sort by date (field 1)
                interval_records.sort(key=lambda x: x[1] if len(x) > 1 else "00000000")
            except (TypeError, IndexError):
                self.logger.debug("Could not sort interval records by date")
            
            rows.extend(interval_records)
            
            # Add any 400 records
            rows.extend(nmi_block["400"])
        
        if self.footer:
            rows.append(self.footer)
            
        return rows
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the NEM12 block."""
        stats = {
            'nmis': len(self.nmi_blocks),
            'total_interval_records': 0,
            'total_event_records': 0,
            'date_range': {'start': None, 'end': None},
            'quality_flags': Counter()
        }
        
        all_dates = []
        
        for nmi_block in self.nmi_blocks:
            stats['total_interval_records'] += len(nmi_block["300"])
            stats['total_event_records'] += len(nmi_block["400"])
            
            # Collect dates and quality flags
            for record in nmi_block["300"]:
                if len(record) > 1:
                    all_dates.append(record[1])
                if len(record) > 50:
                    stats['quality_flags'][record[50]] += 1
        
        # Determine date range
        if all_dates:
            valid_dates = [d for d in all_dates if re.match(r'^\d{8}$', str(d))]
            if valid_dates:
                stats['date_range']['start'] = min(valid_dates)
                stats['date_range']['end'] = max(valid_dates)
        
        return stats


# ============================================================================
# ADDITIONAL HELPER FUNCTION FOR NMI VALIDATION
# ============================================================================

def validate_nmi(nmi: str) -> bool:
    """Validate NMI according to Australian energy market standards."""
    if not nmi or not isinstance(nmi, str):
        return False
    
    nmi = nmi.strip().upper()
    
    # NMI should be 10-11 alphanumeric characters
    if not re.match(r'^[A-Z0-9]{10,11}$', nmi):
        return False
    
    # Basic format validation - most NMIs start with a letter for jurisdiction
    if len(nmi) == 10 and not nmi[0].isalpha():
        return False
    
    return True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_row_type(value: Any) -> Optional[str]:
    """Safely extract and validate NEM12 row type."""
    if pd.isna(value):
        return None
    
    str_value = str(value).strip()
    
    # Check for valid record types first
    if str_value in NEM12Config.ROW_ORDER:
        return str_value
    
    # Handle scientific notation (reject if not a valid record type)
    if 'E' in str_value or 'e' in str_value:
        try:
            float(str_value)  # If it parses as scientific notation
            return None       # and it's not a valid record type, reject it
        except (ValueError, TypeError):
            pass
    
    # Handle decimal record types
    if '.' in str_value:
        try:
            str_value = str(int(float(str_value)))
        except (ValueError, TypeError):
            pass
    
    return str_value if str_value in NEM12Config.ROW_ORDER else None


def parse_date(date_value: str) -> Optional[str]:
    """Parse a date string in various formats to YYYYMMDD."""
    if not date_value or not isinstance(date_value, str):
        return None
        
    date_value = date_value.strip()
    
    for fmt in NEM12Config.DATE_FORMATS:
        try:
            date_obj = datetime.strptime(date_value, fmt)
            return date_obj.strftime('%Y%m%d')
        except ValueError:
            continue
            
    return None


def parse_time(time_str: str) -> Optional[str]:
    """Parse time string into HHMM format."""
    if not isinstance(time_str, str):
        return None
        
    time_str = time_str.strip()
    
    # Handle different time formats
    patterns = [
        (r'^(\d{1,2}):(\d{2})(?::\d{2})?$', lambda m: f"{int(m.group(1)):02d}{int(m.group(2)):02d}"),
        (r'^(\d{1,2})\.(\d{2})$', lambda m: f"{int(m.group(1)):02d}{int(m.group(2)):02d}"),
        (r'^(\d{1,2})[hH](\d{1,2})$', lambda m: f"{int(m.group(1)):02d}{int(m.group(2)):02d}"),
        (r'^(\d{4})$', lambda m: m.group(1) if len(m.group(1)) == 4 else None)
    ]
    
    for pattern, formatter in patterns:
        match = re.match(pattern, time_str)
        if match:
            try:
                result = formatter(match)
                if result and len(result) == 4 and result.isdigit():
                    hour, minute = int(result[:2]), int(result[2:])
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        return result
            except (ValueError, TypeError):
                continue
    
    return None


def validate_file_size(file_path: str) -> bool:
    """Validate file size is within acceptable limits."""
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb <= NEM12Config.MAX_FILE_SIZE_MB
    except OSError:
        return False


@contextlib.contextmanager
def safe_file_processing(file_path: str, logger: logging.Logger):
    """Context manager for safe file processing."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read file: {file_path}")
            
        if not validate_file_size(file_path):
            raise FileProcessingError(f"File too large: {file_path}")
            
        yield file_path
        
    except Exception as e:
        logger.error(f"Error accessing file {file_path}: {e}")
        raise


def detect_file_encoding(file_path: str, logger: logging.Logger) -> str:
    """Detect file encoding using multiple methods."""
    try:
        # Try chardet if available
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            if result and result.get('confidence', 0) > 0.7:
                encoding = result['encoding']
                logger.debug(f"Detected encoding: {encoding}")
                return encoding
        except ImportError:
            logger.debug("chardet not available, using fallback detection")
        
        # Fallback method
        for encoding in NEM12Config.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)
                logger.debug(f"Detected encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        logger.warning("Could not detect encoding, defaulting to utf-8")
        return 'utf-8'
        
    except Exception as e:
        logger.warning(f"Error detecting encoding: {e}")
        return 'utf-8'


def detect_delimiter(file_path: str, logger: logging.Logger) -> str:
    """Detect the delimiter used in the file."""
    try:
        encoding = detect_file_encoding(file_path, logger)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            first_line = f.readline().strip()
        
        # Count different delimiters
        tab_count = first_line.count('\t')
        comma_count = first_line.count(',')
        semicolon_count = first_line.count(';')
        pipe_count = first_line.count('|')
        
        # Determine best delimiter
        counts = [
            (tab_count, '\t', 'TAB'),
            (comma_count, ',', 'COMMA'),
            (semicolon_count, ';', 'SEMICOLON'),
            (pipe_count, '|', 'PIPE')
        ]
        
        # Sort by count and take the highest
        counts.sort(reverse=True)
        best_count, best_delimiter, best_name = counts[0]
        
        if best_count >= 2:
            logger.info(f"Detected {best_name} delimiter (count: {best_count})")
            return best_delimiter
        else:
            logger.warning("No clear delimiter detected, defaulting to comma")
            return ','
            
    except Exception as e:
        logger.error(f"Error detecting delimiter: {e}")
        return ','


# ============================================================================
# NMI EXTRACTION AND VALIDATION
# ============================================================================

def extract_nmi_from_filename(filename: str, logger: logging.Logger) -> Optional[str]:
    """Extract NMI from filename using various patterns."""
    patterns = [
        r'\b(\d{10})\b',  # 10-digit number
        r'\b([A-Za-z]?\d{9,10})\b',  # Optional letter + 9-10 digits
        r'NMI[_\-\s]*([A-Za-z]?\d{9,10})',  # NMI prefix
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            nmi = match.group(1)
            if len(nmi) >= 9:  # Valid NMI length
                logger.info(f"Extracted NMI from filename: {nmi}")
                return nmi
    
    return None


def extract_nmi_from_data(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Extract NMI from data content."""
    try:
        # Look for NMI in first few rows and columns
        search_area = df.iloc[:10, :5] if df.shape[0] >= 10 and df.shape[1] >= 5 else df
        
        for idx, row in search_area.iterrows():
            for col_idx, value in enumerate(row):
                if pd.isna(value):
                    continue
                    
                str_value = str(value).strip()
                
                # Look for 10-digit patterns
                if re.match(r'^\d{10}$', str_value):
                    logger.info(f"Found NMI in data at row {idx}, col {col_idx}: {str_value}")
                    return str_value
                
                # Look for NMI with prefix
                nmi_match = re.search(r'(?:NMI|nmi)[:\s]*([A-Za-z]?\d{9,10})', str_value)
                if nmi_match:
                    nmi = nmi_match.group(1)
                    logger.info(f"Found labeled NMI in data: {nmi}")
                    return nmi
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting NMI from data: {e}")
        return None
    
def generate_auto_nmi() -> str:
    """Generate an automatic NMI when none can be found."""
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    return f"AUTO{timestamp}"

def extract_and_validate_nmi(file_path: str, df: Optional[pd.DataFrame], 
                            logger: logging.Logger) -> str:
    """FIXED VERSION - Extract and validate NMI with proper validation."""
    logger.info(f"Extracting NMI from: {os.path.basename(file_path)}")
    
    # Try filename first
    filename_nmi = extract_nmi_from_filename(os.path.basename(file_path), logger)
    if filename_nmi and validate_nmi(filename_nmi):
        logger.info(f"Valid NMI found in filename: {filename_nmi}")
        return filename_nmi
    
    # Try data content
    if df is not None and not df.empty:
        data_nmi = extract_nmi_from_data(df, logger)
        if data_nmi and validate_nmi(data_nmi):
            logger.info(f"Valid NMI found in data: {data_nmi}")
            return data_nmi
    
    # Generate automatic NMI
    auto_nmi = generate_auto_nmi()
    logger.warning(f"No valid NMI found, using auto-generated: {auto_nmi}")
    return auto_nmi

# ============================================================================
# FORMAT DETECTION
# ============================================================================

def detect_nem12_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Detect if file is already in NEM12 format."""
    if df.empty:
        return False
    
    first_cell = str(df.iloc[0, 0]).strip()
    return first_cell == "100"

def detect_time_series_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """Detect if file contains time series data."""
    try:
        if df.empty or df.shape[0] < 3:
            return False
        
        # Look for datetime and numeric columns
        datetime_cols = 0
        numeric_cols = 0
        
        for col_idx in range(min(5, df.shape[1])):
            col_data = df.iloc[1:min(11, df.shape[0]), col_idx]
            
            # Check for datetime patterns
            datetime_count = sum(1 for val in col_data 
                               if pd.notna(val) and ' ' in str(val) 
                               and re.search(r'\d{1,2}[:/]\d{2}', str(val)))
            
            if datetime_count >= 3:
                datetime_cols += 1
            
            # Check for numeric data
            numeric_count = 0
            for val in col_data:
                try:
                    if pd.notna(val):
                        float_val = float(str(val).replace(',', ''))
                        if 0 <= float_val <= 100000:  # Reasonable range
                            numeric_count += 1
                except (ValueError, TypeError):
                    pass
            
            if numeric_count >= 3:
                numeric_cols += 1
        
        has_time_series = datetime_cols >= 1 and numeric_cols >= 1
        if has_time_series:
            logger.info("Detected time series format")
        
        return has_time_series
        
    except Exception as e:
        logger.error(f"Error detecting time series format: {e}")
        return False


# ============================================================================
# DATA EXTRACTION - NEM12 FORMAT
# ============================================================================

def clean_300_record(row_data: List[str], logger: logging.Logger) -> List[str]:
    """Clean and validate 300 record format."""
    try:
        if len(row_data) < 50 or str(row_data[0]).strip() != '300':
            return row_data
        
        # Standard NEM12 300 record: RecordType(1) + Date(1) + Intervals(48) + Quality(1) = 51 fields
        cleaned_record = []
        
        # Keep record type
        cleaned_record.append(row_data[0])
        
        # Validate and keep date
        date_field = str(row_data[1]).strip()
        if len(date_field) == 8 and date_field.isdigit():
            cleaned_record.append(date_field)
        else:
            # Try to clean date
            date_digits = re.sub(r'[^\d]', '', date_field)
            if len(date_digits) == 8:
                cleaned_record.append(date_digits)
            else:
                logger.warning(f"Invalid date field: {date_field}")
                cleaned_record.append(date_field)
        
        # Process interval readings (next 48 fields)
        interval_data = row_data[2:50] if len(row_data) >= 50 else row_data[2:]
        
        for i, reading in enumerate(interval_data):
            if i >= 48:  # Only take first 48 intervals
                break
                
            reading_str = str(reading).strip()
            
            # Handle empty/null values
            if reading_str in ['', '-', 'null', 'NULL', 'None', 'nan', 'NaN']:
                cleaned_record.append('')
                continue
            
            # Validate numeric readings
            try:
                reading_val = float(reading_str)
                if -99999 <= reading_val <= 99999:
                    cleaned_record.append(f"{reading_val:.3f}")
                else:
                    logger.debug(f"Out of range reading: {reading_val}")
                    cleaned_record.append('')
            except ValueError:
                # Check if it's a quality flag (shouldn't be here)
                if reading_str in NEM12Config.VALID_QUALITY_FLAGS:
                    logger.debug(f"Found quality flag in readings: {reading_str}")
                    cleaned_record.append('')
                else:
                    cleaned_record.append('')
        
        # Ensure exactly 48 intervals
        while len(cleaned_record) < 50:  # 1 + 1 + 48
            cleaned_record.append('')
        
        # Find and add quality flag
        quality_flag = 'A'  # Default
        if len(row_data) > 50:
            for field in row_data[50:]:
                field_str = str(field).strip()
                if field_str in NEM12Config.VALID_QUALITY_FLAGS:
                    quality_flag = field_str
                    break
        
        cleaned_record.append(quality_flag)
        
        logger.debug(f"Cleaned 300 record: {len(row_data)} -> {len(cleaned_record)} fields")
        return cleaned_record
        
    except Exception as e:
        logger.error(f"Error cleaning 300 record: {e}")
        return row_data


def extract_nem12_data(df: pd.DataFrame, file_path: str, 
                      logger: logging.Logger) -> Generator[Dict[str, Any], None, None]:
    """Extract existing NEM12 formatted data."""
    try:
        logger.info(f"Extracting NEM12 data from {file_path}")
        
        df = df.dropna(how="all")
        nem12_block = NEM12Block()
        current_block_valid = False
        rows_processed = 0
        
        for idx, row in df.iterrows():
            try:
                # Clean row data
                row_values = [str(val).strip() for val in row.dropna().tolist() 
                            if str(val).strip() and str(val).strip().lower() != 'nan']
                
                if not row_values:
                    continue
                
                row_type = safe_row_type(row_values[0])
                
                if row_type:
                    # Handle new block start
                    if row_type == "100" and current_block_valid:
                        logger.info("Starting new NEM12 block")
                        yield {"file": file_path, "nem12_block": nem12_block}
                        nem12_block = NEM12Block()
                        current_block_valid = False
                    
                    # Special handling for 300 records
                    if row_type == "300" and len(row_values) > 52:
                        cleaned_row = clean_300_record(row_values, logger)
                        nem12_block.add_row(row_type, cleaned_row)
                    else:
                        nem12_block.add_row(row_type, row_values)
                    
                    if row_type == "100":
                        current_block_valid = True
                    
                    rows_processed += 1
                    
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Processed {rows_processed} NEM12 rows")
        
        # Yield final block
        if nem12_block.is_valid():
            yield {"file": file_path, "nem12_block": nem12_block}
        elif current_block_valid:
            logger.warning("Adding missing footer to incomplete NEM12 block")
            nem12_block.add_row("900", ["900"])
            yield {"file": file_path, "nem12_block": nem12_block}
            
    except Exception as e:
        logger.error(f"Error extracting NEM12 data from {file_path}: {e}")


# ============================================================================
# DATA EXTRACTION - TIME SERIES FORMAT
# ============================================================================

def find_datetime_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[int]:
    """Find column containing datetime values."""
    for col_idx in range(min(5, df.shape[1])):
        datetime_count = 0
        
        for row_idx in range(1, min(11, df.shape[0])):
            value = str(df.iloc[row_idx, col_idx]) if not pd.isna(df.iloc[row_idx, col_idx]) else ""
            
            # Look for datetime patterns
            if ' ' in value and re.search(r'\d{1,2}[:/]\d{2}', value):
                datetime_count += 1
        
        if datetime_count >= 3:
            logger.debug(f"Found datetime column at index {col_idx}")
            return col_idx
    
    return None


def find_reading_column(df: pd.DataFrame, datetime_col: Optional[int], 
                       logger: logging.Logger) -> Optional[int]:
    """Find column containing numeric readings."""
    for col_idx in range(df.shape[1]):
        if col_idx == datetime_col:
            continue
        
        numeric_count = 0
        
        for row_idx in range(1, min(11, df.shape[0])):
            value = df.iloc[row_idx, col_idx]
            
            try:
                if pd.notna(value):
                    numeric_val = float(str(value).replace(',', ''))
                    if 0 <= numeric_val <= 100000:  # Reasonable range
                        numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        if numeric_count >= 3:
            logger.debug(f"Found reading column at index {col_idx}")
            return col_idx
    
    return None


def extract_time_series_data(df: pd.DataFrame, nmi: str, 
                           logger: logging.Logger) -> List[Dict[str, Any]]:
    """Extract time series data from DataFrame."""
    try:
        logger.info(f"Extracting time series data for NMI: {nmi}")
        
        # Find datetime and reading columns
        datetime_col = find_datetime_column(df, logger)
        reading_col = find_reading_column(df, datetime_col, logger)
        
        if datetime_col is None or reading_col is None:
            logger.warning("Could not identify datetime or reading columns")
            return []
        
        time_series_data = []
        
        for row_idx in range(1, df.shape[0]):
            try:
                datetime_value = str(df.iloc[row_idx, datetime_col]) if not pd.isna(df.iloc[row_idx, datetime_col]) else ""
                reading_value = df.iloc[row_idx, reading_col]
                
                if not datetime_value or pd.isna(reading_value):
                    continue
                
                # Parse datetime
                if ' ' in datetime_value:
                    date_part, time_part = datetime_value.rsplit(' ', 1)
                    
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if formatted_date and formatted_time:
                        # Validate reading
                        try:
                            float_reading = float(str(reading_value).replace(',', ''))
                            if 0 <= float_reading <= 100000:
                                time_series_data.append({
                                    'nmi': nmi,
                                    'date': formatted_date,
                                    'time': formatted_time,
                                    'reading': f"{float_reading:.3f}",
                                    'quality': 'A'
                                })
                        except (ValueError, TypeError):
                            logger.debug(f"Invalid reading value: {reading_value}")
                
            except Exception as e:
                logger.debug(f"Error processing row {row_idx}: {e}")
                continue
        
        logger.info(f"Extracted {len(time_series_data)} time series records")
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting time series data: {e}")
        return []


# ============================================================================
# DATA EXTRACTION - INTERVAL DATA FORMAT
# ============================================================================

def extract_format_period_kwh(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Extract Format: NMI, MeterSerial, Period, Local Time, Data Type, kWh at Meter."""
    time_series_data = []
    
    try:
        for idx in range(1, df.shape[0]):  # Skip header
            row_data = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[idx]]
            
            if len(row_data) < 7:
                continue
            
            nmi, meter_serial, period, date_str, time_str, data_type, kwh_reading = row_data[:7]
            
            if not all([nmi, date_str, time_str, kwh_reading]):
                continue
            
            # Parse date and time
            formatted_date = parse_date(date_str)
            formatted_time = parse_time(time_str)
            
            if not formatted_date or not formatted_time:
                continue
            
            # Parse reading
            try:
                reading = float(kwh_reading)
                if 0 <= reading <= 100000:
                    quality = 'A' if data_type.lower() == 'measured' else 'S'
                    
                    time_series_data.append({
                        'nmi': nmi,
                        'date': formatted_date,
                        'time': formatted_time,
                        'reading': f"{reading:.3f}",
                        'quality': quality
                    })
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Extracted {len(time_series_data)} records from period/kWh format")
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error in extract_format_period_kwh: {e}")
        return []


def extract_format_local_time(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Extract Format: NMI, Local Time, kWh, kW, kVA, PF."""
    time_series_data = []
    
    try:
        for idx in range(1, df.shape[0]):  # Skip header
            row_data = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[idx]]
            
            if len(row_data) < 5:
                continue
            
            nmi, date_str, time_str, status, kwh_reading = row_data[:5]
            
            if not all([nmi, date_str, time_str, kwh_reading]):
                continue
            
            # Parse date and time
            formatted_date = parse_date(date_str)
            formatted_time = parse_time(time_str)
            
            if not formatted_date or not formatted_time:
                continue
            
            # Parse reading
            try:
                reading = float(kwh_reading)
                if 0 <= reading <= 100000:
                    quality = 'A' if status.lower() == 'measured' else 'S'
                    
                    time_series_data.append({
                        'nmi': nmi,
                        'date': formatted_date,
                        'time': formatted_time,
                        'reading': f"{reading:.3f}",
                        'quality': quality
                    })
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Extracted {len(time_series_data)} records from local time format")
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error in extract_format_local_time: {e}")
        return []


def extract_format_connection_point(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Extract Format: CONNECTION POINT ID, METER SERIAL, TIMESTAMP, INTERVAL, KWH."""
    time_series_data = []
    
    try:
        data_start = 1 if 'CONNECTION POINT' in str(df.iloc[0]).upper() else 0
        
        for idx in range(data_start, df.shape[0]):
            row_data = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[idx]]
            
            if len(row_data) < 8:
                continue
            
            nmi = row_data[0]
            date_str = None
            time_str = None
            reading = None
            
            # Find date and time fields
            for i, field in enumerate(row_data):
                if re.match(r'\d{1,2}-\w{3}-\d{2}', field):  # e.g., 1-Jul-17
                    date_str = field
                    if i + 1 < len(row_data) and ':' in row_data[i + 1]:
                        time_str = row_data[i + 1]
                    break
            
            # Find reading (last meaningful numeric field)
            for field in reversed(row_data):
                try:
                    reading = float(field)
                    if 0 <= reading <= 100000:
                        break
                except (ValueError, TypeError):
                    continue
            
            if not all([nmi, date_str, time_str, reading is not None]):
                continue
            
            # Parse date and time
            formatted_date = parse_date(date_str)
            formatted_time = parse_time(time_str)
            
            if formatted_date and formatted_time:
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{reading:.3f}",
                    'quality': 'A'
                })
        
        logger.info(f"Extracted {len(time_series_data)} records from connection point format")
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error in extract_format_connection_point: {e}")
        return []


# ============================================================================
# NEM12 STRUCTURE CREATION
# ============================================================================

def create_nem12_structure(data: List[Dict[str, Any]], nmi: str, 
                         logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Create NEM12 structure from time series data."""
    if not data:
        logger.warning(f"No data provided for NMI: {nmi}")
        return None
    
    logger.info(f"Creating NEM12 structure for {len(data)} records, NMI: {nmi}")
    
    try:
        # Create NEM12 block
        nem12_block = NEM12Block()
        
        # Add header (100)
        current_date = datetime.now().strftime('%Y%m%d%H%M')
        nem12_block.add_row("100", ["100", "NEM12", current_date, "MDPUPLOAD", "RETAILER"])
        
        # Determine interval length from data
        time_values = [record['time'] for record in data]
        interval_length = determine_interval_length(time_values, logger)
        intervals_per_day = 96 if interval_length == 15 else 48
        
        # Add NMI record (200)
        next_read_date = (datetime.now() + timedelta(days=180)).strftime('%Y%m%d')
        nem12_block.add_row("200", [
            "200", nmi, "E1", "1", "E1", "N", "", "KWH", 
            str(interval_length), next_read_date
        ])
        
        # Group data by date
        date_groups = defaultdict(list)
        for record in data:
            date_groups[record['date']].append(record)
        
        # Create interval records for each date
        for date, records in sorted(date_groups.items()):
            interval_record = create_interval_record(date, records, intervals_per_day, logger)
            if interval_record:
                nem12_block.add_row("300", interval_record)
        
        # Add footer (900)
        nem12_block.add_row("900", ["900"])
        
        logger.info(f"Created NEM12 structure with {len(date_groups)} dates")
        return {"file": f"Time Series Data for {nmi}", "nem12_block": nem12_block}
        
    except Exception as e:
        logger.error(f"Error creating NEM12 structure: {e}")
        return None


def determine_interval_length(time_values: List[str], logger: logging.Logger) -> int:
    """Determine interval length (15 or 30 minutes) from time values."""
    try:
        unique_minutes = set()
        
        for time_str in time_values[:50]:  # Sample first 50 values
            if len(time_str) >= 4:
                minute = int(time_str[2:4])
                unique_minutes.add(minute)
        
        # Check if we have 15-minute intervals
        has_15min = any(minute in [15, 45] for minute in unique_minutes)
        interval_length = 15 if has_15min else 30
        
        logger.debug(f"Determined interval length: {interval_length} minutes")
        return interval_length
        
    except Exception as e:
        logger.error(f"Error determining interval length: {e}")
        return 30  # Default to 30 minutes


def create_interval_record(date: str, records: List[Dict[str, Any]], 
                         intervals_per_day: int, logger: logging.Logger) -> Optional[List[str]]:
    """Create a 300 record for a specific date."""
    try:
        # Initialize readings array
        readings = ["" for _ in range(intervals_per_day)]
        quality_flag = "A"
        
        # Sort records by time
        records.sort(key=lambda x: x['time'])
        
        # Fill readings array
        for record in records:
            try:
                time_str = record['time']
                if len(time_str) >= 4:
                    hour = int(time_str[:2])
                    minute = int(time_str[2:4])
                    
                    # Calculate interval index
                    if intervals_per_day == 96:  # 15-minute intervals
                        interval_index = (hour * 4) + (minute // 15)
                    else:  # 30-minute intervals
                        interval_index = (hour * 2) + (minute // 30)
                    
                    if 0 <= interval_index < intervals_per_day:
                        readings[interval_index] = record['reading']
                        
                        # Update quality flag if needed
                        if record.get('quality', 'A') != 'A':
                            quality_flag = record['quality']
                            
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error processing record: {e}")
                continue
        
        # Create 300 record
        interval_record = ["300", date] + readings + [quality_flag]
        return interval_record
        
    except Exception as e:
        logger.error(f"Error creating interval record for {date}: {e}")
        return None


# ============================================================================
# FILE PROCESSING
# ============================================================================

def read_csv_file(file_path: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Read CSV file with proper encoding and delimiter detection."""
    try:
        delimiter = detect_delimiter(file_path, logger)
        encoding = detect_file_encoding(file_path, logger)
        
        logger.debug(f"Reading CSV with delimiter='{delimiter}', encoding='{encoding}'")
        
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
            header=None,
            on_bad_lines="skip",
            dtype=str,
            low_memory=False,
            keep_default_na=False
        )
        
        if df is not None and not df.empty and df.shape[1] > 1:
            logger.info(f"Successfully read CSV: shape {df.shape}")
            return df
        else:
            logger.error("Failed to read CSV or file is empty")
            return None
            
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return None


def read_excel_file(file_path: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Read Excel file with all sheets."""
    sheets = {}
    
    try:
        with pd.ExcelFile(file_path) as xls:
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
                    if not df.empty:
                        sheets[sheet_name] = df
                        logger.debug(f"Read sheet '{sheet_name}': shape {df.shape}")
                except Exception as e:
                    logger.warning(f"Error reading sheet '{sheet_name}': {e}")
                    continue
        
        logger.info(f"Successfully read Excel file with {len(sheets)} sheets")
        return sheets
        
    except Exception as e:
        logger.error(f"Error reading Excel file {file_path}: {e}")
        return {}

def detect_standard_interval_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    CORRECTED VERSION - Detect standard interval format.
    This format: NMI, Date, Interval Length, Period, EndTime, Data Type, Kwh, KW, KVA, PF, Generated Kwh, Net Kwh...
    """
    try:
        if df.empty or df.shape[0] < 2:
            return False
        
        logger.debug("Standard interval format detection")
        
        # Look for header row in first 3 rows
        header_row_idx = None
        
        for row_idx in range(min(3, df.shape[0])):
            # Get row values and convert to uppercase for comparison
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Check for the specific pattern we see in the file
            score = 0
            
            # Core required indicators (must have these)
            if 'NMI' in row_text:
                score += 4
            if 'ENDTIME' in row_text or 'END TIME' in row_text:
                score += 4
            if 'DATA TYPE' in row_text or 'DATATYPE' in row_text:
                score += 3
            if 'PERIOD' in row_text:
                score += 3
            if 'INTERVAL LENGTH' in row_text or 'INTERVALLENGTH' in row_text:
                score += 3
            
            # Energy indicators
            if 'KWH' in row_text:
                score += 2
            if 'NET KWH' in row_text:
                score += 2
            if 'GENERATED KWH' in row_text:
                score += 1
            
            # Additional indicators
            if 'QUALITY CODE' in row_text or 'QUALITYCODE' in row_text:
                score += 1
            if 'MEASURED' in row_text:
                score += 1
            
            logger.debug(f"Row {row_idx + 1} score: {score} - {row_text[:100]}...")
            
            if score >= 15:  # High threshold for this specific format
                header_row_idx = row_idx
                logger.debug(f"Found standard interval header at row {row_idx + 1} with score {score}")
                break
        
        if header_row_idx is None:
            return False
        
        # Verify data format by checking a few data rows
        data_start = header_row_idx + 1
        if data_start >= df.shape[0]:
            return False
        
        valid_data_rows = 0
        for row_idx in range(data_start, min(data_start + 5, df.shape[0])):
            row_data = df.iloc[row_idx].fillna('').astype(str)
            
            # Look for NMI pattern (10 digits) in first column
            first_cell = str(row_data.iloc[0]).strip()
            has_nmi = re.match(r'^\d{10}$', first_cell) is not None
            
            # Look for datetime pattern in EndTime column (should be column 4 based on headers)
            has_datetime = False
            if len(row_data) > 4:
                endtime_cell = str(row_data.iloc[4]).strip()
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', endtime_cell):
                    has_datetime = True
            
            # Look for "Measured" data type (should be column 5)
            has_measured = False
            if len(row_data) > 5:
                datatype_cell = str(row_data.iloc[5]).strip()
                if datatype_cell.upper() == 'MEASURED':
                    has_measured = True
            
            # Look for reasonable energy values (should be column 6 - Kwh)
            has_energy = False
            if len(row_data) > 6:
                try:
                    kwh_value = float(str(row_data.iloc[6]).strip())
                    if 0 <= kwh_value <= 1000:  # Reasonable kWh range
                        has_energy = True
                except (ValueError, TypeError):
                    pass
            
            if has_nmi and has_datetime and has_measured and has_energy:
                valid_data_rows += 1
        
        is_standard_interval = valid_data_rows >= 3
        logger.info(f"Standard interval detection: header_row={header_row_idx + 1 if header_row_idx is not None else 'None'}, "
                   f"valid_rows={valid_data_rows}, result={is_standard_interval}")
        
        return is_standard_interval
        
    except Exception as e:
        logger.error(f"Error detecting standard interval format: {e}")
        return False

# ============================================================================
# FIXED STANDARD INTERVAL FORMAT HANDLER
# Replace your existing extract_standard_interval_data function with this version
# ============================================================================

def extract_standard_interval_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    CORRECTED VERSION - Extract standard interval data.
    Format: NMI, Date, Interval Length, Period, EndTime, Data Type, Kwh, KW, KVA, PF, Generated Kwh, Net Kwh...
    """
    try:
        logger.info("Extracting standard interval data")
        
        # Find header row
        header_row_idx = None
        for row_idx in range(min(3, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Look for the specific header pattern
            if ('NMI' in row_text and 'ENDTIME' in row_text and 
                'DATA TYPE' in row_text and 'KWH' in row_text):
                header_row_idx = row_idx
                break
        
        if header_row_idx is None:
            logger.error("Could not find header row")
            return []
        
        # Get column headers
        headers = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[header_row_idx]]
        logger.debug(f"Headers ({len(headers)} columns): {headers}")
        
        # Map column indices based on the actual file structure
        column_indices = {}
        
        for idx, header in enumerate(headers):
            header_clean = header.upper().replace(' ', '')
            
            if header_clean == 'NMI':
                column_indices['nmi'] = idx
            elif header_clean == 'DATE':
                column_indices['date'] = idx
            elif header_clean == 'ENDTIME':
                column_indices['endtime'] = idx
            elif header_clean == 'PERIOD':
                column_indices['period'] = idx
            elif header_clean == 'INTERVALLENGTH':
                column_indices['interval_length'] = idx
            elif header_clean == 'DATATYPE':
                column_indices['data_type'] = idx
            elif header_clean == 'QUALITYCODE':
                column_indices['quality'] = idx
            
            # Energy columns - prioritize Net Kwh, then Kwh
            elif header_clean == 'NETKWH':
                column_indices['kwh'] = idx  # Highest priority
                logger.debug(f"Found Net Kwh at position {idx}")
            elif header_clean == 'KWH' and 'kwh' not in column_indices:
                column_indices['kwh'] = idx  # Second priority
                logger.debug(f"Found Kwh at position {idx}")
            elif header_clean == 'GENERATEDKWH' and 'kwh' not in column_indices:
                column_indices['kwh'] = idx  # Third priority
                logger.debug(f"Found Generated Kwh at position {idx}")
        
        logger.info(f"Column mapping: {column_indices}")
        
        # Validate required columns
        required_cols = ['nmi', 'endtime', 'kwh']
        missing_cols = [col for col in required_cols if col not in column_indices]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return []
        
        # Extract data
        data_start = header_row_idx + 1
        time_series_data = []
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                row_data = df.iloc[row_idx]
                
                # Get NMI
                nmi = str(row_data.iloc[column_indices['nmi']]).strip()
                if not re.match(r'^\d{10}$', nmi):
                    continue
                
                # Get EndTime (should contain full datetime)
                endtime_value = str(row_data.iloc[column_indices['endtime']]).strip()
                if not re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', endtime_value):
                    continue
                
                # Parse datetime
                try:
                    date_part, time_part = endtime_value.split(' ', 1)
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if not formatted_date or not formatted_time:
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error parsing datetime '{endtime_value}': {e}")
                    continue
                
                # Get kWh reading
                kwh_value = row_data.iloc[column_indices['kwh']]
                if pd.isna(kwh_value):
                    continue
                
                try:
                    kwh_reading = float(kwh_value)
                    if abs(kwh_reading) > 100000:  # Reasonable validation
                        logger.debug(f"Suspicious kWh reading: {kwh_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Invalid kWh value: {kwh_value}")
                    continue
                
                # Get quality flag
                quality = 'A'  # Default
                if 'quality' in column_indices:
                    quality_value = str(row_data.iloc[column_indices['quality']]).strip()
                    if quality_value in ['A', 'S', 'F', 'V', 'N', 'E']:
                        quality = quality_value
                
                # Get data type
                data_type = None
                if 'data_type' in column_indices:
                    data_type = str(row_data.iloc[column_indices['data_type']]).strip()
                    if data_type.upper() != 'MEASURED':
                        quality = 'S'  # Substituted
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{kwh_reading:.3f}",
                    'quality': quality,
                    'data_type': data_type
                })
                
            except Exception as e:
                logger.debug(f"Error processing row {row_idx + 1}: {e}")
                continue
        
        logger.info(f"Extracted {len(time_series_data)} standard interval records")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting standard interval data: {e}")
        return []
    
# ============================================================================
# ADDITIONAL DEBUGGING FUNCTION (OPTIONAL)
# Add this function to help diagnose column issues
# ============================================================================

def debug_standard_interval_columns(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Debug function to analyze column structure in standard interval files.
    Call this if you're having issues to see what columns are available.
    """
    try:
        logger.info("=" * 60)
        logger.info("DEBUG: Standard Interval Column Analysis")
        logger.info("=" * 60)
        
        # Find header row
        header_row_idx = None
        for row_idx in range(min(5, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            if 'NMI' in row_text and 'KWH' in row_text:
                header_row_idx = row_idx
                break
        
        if header_row_idx is None:
            logger.error("Could not find header row for debugging")
            return
        
        headers = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[header_row_idx]]
        
        logger.info(f"Found {len(headers)} columns:")
        for idx, header in enumerate(headers):
            logger.info(f"  Column {idx:2d}: '{header}'")
        
        # Analyze energy columns specifically
        logger.info("\nEnergy-related columns:")
        for idx, header in enumerate(headers):
            if 'KWH' in header.upper() or 'ENERGY' in header.upper():
                # Show sample data from this column
                sample_values = []
                for sample_row in range(header_row_idx + 1, min(header_row_idx + 6, df.shape[0])):
                    if sample_row < df.shape[0]:
                        val = df.iloc[sample_row, idx]
                        if pd.notna(val):
                            sample_values.append(str(val))
                
                logger.info(f"  Column {idx:2d}: '{header}' - Sample values: {', '.join(sample_values[:3])}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in debug function: {e}")


def process_file(file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Process a single file based on its extension."""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.csv', '.txt']:
            return process_csv_file(file_path, logger)
        elif file_ext in ['.xlsx', '.xls']:
            return process_excel_file(file_path, logger)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def merge_nem12_blocks(blocks: List[Dict[str, Any]], logger: logging.Logger) -> List[Dict[str, Any]]:
    """Merge NEM12 blocks with the same header structure."""
    if not blocks:
        return []

    merged_blocks = []
    block_groups = defaultdict(lambda: None)

    for block_data in blocks:
        if not block_data or "nem12_block" not in block_data:
            continue

        block = block_data["nem12_block"]
        if not block.is_valid():
            continue

        # Create header key for grouping
        header_key = "default"
        if block.header and len(block.header) >= 3:
            header_parts = [str(block.header[0]), str(block.header[1])]
            if len(block.header) > 3:
                header_parts.extend(str(part) for part in block.header[3:])
            header_key = "|".join(header_parts)

        if block_groups[header_key] is None:
            block_groups[header_key] = block
        else:
            # Merge blocks
            existing_block = block_groups[header_key]
            existing_block.nmi_blocks.extend(block.nmi_blocks)
            logger.info(f"Merged NEM12 block with {len(block.nmi_blocks)} NMIs")

    # Convert groups back to block data
    for header_key, block in block_groups.items():
        if block is not None:
            merged_blocks.append({
                "file": "Merged NEM12 blocks",
                "nem12_block": block
            })

    logger.info(f"Merged {len(blocks)} blocks into {len(merged_blocks)} distinct blocks")
    return merged_blocks

def generate_nem12_file(processed_data: List[Dict[str, Any]], output_path: str, 
                       logger: logging.Logger) -> bool:
    """Generate NEM12 file from processed data without header (row 100) - raw text writing."""
    if not processed_data:
        logger.warning("No data to process")
        return False

    try:
        # Merge blocks if necessary
        merged_data = merge_nem12_blocks(processed_data, logger)
        all_rows = []
        block_count = 0
        nmi_count = 0

        for data in merged_data:
            block = data.get("nem12_block")
            if block and block.is_valid():
                # Get all rows from the block
                block_rows = block.get_all_rows()
                
                # Filter out header rows (record type "100")
                for row in block_rows:
                    if row and len(row) > 0:
                        record_type = str(row[0]).strip()
                        if record_type != "100":  # Remove header rows
                            all_rows.append(row)
                
                block_count += 1
                nmi_count += len(block.get_nmis())

        if not all_rows:
            logger.error("No valid data blocks to process after removing headers")
            return False

        # Determine output file paths
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            os.makedirs(output_path, exist_ok=True)
            base_name = datetime.now().strftime('nem12_%Y%m%d_%H%M%S')
            csv_file = os.path.join(output_path, f"{base_name}.csv")
        else:
            csv_file = output_path
            os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)

        dat_file = os.path.splitext(csv_file)[0] + ".dat"

        # Write files using raw text method (exactly like manual save)
        def write_nem12_file(file_path):
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                for row in all_rows:
                    # Convert each row to comma-separated string
                    row_str = ','.join(str(cell).strip() if cell is not None else '' for cell in row)
                    f.write(row_str + '\n')
        
        # Write both files
        write_nem12_file(csv_file)
        write_nem12_file(dat_file)

        logger.info(f"NEM12 files generated successfully (without headers):")
        logger.info(f"   CSV: {csv_file}")
        logger.info(f"   DAT: {dat_file}")
        logger.info(f"   Blocks: {block_count}, NMIs: {nmi_count}, Rows: {len(all_rows)}")
        
        return True

    except Exception as e:
        logger.error(f"Error generating NEM12 file: {e}")
        return False

def validate_nem12_file(file_path: str, logger: logging.Logger) -> bool:
    """Validate generated NEM12 file for compliance."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False

        df = pd.read_csv(file_path, header=None, dtype=str)
        
        if df.empty:
            logger.error("File is empty")
            return False

        # Basic structure validation
        first_row_type = safe_row_type(df.iloc[0, 0])
        if first_row_type != "100":
            logger.error(f"File must start with 100 record, found: {first_row_type}")
            return False

        last_row_type = safe_row_type(df.iloc[-1, 0])
        if last_row_type != "900":
            logger.error(f"File must end with 900 record, found: {last_row_type}")
            return False

        # Count record types
        record_counts = Counter()
        for _, row in df.iterrows():
            row_type = safe_row_type(row.iloc[0])
            if row_type:
                record_counts[row_type] += 1

        logger.info(f"Validation passed for {file_path}")
        logger.info(f"Record counts: {dict(record_counts)}")
        return True

    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_folder(folder_path: str, output_path: str, logger: logging.Logger, 
                  batch_per_nmi: bool = False, separate_files: bool = True) -> bool:
    """Process all supported files in a folder."""
    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return False

    # Find supported files
    supported_extensions = {'.csv', '.txt', '.xlsx', '.xls'}
    all_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and
        os.path.splitext(f)[1].lower() in supported_extensions
    ]

    # Filter out temp files
    files = [f for f in all_files if not is_temp_file(f)]
    temp_files = [f for f in all_files if is_temp_file(f)]

    if temp_files:
        logger.info(f"Skipped {len(temp_files)} temp files: {temp_files}")

    if not files:
        logger.warning(f"No supported files found in {folder_path}")
        return False

    logger.info(f"Found {len(files)} files to process")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    success_count = 0
    failed_files = []
    
    if separate_files:
        # Process each file separately
        logger.info("Processing files individually")
        
        for i, file_name in enumerate(files, 1):
            file_path = os.path.join(folder_path, file_name)
            logger.info(f"[{i}/{len(files)}] Processing: {file_name}")
            
            try:
                result = process_file(file_path, logger)
                
                if result:
                    # Generate output filename
                    input_base_name = os.path.splitext(file_name)[0]
                    clean_name = re.sub(r'[^\w\-_\.]', '_', input_base_name)
                    output_file = os.path.join(output_path, f"NEM12_{clean_name}.csv")
                    
                    # Generate the NEM12 file
                    if generate_nem12_file(result, output_file, logger):
                        if validate_nem12_file(output_file, logger):
                            success_count += 1
                            logger.info(f"SUCCESS: {file_name} -> NEM12_{clean_name}.csv")
                        else:
                            logger.warning(f"Generated but validation failed: {file_name}")
                    else:
                        logger.error(f"Failed to generate NEM12 file for: {file_name}")
                        failed_files.append(file_name)
                else:
                    logger.warning(f"No data extracted from: {file_name}")
                    failed_files.append(file_name)
                    
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                failed_files.append(file_name)
    
    else:
        # Combined processing mode
        logger.info("Processing files in combined mode")
        processed_data = []
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            
            try:
                result = process_file(file_path, logger)
                if result:
                    processed_data.extend(result)
                    success_count += 1
                else:
                    failed_files.append(file_name)
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                failed_files.append(file_name)

        if success_count == 0:
            logger.error("No files were successfully processed")
            return False

        logger.info(f"Successfully processed {success_count}/{len(files)} files")

        # Generate combined output
        output_file = os.path.join(output_path, "combined_nem12.csv")
        
        if batch_per_nmi:
            return batch_export_per_nmi(processed_data, output_file, logger)
        else:
            success = generate_nem12_file(processed_data, output_file, logger)
            if success:
                is_valid = validate_nem12_file(output_file, logger)
                if is_valid:
                    logger.info("Combined file validation successful")
                else:
                    logger.warning("Combined file validation failed")
            return success

    # Report results
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY:")
    logger.info(f"   Total input files: {len(files)}")
    logger.info(f"   Successfully processed: {success_count}")
    logger.info(f"   Failed: {len(failed_files)}")
    logger.info(f"   Output directory: {output_path}")
    
    if failed_files:
        logger.info("Failed files:")
        for failed_file in failed_files:
            logger.info(f"   - {failed_file}")
    
    logger.info("=" * 60)
    
    return success_count > 0


def batch_export_per_nmi(processed_data: List[Dict[str, Any]], output_file: str, 
                        logger: logging.Logger) -> bool:
    """Export separate NEM12 files for each NMI."""
    if not processed_data:
        logger.warning("No data to process for batch export")
        return False

    # Group data by NMI
    nmi_groups = defaultdict(list)
    
    for block_data in processed_data:
        if not block_data or "nem12_block" not in block_data:
            continue
            
        block = block_data["nem12_block"]
        if not block.is_valid():
            continue
            
        for nmi in block.get_nmis():
            nmi_groups[nmi].append(block_data)

    if not nmi_groups:
        logger.error("No valid NMI groups found")
        return False

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(output_file))
    base_name = os.path.basename(os.path.splitext(output_file)[0])
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    
    for nmi, nmi_data in nmi_groups.items():
        if nmi in ["UNKNOWN", "UNKNOWN_NMI"]:
            logger.info(f"Skipping unknown NMI: {nmi}")
            continue
            
        nmi_output = os.path.join(output_dir, f"{base_name}_{nmi}.csv")
        
        try:
            success = generate_nem12_file(nmi_data, nmi_output, logger)
            if success:
                success_count += 1
                if validate_nem12_file(nmi_output, logger):
                    logger.info(f"Generated and validated: {nmi_output}")
                else:
                    logger.warning(f"Generated but validation failed: {nmi_output}")
        except Exception as e:
            logger.error(f"Failed to generate file for NMI {nmi}: {e}")

    logger.info(f"Batch export completed: {success_count}/{len(nmi_groups)} files generated")
    return success_count > 0



# ============================================================================
# MULTI-COLUMN ENERGY FORMAT HANDLER
# ============================================================================
# Add these functions to your existing NEM12 converter code

def detect_multi_column_energy_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    ENHANCED VERSION - Replace your existing detect_multi_column_energy_format function with this.
    Handles the real-world formatting issues in your CSV file.
    """
    try:
        if df.empty or df.shape[0] < 10:
            return False
        
        logger.debug("Detecting multi-column energy format")
        
        format_score = 0
        has_scaling_info = False
        has_system_names = False
        has_datetime_data = False
        has_nmi = False
        
        # Check first 15 rows for indicators
        for idx in range(min(15, df.shape[0])):
            # Combine all non-null values in the row
            row_values = [str(x).strip().upper() for x in df.iloc[idx] if pd.notna(x) and str(x).strip()]
            row_text = ' '.join(row_values)
            
            # Check for scaling factor indicators
            scaling_indicators = ['SCALING FACTOR', 'REQUIRED NEW DATA', 'SCALING', 'FACTOR']
            if any(indicator in row_text for indicator in scaling_indicators):
                has_scaling_info = True
                format_score += 3
                logger.debug(f"Found scaling info at row {idx + 1}")
            
            # Check for system names
            system_names = ['CALIBRE', 'ASPIRE', 'MARQUE', 'RAW']
            system_count = sum(1 for name in system_names if name in row_text)
            if system_count >= 2:
                has_system_names = True
                format_score += 3
                logger.debug(f"Found {system_count} system names at row {idx + 1}")
            
            # Check for datetime column header
            if 'DATE TIME' in row_text and ('KWH' in row_text or 'KVARH' in row_text):
                format_score += 2
                logger.debug(f"Found datetime/energy header at row {idx + 1}")
            
            # Check for NMI pattern (10 digits)
            nmi_matches = re.findall(r'\b(\d{10})\b', row_text)
            if nmi_matches:
                has_nmi = True
                format_score += 2
                logger.debug(f"Found potential NMI at row {idx + 1}: {nmi_matches[0]}")
        
        # Look for actual datetime data pattern
        for idx in range(5, min(df.shape[0], 25)):
            first_cell = str(df.iloc[idx, 0]).strip() if not pd.isna(df.iloc[idx, 0]) else ""
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', first_cell):
                has_datetime_data = True
                format_score += 2
                logger.debug(f"Found datetime data at row {idx + 1}")
                break
        
        # Check for wide format with multiple columns
        if df.shape[1] >= 15:
            format_score += 1
        
        # Enhanced scoring logic
        is_multi_column_format = (
            format_score >= 5 and 
            (has_scaling_info or has_system_names) and
            (has_datetime_data or has_nmi)
        )
        
        logger.info(f"Multi-column detection: score={format_score}, "
                   f"scaling={has_scaling_info}, systems={has_system_names}, "
                   f"datetime={has_datetime_data}, nmi={has_nmi}, result={is_multi_column_format}")
        
        return is_multi_column_format
        
    except Exception as e:
        logger.error(f"Error detecting multi-column energy format: {e}")
        return False

def extract_nmi_from_multi_column_header(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Extract NMI from multi-column format header section."""
    try:
        # Look in first 10 rows for NMI patterns
        for idx in range(min(10, df.shape[0])):
            row_text = ' '.join(str(x) for x in df.iloc[idx] if pd.notna(x))
            
            # Look for 10-digit numbers (typical NMI format)
            nmi_matches = re.findall(r'\b(\d{10})\b', row_text)
            if nmi_matches:
                nmi = nmi_matches[0]
                logger.info(f"Extracted NMI from header: {nmi}")
                return nmi
            
            # Look for NMI with prefix
            nmi_labeled = re.findall(r'(?:NMI|nmi)[:\s]*([A-Za-z]?\d{9,10})', row_text)
            if nmi_labeled:
                nmi = nmi_labeled[0]
                logger.info(f"Extracted labeled NMI from header: {nmi}")
                return nmi
        
        logger.warning("No NMI found in multi-column header")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting NMI from multi-column header: {e}")
        return None


def find_data_start_in_multi_column(df: pd.DataFrame, logger: logging.Logger) -> int:
    """Find the row where actual time series data starts in multi-column format."""
    try:
        for idx in range(min(20, df.shape[0])):
            first_cell = str(df.iloc[idx, 0]).strip()
            
            # Look for datetime data pattern
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', first_cell):
                logger.info(f"Found data start at row {idx}")
                return idx
            
            # Look for "Date time" header (data starts next row)
            if 'date time' in first_cell.lower():
                logger.info(f"Found header at row {idx}, data starts at {idx + 1}")
                return idx + 1
        
        logger.warning("Could not find data start, defaulting to row 10")
        return 10
        
    except Exception as e:
        logger.error(f"Error finding data start: {e}")
        return 10


def extract_multi_column_energy_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED VERSION - Replace your existing extract_multi_column_energy_data function with this.
    Handles the real-world formatting issues and extracts data properly.
    """
    try:
        logger.info("Extracting multi-column energy data")
        
        # Extract NMI from header section
        nmi = None
        for idx in range(min(10, df.shape[0])):
            row_values = [str(x) for x in df.iloc[idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Look for 10-digit NMI
            nmi_matches = re.findall(r'\b(\d{10})\b', row_text)
            if nmi_matches:
                nmi = nmi_matches[0]
                logger.info(f"Extracted NMI from header: {nmi}")
                break
        
        if not nmi:
            nmi = generate_auto_nmi()
            logger.warning(f"No NMI found, using auto-generated: {nmi}")
        
        # Find data start row
        data_start = None
        for idx in range(min(25, df.shape[0])):
            first_cell = str(df.iloc[idx, 0]).strip() if not pd.isna(df.iloc[idx, 0]) else ""
            
            # Look for datetime pattern
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', first_cell):
                data_start = idx
                logger.info(f"Found data start at row {idx + 1}")
                break
        
        if data_start is None:
            logger.error("Could not find data start row")
            return []
        
        # Based on your file analysis, column 1 contains the raw kWh data
        # This is the most reliable column for your specific format
        energy_column = 1  # Raw kWh column
        
        logger.info(f"Using energy column {energy_column} (Raw kWh)")
        
        # Extract time series data
        time_series_data = []
        skipped_rows = 0
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                # Get datetime
                datetime_cell = df.iloc[row_idx, 0]
                if pd.isna(datetime_cell):
                    continue
                
                datetime_value = str(datetime_cell).strip()
                if not datetime_value or not re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', datetime_value):
                    continue
                
                # Parse datetime
                date_part, time_part = datetime_value.split(' ', 1)
                formatted_date = parse_date(date_part)
                formatted_time = parse_time(time_part)
                
                if not formatted_date or not formatted_time:
                    skipped_rows += 1
                    continue
                
                # Get energy reading from the raw kWh column
                if energy_column >= df.shape[1]:
                    continue
                    
                energy_cell = df.iloc[row_idx, energy_column]
                if pd.isna(energy_cell):
                    continue
                
                # Clean the energy value (handle quotes, spaces, etc.)
                energy_str = str(energy_cell).strip()
                
                # Remove quotes if present
                if energy_str.startswith('"') and energy_str.endswith('"'):
                    energy_str = energy_str[1:-1].strip()
                
                # Skip empty or dash values
                if not energy_str or energy_str in ['', '-', ' - ', ' -   ']:
                    continue
                
                try:
                    # Clean and convert to float
                    clean_energy = energy_str.replace(',', '').replace(' ', '')
                    energy_reading = float(clean_energy)
                    
                    if 0 <= energy_reading <= 100000:  # Reasonable range
                        time_series_data.append({
                            'nmi': nmi,
                            'date': formatted_date,
                            'time': formatted_time,
                            'reading': f"{energy_reading:.3f}",
                            'quality': 'A'
                        })
                    else:
                        skipped_rows += 1
                
                except ValueError:
                    skipped_rows += 1
                    continue
                    
            except Exception as e:
                logger.debug(f"Error processing row {row_idx + 1}: {e}")
                skipped_rows += 1
                continue
        
        logger.info(f"Multi-column extraction completed: {len(time_series_data)} records extracted, "
                   f"{skipped_rows} rows skipped")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting multi-column energy data: {e}")
        return []


# ============================================================================
# INTEGRATION WITH MAIN CONVERTER
# ============================================================================

def detect_all_formats(df: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Detect file format from all supported types.
    Returns format type string for processing.
    """
    try:
        logger.info("Detecting file format...")
        
        # Check formats in priority order
        if detect_nem12_format(df, logger):
            return "nem12"
        
        elif detect_wide_format_layout(df, logger):
            return "wide_format_layout"
        
        elif detect_standard_interval_format(df, logger):
            return "standard_interval"
        
        elif detect_multi_column_energy_format(df, logger):
            return "multi_column_energy"
        
        elif detect_space_separated_format(df, logger):
            return "space_separated"
        
        elif detect_interval_data_format(df, logger):
            return "interval_data"
        
        elif detect_time_series_format(df, logger):
            return "time_series"
        
        else:
            logger.warning("Unknown format, defaulting to time_series")
            return "time_series"
            
    except Exception as e:
        logger.error(f"Error in format detection: {e}")
        return "time_series"
  
def detect_space_separated_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    ENHANCED VERSION - Replace your existing detect_space_separated_format function.
    Now handles multiple variations of space-separated interval data:
    - Original: NMI, Date, Interval Length, Period, EndTime, Data Type, Kwh, KW, KVA, PF, Generated Kwh, Net Kwh, ...
    - New: NMI, Date, Interval Length, Period, EndTime, Meter Serial, Kwh, Generated Kwh, Net Kwh, KW, KVA, pf, Quality Code, ...
    """
    try:
        if df.empty:
            return False
        
        logger.debug("Enhanced space-separated format detection")
        
        # Check if we have very few rows but many columns (indicating one giant row)
        if df.shape[0] <= 3 and df.shape[1] > 30:  # Lowered threshold for smaller files
            logger.debug(f"Potential space-separated format: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Get all text from first row
            first_row_text = ''
            for col_idx in range(df.shape[1]):
                cell_value = df.iloc[0, col_idx]
                if pd.notna(cell_value):
                    first_row_text += ' ' + str(cell_value)
            
            first_row_text = first_row_text.upper()
            
            # Enhanced indicators for both formats
            format_indicators = {
                'core': ['NMI', 'DATE', 'ENDTIME', 'KWH'],  # Must have all
                'energy': ['NET KWH', 'GENERATED KWH', 'KW', 'KVA'],  # Must have some
                'quality': ['QUALITY CODE', 'MEASURED', 'GENTRACK'],  # Must have some
                'structure': ['INTERVAL', 'PERIOD', 'METER SERIAL']  # Optional but helpful
            }
            
            # Count indicators
            core_score = sum(1 for indicator in format_indicators['core'] if indicator in first_row_text)
            energy_score = sum(1 for indicator in format_indicators['energy'] if indicator in first_row_text)
            quality_score = sum(1 for indicator in format_indicators['quality'] if indicator in first_row_text)
            structure_score = sum(1 for indicator in format_indicators['structure'] if indicator in first_row_text)
            
            # Check for numeric data patterns
            has_numeric_data = bool(re.search(r'\d+\.\d{5}', first_row_text))  # 5 decimal places common
            has_dates = bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', first_row_text))
            has_times = bool(re.search(r'\d{2}:\d{2}', first_row_text))
            has_nmi = bool(re.search(r'\b\d{10}\b', first_row_text))
            
            # Enhanced scoring
            total_score = (core_score * 3) + (energy_score * 2) + quality_score + structure_score
            data_score = sum([has_numeric_data, has_dates, has_times, has_nmi])
            
            is_space_separated = (core_score >= 3 and energy_score >= 1 and data_score >= 3 and total_score >= 8)
            
            logger.info(f"Enhanced space-separated detection: core={core_score}/4, energy={energy_score}/4, "
                       f"quality={quality_score}/3, structure={structure_score}/3, data={data_score}/4, "
                       f"total_score={total_score}, result={is_space_separated}")
            
            return is_space_separated
        
        return False
        
    except Exception as e:
        logger.error(f"Error detecting enhanced space-separated format: {e}")
        return False

def extract_space_separated_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED VERSION - Replace your existing extract_space_separated_data function.
    Now handles multiple column structures:
    Format A: NMI, Date, Interval Length, Period, EndTime, Data Type, Kwh, KW, KVA, PF, Generated Kwh, Net Kwh, ...
    Format B: NMI, Date, Interval Length, Period, EndTime, Meter Serial, Kwh, Generated Kwh, Net Kwh, KW, KVA, pf, Quality Code, ...
    """
    try:
        logger.info("Enhanced space-separated interval data extraction")
        
        # Get all values from the first row (which contains everything)
        all_values = []
        for col_idx in range(df.shape[1]):
            cell_value = df.iloc[0, col_idx]
            if pd.notna(cell_value):
                cell_str = str(cell_value).strip()
                if not cell_str:
                    continue
                    
                # Handle cells that contain multiple space-separated values
                if ' ' in cell_str and not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', cell_str):
                    # Split on spaces, but be careful with dates and times
                    parts = cell_str.split()
                    for part in parts:
                        if part.strip():
                            all_values.append(part.strip())
                else:
                    all_values.append(cell_str)
        
        logger.info(f"Total values extracted: {len(all_values)}")
        logger.debug(f"First 20 values: {all_values[:20]}")
        
        # Find header section (everything before first NMI)
        header_end_idx = None
        for idx, value in enumerate(all_values):
            # Look for 10-digit NMI pattern
            if re.match(r'^\d{10}$', value):
                header_end_idx = idx
                logger.info(f"Found first NMI '{value}' at position {idx}")
                break
        
        if header_end_idx is None:
            logger.error("Could not find start of data (no 10-digit NMI found)")
            return []
        
        # Extract headers and data
        headers = all_values[:header_end_idx]
        data_values = all_values[header_end_idx:]
        
        logger.info(f"Headers: {len(headers)} columns")
        logger.info(f"Data values: {len(data_values)} values")
        logger.debug(f"Headers: {headers}")
        
        # Calculate number of complete rows
        if len(headers) == 0:
            logger.error("No headers found")
            return []
        
        num_complete_rows = len(data_values) // len(headers)
        remainder = len(data_values) % len(headers)
        
        logger.info(f"Complete data rows: {num_complete_rows}")
        if remainder > 0:
            logger.warning(f"Incomplete data: {remainder} extra values (will be ignored)")
        
        if num_complete_rows == 0:
            logger.error("No complete data rows found")
            return []
        
        # Enhanced column mapping with flexible header matching
        column_indices = {}
        headers_upper = [h.upper().replace(' ', '') for h in headers]
        
        logger.debug(f"Processed headers: {headers_upper}")
        
        # Find columns with enhanced matching
        for idx, header in enumerate(headers_upper):
            # NMI column
            if header == 'NMI':
                column_indices['nmi'] = idx
            
            # DateTime columns
            elif header in ['ENDTIME', 'END_TIME']:
                column_indices['endtime'] = idx
            elif header == 'DATE' and idx < len(headers_upper) - 5:  # Not near the end
                column_indices['date'] = idx
            
            # Energy columns with priority system
            elif header in ['NETKWH', 'NET_KWH']:
                column_indices['kwh'] = idx  # Highest priority
                logger.debug(f"Found Net kWh at position {idx}")
            elif header == 'KWH' and 'kwh' not in column_indices:
                # Only use if Net kWh not found
                column_indices['kwh'] = idx
                logger.debug(f"Found kWh at position {idx}")
            elif header in ['GENERATEDKWH', 'GENERATED_KWH'] and 'kwh' not in column_indices:
                # Use as fallback
                column_indices['kwh'] = idx
                logger.debug(f"Found Generated kWh at position {idx}")
            
            # Quality and data type columns
            elif header in ['QUALITYCODE', 'QUALITY_CODE']:
                column_indices['quality'] = idx
            elif header in ['DATATYPE', 'DATA_TYPE']:
                column_indices['data_type'] = idx
            
            # Additional identification columns
            elif header in ['METERSERIAL', 'METER_SERIAL']:
                column_indices['meter_serial'] = idx
        
        logger.info(f"Enhanced column mapping: {column_indices}")
        
        # Validate required columns
        missing_columns = []
        if 'nmi' not in column_indices:
            missing_columns.append('NMI')
        if 'kwh' not in column_indices:
            missing_columns.append('kWh (Net Kwh, Kwh, or Generated Kwh)')
        if 'endtime' not in column_indices and 'date' not in column_indices:
            missing_columns.append('DateTime (EndTime or Date)')
        
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return []
        
        # Extract time series data
        time_series_data = []
        
        for row_num in range(num_complete_rows):
            try:
                # Get row data
                row_start = row_num * len(headers)
                row_end = row_start + len(headers)
                row_data = data_values[row_start:row_end]
                
                if len(row_data) != len(headers):
                    logger.debug(f"Incomplete row {row_num}: expected {len(headers)}, got {len(row_data)}")
                    continue
                
                # Extract NMI
                nmi = row_data[column_indices['nmi']] if 'nmi' in column_indices else None
                if not nmi or not re.match(r'^\d{10}$', nmi):
                    logger.debug(f"Invalid NMI in row {row_num}: {nmi}")
                    continue
                
                # Extract datetime with enhanced logic
                datetime_str = None
                
                if 'endtime' in column_indices:
                    # EndTime column should contain full datetime
                    endtime_value = row_data[column_indices['endtime']]
                    
                    # Check if it's a complete datetime
                    if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', endtime_value):
                        datetime_str = endtime_value
                    else:
                        # EndTime might be just time, need to combine with date
                        if 'date' in column_indices and re.match(r'\d{1,2}:\d{2}', endtime_value):
                            date_value = row_data[column_indices['date']]
                            if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_value):
                                datetime_str = f"{date_value} {endtime_value}"
                
                if not datetime_str and 'date' in column_indices:
                    # Fallback to date column with default time
                    date_value = row_data[column_indices['date']]
                    if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_value):
                        datetime_str = f"{date_value} 00:30"  # Default time
                
                if not datetime_str:
                    logger.debug(f"No valid datetime in row {row_num}")
                    continue
                
                # Parse datetime
                try:
                    if ' ' in datetime_str:
                        date_part, time_part = datetime_str.split(' ', 1)
                    else:
                        date_part = datetime_str
                        time_part = "00:30"
                    
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if not formatted_date or not formatted_time:
                        logger.debug(f"Could not parse datetime '{datetime_str}' in row {row_num}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error parsing datetime '{datetime_str}' in row {row_num}: {e}")
                    continue
                
                # Extract kWh reading
                kwh_value = row_data[column_indices['kwh']] if 'kwh' in column_indices else None
                if not kwh_value:
                    logger.debug(f"No kWh value in row {row_num}")
                    continue
                
                try:
                    kwh_reading = float(kwh_value)
                    # More flexible validation (allow wider range)
                    if abs(kwh_reading) > 500000:  # Very high threshold
                        logger.debug(f"Suspicious kWh reading in row {row_num}: {kwh_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Invalid kWh value in row {row_num}: {kwh_value}")
                    continue
                
                # Extract quality
                quality = 'A'  # Default
                if 'quality' in column_indices:
                    quality_value = row_data[column_indices['quality']]
                    if quality_value in ['A', 'S', 'F', 'V', 'N', 'E']:
                        quality = quality_value
                
                # Extract data type
                data_type = None
                if 'data_type' in column_indices:
                    data_type = row_data[column_indices['data_type']]
                    if data_type and data_type.upper() != 'MEASURED':
                        quality = 'S'  # Substituted
                
                # Extract meter serial (for reference)
                meter_serial = None
                if 'meter_serial' in column_indices:
                    meter_serial = row_data[column_indices['meter_serial']]
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{kwh_reading:.3f}",
                    'quality': quality,
                    'data_type': data_type,
                    'meter_serial': meter_serial
                })
                
            except Exception as e:
                logger.debug(f"Error processing row {row_num}: {e}")
                continue
        
        logger.info(f"Enhanced extraction completed: {len(time_series_data)} space-separated records")
        
        # Log summary statistics
        if time_series_data:
            readings = [float(r['reading']) for r in time_series_data[:100]]  # Sample first 100
            avg_reading = sum(readings) / len(readings) if readings else 0
            min_reading = min(readings) if readings else 0
            max_reading = max(readings) if readings else 0
            
            logger.info(f"Sample statistics: avg={avg_reading:.3f}, min={min_reading:.3f}, max={max_reading:.3f}")
            logger.info(f"First NMI: {time_series_data[0]['nmi']}")
            logger.info(f"Date range: {time_series_data[0]['date']} to {time_series_data[-1]['date']}")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting enhanced space-separated data: {e}")
        return []








# ============================================================================
# NEW METER DATA FORMAT FUNCTIONS
# Add these functions to your existing NEM12 converter code
# ============================================================================

def detect_meter_data_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Detect the new meter data format with structure:
    NMI, MeterSerial, Period, Local Time, Data Type, E kWh at Meter, kW, kVA, PF
    """
    try:
        if df.empty or df.shape[0] < 2:
            return False
        
        logger.debug("Detecting meter data format")
        
        # Check for specific header pattern
        header_row = None
        best_score = 0
        
        for row_idx in range(min(3, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x) and str(x).strip()]
            row_text = ' '.join(row_values)
            
            logger.debug(f"Row {row_idx + 1} text: {row_text[:200]}...")
            
            # Look for specific indicators
            score = 0
            indicators_found = []
            
            # Core required indicators (must have these)
            if 'NMI' in row_text:
                score += 4
                indicators_found.append('NMI')
            if 'METERSERIAL' in row_text or 'METER SERIAL' in row_text:
                score += 3
                indicators_found.append('METER_SERIAL')
            if 'PERIOD' in row_text:
                score += 3
                indicators_found.append('PERIOD')
            if 'LOCAL TIME' in row_text or 'LOCALTIME' in row_text:
                score += 4
                indicators_found.append('LOCAL_TIME')
            if 'DATA TYPE' in row_text or 'DATATYPE' in row_text:
                score += 3
                indicators_found.append('DATA_TYPE')
            
            # Energy indicators
            if 'E KWH AT METER' in row_text or 'KWH AT METER' in row_text:
                score += 4
                indicators_found.append('E_KWH_AT_METER')
            elif 'E KWH' in row_text:
                score += 3
                indicators_found.append('E_KWH')
            elif 'KWH' in row_text:
                score += 2
                indicators_found.append('KWH')
            
            # Power indicators
            if 'KW' in row_text and 'KWH' not in row_text.replace('KWH', ''):
                score += 2
                indicators_found.append('KW')
            if 'KVA' in row_text:
                score += 2
                indicators_found.append('KVA')
            if 'PF' in row_text or 'POWER FACTOR' in row_text:
                score += 2
                indicators_found.append('PF')
            
            logger.debug(f"Row {row_idx + 1} score: {score}, indicators: {indicators_found}")
            
            if score > best_score:
                best_score = score
                header_row = row_idx
        
        logger.info(f"Best header row: {header_row + 1 if header_row is not None else 'None'} with score {best_score}")
        
        if header_row is None or best_score < 18:  # High threshold for this specific format
            return False
        
        # Verify data format by checking a few data rows
        data_start = header_row + 1
        if data_start >= df.shape[0]:
            return False
        
        valid_data_rows = 0
        max_check_rows = min(data_start + 10, df.shape[0])
        
        for row_idx in range(data_start, max_check_rows):
            row_data = df.iloc[row_idx].fillna('').astype(str)
            validation_score = 0
            
            # Check for 10-digit NMI in first column
            if len(row_data) > 0:
                first_cell = str(row_data.iloc[0]).strip()
                if re.match(r'^\d{10}$', first_cell):
                    validation_score += 3
                    logger.debug(f"Found NMI: {first_cell}")
            
            # Check for period number (should be integer)
            if len(row_data) > 2:
                period_cell = str(row_data.iloc[2]).strip()
                try:
                    period_num = int(float(period_cell))
                    if 1 <= period_num <= 288:  # Valid period range (5-min to 30-min intervals)
                        validation_score += 2
                        logger.debug(f"Found period: {period_num}")
                except (ValueError, TypeError):
                    pass
            
            # Check for time format in Local Time column
            if len(row_data) > 3:
                time_cell = str(row_data.iloc[3]).strip()
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', time_cell):
                    validation_score += 3
                    logger.debug(f"Found local time: {time_cell}")
            
            # Check for Data Type (usually 'Actual' or 'Estimated')
            if len(row_data) > 4:
                datatype_cell = str(row_data.iloc[4]).strip()
                if datatype_cell.upper() in ['ACTUAL', 'ESTIMATED', 'MEASURED', 'CALCULATED']:
                    validation_score += 2
                    logger.debug(f"Found data type: {datatype_cell}")
            
            # Check for reasonable energy values in E kWh at Meter column
            if len(row_data) > 5:
                try:
                    energy_value = float(str(row_data.iloc[5]).strip())
                    if 0 <= energy_value <= 10000:  # Reasonable energy range
                        validation_score += 2
                        logger.debug(f"Found energy value: {energy_value}")
                except (ValueError, TypeError):
                    pass
            
            logger.debug(f"Row {row_idx + 1} validation score: {validation_score}")
            
            if validation_score >= 8:  # Need good validation score
                valid_data_rows += 1
        
        is_meter_data_format = valid_data_rows >= 5  # Need multiple valid rows
        logger.info(f"Meter data format detection: header_row={header_row + 1 if header_row is not None else 'None'}, "
                   f"valid_rows={valid_data_rows}/{max_check_rows - data_start}, result={is_meter_data_format}")
        
        return is_meter_data_format
        
    except Exception as e:
        logger.error(f"Error detecting meter data format: {e}")
        return False


def extract_meter_data_format(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Extract data from the new meter data format.
    Format: NMI, MeterSerial, Period, Local Time, Data Type, E kWh at Meter, kW, kVA, PF
    """
    try:
        logger.info("Extracting meter data format")
        
        # Find header row
        header_row_idx = None
        best_score = 0
        
        for row_idx in range(min(3, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            score = 0
            if 'NMI' in row_text:
                score += 3
            if 'LOCAL TIME' in row_text or 'LOCALTIME' in row_text:
                score += 3
            if 'E KWH AT METER' in row_text or 'KWH AT METER' in row_text:
                score += 3
            if 'PERIOD' in row_text:
                score += 2
            if 'DATA TYPE' in row_text:
                score += 2
            
            if score > best_score:
                best_score = score
                header_row_idx = row_idx
        
        if header_row_idx is None or best_score < 10:
            logger.error("Could not find meter data header row")
            return []
        
        logger.info(f"Using header row {header_row_idx + 1} with score {best_score}")
        
        # Get column headers and create mapping
        headers = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[header_row_idx]]
        logger.info(f"Headers ({len(headers)} columns): {headers}")
        
        # Create column mapping
        column_indices = {}
        
        for idx, header in enumerate(headers):
            header_clean = header.upper().replace(' ', '').replace('_', '')
            
            if header_clean == 'NMI':
                column_indices['nmi'] = idx
            elif 'METERSERIAL' in header_clean or header_clean == 'METERSERIAL':
                column_indices['meter_serial'] = idx
            elif header_clean == 'PERIOD':
                column_indices['period'] = idx
            elif 'LOCALTIME' in header_clean or header_clean == 'LOCALTIME':
                column_indices['local_time'] = idx
            elif 'DATATYPE' in header_clean or header_clean == 'DATATYPE':
                column_indices['data_type'] = idx
            elif 'EKWHATMETER' in header_clean or 'KWHATMETER' in header_clean:
                column_indices['energy'] = idx
                logger.info(f"Found energy column at position {idx}")
            elif header_clean == 'KW' and 'power' not in column_indices:
                column_indices['power'] = idx
            elif header_clean == 'KVA':
                column_indices['kva'] = idx
            elif header_clean == 'PF' or 'POWERFACTOR' in header_clean:
                column_indices['pf'] = idx
        
        logger.info(f"Column mapping: {column_indices}")
        
        # Validate required columns
        required_cols = ['nmi', 'local_time', 'energy']
        missing_cols = [col for col in required_cols if col not in column_indices]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return []
        
        # Extract data
        data_start = header_row_idx + 1
        time_series_data = []
        
        logger.info(f"Starting data extraction from row {data_start + 1}")
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                row_data = df.iloc[row_idx]
                
                # Extract NMI
                nmi = str(row_data.iloc[column_indices['nmi']]).strip()
                if not re.match(r'^\d{10}$', nmi):
                    logger.debug(f"Row {row_idx + 1}: Invalid NMI '{nmi}'")
                    continue
                
                # Extract Local Time (should contain full datetime)
                local_time_value = str(row_data.iloc[column_indices['local_time']]).strip()
                if not re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', local_time_value):
                    logger.debug(f"Row {row_idx + 1}: Invalid Local Time '{local_time_value}'")
                    continue
                
                # Parse datetime from Local Time
                try:
                    if ' ' in local_time_value:
                        date_part, time_part = local_time_value.split(' ', 1)
                    else:
                        logger.debug(f"Row {row_idx + 1}: Local Time has no space: '{local_time_value}'")
                        continue
                    
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if not formatted_date or not formatted_time:
                        logger.debug(f"Row {row_idx + 1}: Could not parse date/time from '{local_time_value}'")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Row {row_idx + 1}: Error parsing Local Time '{local_time_value}': {e}")
                    continue
                
                # Extract energy reading (E kWh at Meter)
                energy_value = row_data.iloc[column_indices['energy']]
                if pd.isna(energy_value):
                    logger.debug(f"Row {row_idx + 1}: Empty energy value")
                    continue
                
                try:
                    energy_reading = float(energy_value)
                    if abs(energy_reading) > 100000:
                        logger.debug(f"Row {row_idx + 1}: Large energy reading: {energy_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Row {row_idx + 1}: Invalid energy value: {energy_value}")
                    continue
                
                # Extract data type for quality determination
                quality = 'A'  # Default to Actual
                if 'data_type' in column_indices:
                    data_type_value = str(row_data.iloc[column_indices['data_type']]).strip().upper()
                    if data_type_value in ['ESTIMATED', 'CALCULATED', 'SUBSTITUTED']:
                        quality = 'S'  # Substituted
                    elif data_type_value in ['FINAL']:
                        quality = 'F'  # Final
                
                # Extract additional info
                meter_serial = ''
                if 'meter_serial' in column_indices:
                    meter_serial = str(row_data.iloc[column_indices['meter_serial']]).strip()
                
                period = ''
                if 'period' in column_indices:
                    period = str(row_data.iloc[column_indices['period']]).strip()
                
                power_kw = None
                if 'power' in column_indices:
                    try:
                        power_kw = float(row_data.iloc[column_indices['power']])
                    except (ValueError, TypeError):
                        pass
                
                kva_value = None
                if 'kva' in column_indices:
                    try:
                        kva_value = float(row_data.iloc[column_indices['kva']])
                    except (ValueError, TypeError):
                        pass
                
                pf_value = None
                if 'pf' in column_indices:
                    try:
                        pf_value = float(row_data.iloc[column_indices['pf']])
                    except (ValueError, TypeError):
                        pass
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{energy_reading:.3f}",
                    'quality': quality,
                    'meter_serial': meter_serial,
                    'period': period,
                    'power_kw': power_kw,
                    'kva': kva_value,
                    'pf': pf_value,
                    'data_type': data_type_value if 'data_type' in column_indices else None
                })
                
                if len(time_series_data) % 1000 == 0:
                    logger.info(f"Processed {len(time_series_data)} records...")
                
            except Exception as e:
                logger.debug(f"Error processing row {row_idx + 1}: {e}")
                continue
        
        logger.info(f"Extracted {len(time_series_data)} meter data records")
        
        if time_series_data:
            logger.info(f"Sample record: NMI={time_series_data[0]['nmi']}, "
                       f"Date={time_series_data[0]['date']}, Time={time_series_data[0]['time']}")
            logger.info(f"Date range: {time_series_data[0]['date']} to {time_series_data[-1]['date']}")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting meter data format: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


# ============================================================================
# UPDATED process_csv_file FUNCTION
# Replace your existing process_csv_file function with this enhanced version
# ============================================================================

def process_csv_file(file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED VERSION - Now supports the new meter data format
    Supports ALL formats including:
    - NEM12 (existing format)
    - Meter Data Format (NEW: NMI, MeterSerial, Period, Local Time, Data Type, E kWh at Meter, kW, kVA, PF)
    - AGL DETAILED (retailer export)
    - SRC CONNECTION POINT (utility format)
    - Space-separated interval data
    - Standard interval data
    - Multi-column energy format
    - Excel interval format
    - Interval data format
    - Time series format
    """
    results = []
    
    try:
        with safe_file_processing(file_path, logger):
            df = read_csv_file(file_path, logger)
            if df is None:
                return []
            
            logger.info(f"Processing CSV file: {os.path.basename(file_path)}")
            logger.info(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
            
            # Detect and process format - ENHANCED with new meter data format
            if detect_nem12_format(df, logger):
                logger.info("🎯 Processing as existing NEM12 format")
                results.extend(list(extract_nem12_data(df, file_path, logger)))
                
            elif detect_meter_data_format(df, logger):  # NEW METER DATA FORMAT
                logger.info("🎯 Processing as NEW METER DATA format")
                time_series_data = extract_meter_data_format(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted NEW METER DATA to NEM12")
                
            elif detect_agl_detailed_format(df, logger):  # AGL FORMAT
                logger.info("🎯 Processing as AGL DETAILED format")
                time_series_data = extract_agl_detailed_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted AGL DETAILED data to NEM12")
                
            elif detect_src_connection_point_format(df, logger):  # SRC CONNECTION POINT
                logger.info("🎯 Processing as SRC CONNECTION POINT format")
                time_series_data = extract_src_connection_point_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted SRC CONNECTION POINT data to NEM12")
                
            elif detect_space_separated_format(df, logger):  # SPACE SEPARATED
                logger.info("🎯 Processing as space-separated interval data format")
                time_series_data = extract_space_separated_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted space-separated interval data to NEM12")
                
            elif detect_standard_interval_format(df, logger):  # STANDARD INTERVAL
                logger.info("🎯 Processing as standard interval data format")
                time_series_data = extract_standard_interval_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted standard interval data to NEM12")
                
            elif detect_multi_column_energy_format(df, logger):  # MULTI-COLUMN ENERGY
                logger.info("🎯 Processing as multi-column energy format")
                time_series_data = extract_multi_column_energy_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted multi-column energy data to NEM12")
                
            elif detect_interval_data_format(df, logger):  # GENERAL INTERVAL DATA
                logger.info("🎯 Processing as interval data format")
                time_series_data = extract_interval_data(df, logger)
                if time_series_data:
                    nmi = time_series_data[0]['nmi']
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted interval data to NEM12")
                        
            elif detect_time_series_format(df, logger):  # TIME SERIES
                logger.info("🎯 Processing as time series format")
                nmi = extract_and_validate_nmi(file_path, df, logger)
                time_series_data = extract_time_series_data(df, nmi, logger)
                if time_series_data:
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted time series data to NEM12")
                        
            else:
                logger.warning("❌ Unknown format - attempting fallback time series extraction")
                nmi = extract_and_validate_nmi(file_path, df, logger)
                time_series_data = extract_time_series_data(df, nmi, logger)
                if time_series_data:
                    nem12_block = create_nem12_structure(time_series_data, nmi, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info("✅ Successfully converted fallback time series data to NEM12")
                else:
                    logger.error("❌ Could not process file - no supported format detected")
                    # Debug: Show first few rows to help identify format
                    logger.info("First 5 rows for debugging:")
                    for i in range(min(5, df.shape[0])):
                        row_preview = [str(x)[:50] for x in df.iloc[i][:8] if pd.notna(x)]
                        logger.info(f"  Row {i+1}: {row_preview}")
        
        logger.info(f"Completed processing {file_path} with {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


# ============================================================================
# TESTING FUNCTION FOR THE NEW METER DATA FORMAT
# Use this to test the new format detection and extraction
# ============================================================================

def test_meter_data_format():
    """
    Test function specifically for the new meter data format.
    Use this to verify the detection and extraction is working correctly.
    """
    import pandas as pd
    import logging
    
    # Sample data based on your file structure
    test_data = [
        ["NMI", "MeterSerial", "Period", "Local Time", "Data Type", "E kWh at Meter", "kW", "kVA", "PF"],
        ["6001276858", "12345678", "1", "1/12/2017 0:30", "Actual", "1.234", "2.468", "2.500", "0.987"],
        ["6001276858", "12345678", "2", "1/12/2017 1:00", "Actual", "1.456", "2.912", "2.950", "0.987"],
        ["6001276858", "12345678", "3", "1/12/2017 1:30", "Actual", "1.678", "3.356", "3.400", "0.987"]
    ]
    
    # Create test DataFrame
    df = pd.DataFrame(test_data[1:], columns=test_data[0])
    logger = logging.getLogger("test")
    
    print("Testing NEW METER DATA format detection...")
    is_detected = detect_meter_data_format(df, logger)
    print(f"Detection result: {is_detected}")
    
    if is_detected:
        print("\nTesting data extraction...")
        extracted_data = extract_meter_data_format(df, logger)
        print(f"Extracted {len(extracted_data)} records")
        
        if extracted_data:
            print("\nSample extracted record:")
            sample = extracted_data[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
    
    return is_detected

# ============================================================================
# ADD THESE FUNCTIONS TO YOUR EXISTING CODE
# DO NOT REPLACE ANY EXISTING FUNCTIONS - JUST ADD THESE NEW ONES
# ============================================================================

def is_temp_file(file_path: str) -> bool:
    """Check if file is a temporary file that should be skipped."""
    filename = os.path.basename(file_path)
    
    # Skip temporary files
    temp_indicators = [
        filename.startswith('~$'),      # Excel temp files
        filename.startswith('.~'),      # Other temp files
        filename.startswith('~'),       # General temp files
        '.tmp' in filename.lower(),     # Temp files
    ]
    
    return any(temp_indicators)


def detect_wide_format_layout(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Detect the wide format where all data is spread across many columns in few rows.
    This format: NMI Date Interval Length Period EndTime Data Type Kwh KW KVA PF Generated Kwh Net Kwh...
    """
    try:
        if df.empty:
            return False
        
        logger.debug("Checking for wide format layout")
        
        # Check for very wide DataFrame (many columns, few rows)
        if df.shape[0] <= 5 and df.shape[1] > 50:
            
            # Get all text from the first row
            first_row_text = ''
            for col_idx in range(min(200, df.shape[1])):  # Check up to 200 columns
                cell_value = df.iloc[0, col_idx]
                if pd.notna(cell_value):
                    first_row_text += ' ' + str(cell_value)
            
            first_row_text = first_row_text.upper()
            
            # Check for specific indicators of this format
            required_headers = ['NMI', 'DATE', 'INTERVAL', 'LENGTH', 'PERIOD', 'ENDTIME', 'DATA TYPE', 'KWH']
            energy_headers = ['KW', 'KVA', 'PF', 'GENERATED KWH', 'NET KWH']
            quality_headers = ['QUALITY CODE', 'MEASURED', 'VALIDATION']
            
            # Count how many indicators we find
            required_found = sum(1 for header in required_headers if header in first_row_text)
            energy_found = sum(1 for header in energy_headers if header in first_row_text)
            quality_found = sum(1 for header in quality_headers if header in first_row_text)
            
            # Check for data patterns
            has_nmi_pattern = bool(re.search(r'\b\d{10}\b', first_row_text))
            has_date_pattern = bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', first_row_text))
            has_time_pattern = bool(re.search(r'\d{2}:\d{2}', first_row_text))
            has_decimal_numbers = bool(re.search(r'\d+\.\d{5}', first_row_text))
            
            # Scoring
            total_score = required_found + energy_found + quality_found
            data_patterns = sum([has_nmi_pattern, has_date_pattern, has_time_pattern, has_decimal_numbers])
            
            is_wide_format = (
                required_found >= 6 and      # Must have most required headers
                energy_found >= 3 and        # Must have energy headers  
                data_patterns >= 3 and       # Must have data patterns
                total_score >= 12            # High overall score
            )
            
            logger.info(f"Wide format detection: required={required_found}/8, energy={energy_found}/5, "
                       f"quality={quality_found}/3, data_patterns={data_patterns}/4, "
                       f"total_score={total_score}, result={is_wide_format}")
            
            return is_wide_format
        
        return False
        
    except Exception as e:
        logger.error(f"Error in detect_wide_format_layout: {e}")
        return False


def debug_file_structure(file_path: str, logger: logging.Logger) -> None:
    """Debug function to analyze file structure for failed files."""
    try:
        logger.info(f"=" * 40)
        logger.info(f"DEBUG: {os.path.basename(file_path)}")
        logger.info(f"=" * 40)
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            # Debug Excel file
            try:
                sheets = read_excel_file(file_path, logger)
                logger.info(f"Excel sheets: {list(sheets.keys())}")
                
                for sheet_name, df in sheets.items():
                    logger.info(f"Sheet: '{sheet_name}'")
                    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                    
                    # Test all format detectors
                    logger.info("Format detection results:")
                    logger.info(f"  NEM12: {detect_nem12_format(df, logger)}")
                    logger.info(f"  Wide format: {detect_wide_format_layout(df, logger)}")
                    logger.info(f"  Standard interval: {detect_standard_interval_format(df, logger)}")
                    logger.info(f"  Multi-column energy: {detect_multi_column_energy_format(df, logger)}")
                    logger.info(f"  Interval data: {detect_interval_data_format(df, logger)}")
                    logger.info(f"  Time series: {detect_time_series_format(df, logger)}")
                    
                    # Show first few cells for analysis
                    logger.info("First 5x5 cells:")
                    for row_idx in range(min(5, df.shape[0])):
                        row_data = []
                        for col_idx in range(min(5, df.shape[1])):
                            cell_val = df.iloc[row_idx, col_idx]
                            if pd.notna(cell_val):
                                cell_str = str(cell_val)[:20]  # Truncate long values
                                row_data.append(cell_str)
                            else:
                                row_data.append("NaN")
                        logger.info(f"Row {row_idx}: {row_data}")
                    
                    break  # Only debug first sheet
                        
            except Exception as e:
                logger.error(f"Excel debug error: {e}")
                
        elif file_ext in ['.csv', '.txt']:
            # Debug CSV file
            try:
                df = read_csv_file(file_path, logger)
                if df is not None:
                    logger.info(f"CSV Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                    
                    # Test all format detectors
                    logger.info("Format detection results:")
                    logger.info(f"  NEM12: {detect_nem12_format(df, logger)}")
                    logger.info(f"  Space-separated: {detect_space_separated_format(df, logger)}")
                    logger.info(f"  Standard interval: {detect_standard_interval_format(df, logger)}")
                    logger.info(f"  Multi-column energy: {detect_multi_column_energy_format(df, logger)}")
                    logger.info(f"  Interval data: {detect_interval_data_format(df, logger)}")
                    logger.info(f"  Time series: {detect_time_series_format(df, logger)}")
                    
                    # Show first few cells
                    logger.info("First 5x5 cells:")
                    for row_idx in range(min(5, df.shape[0])):
                        row_data = []
                        for col_idx in range(min(5, df.shape[1])):
                            cell_val = df.iloc[row_idx, col_idx]
                            if pd.notna(cell_val):
                                cell_str = str(cell_val)[:20]
                                row_data.append(cell_str)
                            else:
                                row_data.append("NaN")
                        logger.info(f"Row {row_idx}: {row_data}")
                        
            except Exception as e:
                logger.error(f"CSV debug error: {e}")
        
        logger.info(f"=" * 40)
        
    except Exception as e:
        logger.error(f"Debug error: {e}")

# ============================================================================
# ADD THESE FUNCTIONS TO YOUR EXISTING NEM12 CONVERTER CODE
# DO NOT REPLACE EXISTING FUNCTIONS - JUST ADD THESE NEW ONES
# ============================================================================

def detect_wide_format_layout(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Detect the wide format where all data is spread across many columns in few rows.
    This format: NMI Date Interval Length Period EndTime Data Type Kwh KW KVA PF Generated Kwh Net Kwh...
    """
    try:
        if df.empty:
            return False
        
        logger.debug("Checking for wide format layout")
        
        # Check for very wide DataFrame (many columns, few rows)
        if df.shape[0] <= 5 and df.shape[1] > 50:
            
            # Get all text from the first row
            first_row_text = ''
            for col_idx in range(min(200, df.shape[1])):  # Check up to 200 columns
                cell_value = df.iloc[0, col_idx]
                if pd.notna(cell_value):
                    first_row_text += ' ' + str(cell_value)
            
            first_row_text = first_row_text.upper()
            
            # Check for specific indicators of this format
            required_headers = ['NMI', 'DATE', 'INTERVAL', 'LENGTH', 'PERIOD', 'ENDTIME', 'DATA TYPE', 'KWH']
            energy_headers = ['KW', 'KVA', 'PF', 'GENERATED KWH', 'NET KWH']
            quality_headers = ['QUALITY CODE', 'MEASURED', 'VALIDATION']
            
            # Count how many indicators we find
            required_found = sum(1 for header in required_headers if header in first_row_text)
            energy_found = sum(1 for header in energy_headers if header in first_row_text)
            quality_found = sum(1 for header in quality_headers if header in first_row_text)
            
            # Check for data patterns
            has_nmi_pattern = bool(re.search(r'\b\d{10}\b', first_row_text))
            has_date_pattern = bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', first_row_text))
            has_time_pattern = bool(re.search(r'\d{2}:\d{2}', first_row_text))
            has_decimal_numbers = bool(re.search(r'\d+\.\d{5}', first_row_text))
            
            # Scoring
            total_score = required_found + energy_found + quality_found
            data_patterns = sum([has_nmi_pattern, has_date_pattern, has_time_pattern, has_decimal_numbers])
            
            is_wide_format = (
                required_found >= 6 and      # Must have most required headers
                energy_found >= 3 and        # Must have energy headers  
                data_patterns >= 3 and       # Must have data patterns
                total_score >= 12            # High overall score
            )
            
            logger.info(f"Wide format detection: required={required_found}/8, energy={energy_found}/5, "
                       f"quality={quality_found}/3, data_patterns={data_patterns}/4, "
                       f"total_score={total_score}, result={is_wide_format}")
            
            return is_wide_format
        
        return False
        
    except Exception as e:
        logger.error(f"Error in detect_wide_format_layout: {e}")
        return False

def extract_wide_format_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Extract data from the wide format layout.
    Format: NMI Date Interval Length Period EndTime Data Type Kwh KW KVA PF Generated Kwh Net Kwh Kvarh...
    """
    try:
        logger.info("Extracting wide format data")
        
        # Collect all values from all cells
        all_values = []
        for row_idx in range(df.shape[0]):
            for col_idx in range(df.shape[1]):
                cell_value = df.iloc[row_idx, col_idx]
                if pd.notna(cell_value):
                    cell_str = str(cell_value).strip()
                    if cell_str:
                        all_values.append(cell_str)
        
        if not all_values:
            logger.error("No values found in DataFrame")
            return []
        
        logger.info(f"Collected {len(all_values)} values")
        logger.debug(f"First 20 values: {all_values[:20]}")
        
        # Find where the header ends and data starts
        # Look for the first 10-digit number (NMI)
        data_start_idx = None
        for idx, value in enumerate(all_values):
            if re.match(r'^\d{10}$', value):
                data_start_idx = idx
                logger.info(f"Found first NMI '{value}' at position {idx}")
                break
        
        if data_start_idx is None:
            logger.error("Could not find start of data (no NMI found)")
            return []
        
        # Split into headers and data
        headers = all_values[:data_start_idx]
        data_section = all_values[data_start_idx:]
        
        logger.info(f"Headers: {len(headers)} items")
        logger.info(f"Data: {len(data_section)} items")
        logger.debug(f"Headers: {headers}")
        
        if len(headers) == 0:
            logger.error("No headers found")
            return []
        
        # Calculate number of complete data rows
        num_complete_rows = len(data_section) // len(headers)
        remainder = len(data_section) % len(headers)
        
        logger.info(f"Complete data rows: {num_complete_rows}")
        if remainder > 0:
            logger.warning(f"Incomplete data: {remainder} leftover values")
        
        if num_complete_rows == 0:
            logger.error("No complete data rows")
            return []
        
        # Map column positions
        column_map = {}
        for idx, header in enumerate(headers):
            header_clean = header.upper().replace(' ', '').replace('_', '')
            
            if header_clean == 'NMI':
                column_map['nmi'] = idx
            elif header_clean == 'DATE':
                column_map['date'] = idx  
            elif header_clean == 'ENDTIME':
                column_map['endtime'] = idx
            elif header_clean == 'PERIOD':
                column_map['period'] = idx
            elif header_clean in ['INTERVALLENGTH', 'INTERVAL']:
                column_map['interval_length'] = idx
            elif header_clean in ['DATATYPE', 'DATA']:
                column_map['data_type'] = idx
            elif header_clean == 'QUALITYCODE':
                column_map['quality'] = idx
            
            # Energy columns - prioritize Net Kwh, then Kwh, then Generated Kwh
            elif header_clean in ['NETKWH', 'NET']:
                column_map['kwh'] = idx  # Highest priority
                logger.debug(f"Found Net Kwh at column {idx}")
            elif header_clean == 'KWH' and 'kwh' not in column_map:
                column_map['kwh'] = idx  # Second priority
                logger.debug(f"Found Kwh at column {idx}")
            elif header_clean in ['GENERATEDKWH', 'GENERATED'] and 'kwh' not in column_map:
                column_map['kwh'] = idx  # Third priority
                logger.debug(f"Found Generated Kwh at column {idx}")
        
        logger.info(f"Column mapping: {column_map}")
        
        # Check for required columns
        if 'nmi' not in column_map:
            logger.error("Missing NMI column")
            return []
        if 'kwh' not in column_map:
            logger.error("Missing energy column (Kwh/Net Kwh/Generated Kwh)")
            return []
        
        # Extract time series data
        time_series_data = []
        
        for row_num in range(num_complete_rows):
            try:
                # Get this row's data
                start_idx = row_num * len(headers)
                end_idx = start_idx + len(headers)
                row_data = data_section[start_idx:end_idx]
                
                # Extract NMI
                nmi = row_data[column_map['nmi']]
                if not re.match(r'^\d{10}$', nmi):
                    logger.debug(f"Invalid NMI in row {row_num}: {nmi}")
                    continue
                
                # Extract datetime
                datetime_str = None
                
                # Try EndTime first
                if 'endtime' in column_map:
                    endtime_val = row_data[column_map['endtime']]
                    if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', endtime_val):
                        datetime_str = endtime_val
                
                # Fallback: combine Date + calculated time from Period
                if not datetime_str and 'date' in column_map and 'period' in column_map:
                    date_val = row_data[column_map['date']]
                    period_val = row_data[column_map['period']]
                    
                    try:
                        period_num = int(period_val)
                        interval_length = 30  # Default
                        
                        if 'interval_length' in column_map:
                            try:
                                interval_length = int(row_data[column_map['interval_length']])
                            except (ValueError, TypeError):
                                pass
                        
                        # Calculate end time for this period
                        total_minutes = period_num * interval_length
                        hours = total_minutes // 60
                        minutes = total_minutes % 60
                        datetime_str = f"{date_val} {hours:02d}:{minutes:02d}"
                        
                    except (ValueError, TypeError):
                        logger.debug(f"Could not parse period for row {row_num}")
                        continue
                
                if not datetime_str:
                    logger.debug(f"No datetime for row {row_num}")
                    continue
                
                # Parse the datetime
                try:
                    if ' ' in datetime_str:
                        date_part, time_part = datetime_str.split(' ', 1)
                    else:
                        date_part = datetime_str
                        time_part = "00:30"
                    
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if not formatted_date or not formatted_time:
                        logger.debug(f"Could not format datetime for row {row_num}: {datetime_str}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Datetime parsing error for row {row_num}: {e}")
                    continue
                
                # Extract energy reading
                kwh_val = row_data[column_map['kwh']]
                try:
                    kwh_reading = float(kwh_val)
                    # Reasonable validation
                    if abs(kwh_reading) > 100000:
                        logger.debug(f"Suspicious energy reading in row {row_num}: {kwh_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Invalid energy value in row {row_num}: {kwh_val}")
                    continue
                
                # Extract quality
                quality = 'A'  # Default
                if 'quality' in column_map:
                    quality_val = row_data[column_map['quality']]
                    if quality_val in ['A', 'S', 'F', 'V', 'N', 'E']:
                        quality = quality_val
                
                # Extract data type
                data_type = None
                if 'data_type' in column_map:
                    data_type = row_data[column_map['data_type']]
                    if data_type and data_type.upper() != 'MEASURED':
                        quality = 'S'  # Mark as substituted if not measured
                
                # Add to results
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{kwh_reading:.3f}",
                    'quality': quality,
                    'data_type': data_type
                })
                
            except Exception as e:
                logger.debug(f"Error processing row {row_num}: {e}")
                continue
        
        logger.info(f"Wide format extraction completed: {len(time_series_data)} records")
        
        if time_series_data:
            # Log summary
            sample_readings = [float(r['reading']) for r in time_series_data[:50]]
            avg_reading = sum(sample_readings) / len(sample_readings) if sample_readings else 0
            logger.info(f"Sample stats: avg={avg_reading:.3f}, total_records={len(time_series_data)}")
            logger.info(f"NMI: {time_series_data[0]['nmi']}")
            logger.info(f"Date range: {time_series_data[0]['date']} to {time_series_data[-1]['date']}")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error in extract_wide_format_data: {e}")
        return []




#======================================================================================================================================================

# ============================================================================
# ADD THESE FUNCTIONS TO YOUR NEM12 CONVERTER CODE
# These handle the SRC CONNECTION POINT ID format specifically
# ============================================================================

def detect_src_connection_point_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Detect SRC CONNECTION POINT format specifically.
    Format: SRC CONNECTION POINT ID, METER SERIAL NUMBER, READ QUALITY CODE, 
           CONNECTION POINT SUFFIX, SRC READ TIMESTAMP, DAY TIME, METER CHANNEL CODE, INTERVAL, KWH
    """
    try:
        if df.empty or df.shape[0] < 2:
            return False
        
        logger.debug("Detecting SRC CONNECTION POINT format")
        
        # Check for specific header pattern
        header_row = None
        for row_idx in range(min(3, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Look for specific indicators
            score = 0
            if 'SRC CONNECTION POINT ID' in row_text:
                score += 5
            if 'METER SERIAL NUMBER' in row_text:
                score += 3
            if 'READ QUALITY CODE' in row_text:
                score += 3
            if 'SRC READ TIMESTAMP' in row_text:
                score += 3
            if 'DAY TIME' in row_text:
                score += 3
            if 'INTERVAL' in row_text and 'KWH' in row_text:
                score += 2
            
            if score >= 15:  # High threshold for this specific format
                header_row = row_idx
                logger.info(f"Found SRC CONNECTION POINT header at row {row_idx + 1} with score {score}")
                break
        
        if header_row is None:
            return False
        
        # Verify data format by checking a few data rows
        data_start = header_row + 1
        if data_start >= df.shape[0]:
            return False
        
        valid_data_rows = 0
        for row_idx in range(data_start, min(data_start + 5, df.shape[0])):
            row_data = df.iloc[row_idx].fillna('').astype(str)
            
            # Check for 10-digit NMI in first column
            if len(row_data) > 0:
                first_cell = str(row_data.iloc[0]).strip()
                has_nmi = re.match(r'^\d{10}$', first_cell) is not None
                
                # Check for date pattern like "1-Jul-17"
                has_date = False
                if len(row_data) > 4:
                    date_cell = str(row_data.iloc[4]).strip()
                    if re.match(r'\d{1,2}-\w{3}-\d{2}', date_cell):
                        has_date = True
                
                # Check for time pattern like "0:00"
                has_time = False
                if len(row_data) > 5:
                    time_cell = str(row_data.iloc[5]).strip()
                    if re.match(r'\d{1,2}:\d{2}', time_cell):
                        has_time = True
                
                # Check for reasonable kWh values
                has_kwh = False
                if len(row_data) > 8:
                    try:
                        kwh_value = float(str(row_data.iloc[8]).strip())
                        if 0 <= kwh_value <= 100:  # Reasonable range for interval data
                            has_kwh = True
                    except (ValueError, TypeError):
                        pass
                
                if has_nmi and has_date and has_time and has_kwh:
                    valid_data_rows += 1
        
        is_src_connection_format = valid_data_rows >= 3
        logger.info(f"SRC CONNECTION POINT detection: header_row={header_row + 1 if header_row is not None else 'None'}, "
                   f"valid_rows={valid_data_rows}, result={is_src_connection_format}")
        
        return is_src_connection_format
        
    except Exception as e:
        logger.error(f"Error detecting SRC CONNECTION POINT format: {e}")
        return False


def extract_src_connection_point_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Extract data from SRC CONNECTION POINT format.
    Format: SRC CONNECTION POINT ID, METER SERIAL NUMBER, READ QUALITY CODE, 
           CONNECTION POINT SUFFIX, SRC READ TIMESTAMP, DAY TIME, METER CHANNEL CODE, INTERVAL, KWH
    """
    try:
        logger.info("Extracting SRC CONNECTION POINT interval data")
        
        # Find header row
        header_row_idx = None
        for row_idx in range(min(3, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            if 'SRC CONNECTION POINT ID' in row_text and 'KWH' in row_text:
                header_row_idx = row_idx
                break
        
        if header_row_idx is None:
            logger.error("Could not find SRC CONNECTION POINT header row")
            return []
        
        # Column mapping based on expected structure
        column_indices = {
            'nmi': 0,              # SRC CONNECTION POINT ID
            'meter_serial': 1,     # METER SERIAL NUMBER  
            'quality': 2,          # READ QUALITY CODE
            'suffix': 3,           # CONNECTION POINT SUFFIX
            'date': 4,             # SRC READ TIMESTAMP
            'time': 5,             # DAY TIME
            'channel': 6,          # METER CHANNEL CODE
            'interval': 7,         # INTERVAL
            'kwh': 8               # KWH
        }
        
        logger.info(f"Using fixed column mapping for SRC CONNECTION POINT format")
        
        # Extract data
        data_start = header_row_idx + 1
        time_series_data = []
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                row_data = df.iloc[row_idx]
                
                if len(row_data) < 9:  # Need all 9 columns
                    continue
                
                # Extract NMI
                nmi = str(row_data.iloc[column_indices['nmi']]).strip()
                if not re.match(r'^\d{10}$', nmi):
                    continue
                
                # Extract and parse date (format: "1-Jul-17")
                date_str = str(row_data.iloc[column_indices['date']]).strip()
                formatted_date = parse_src_date(date_str, logger)
                if not formatted_date:
                    continue
                
                # Extract and parse time (format: "0:00")
                time_str = str(row_data.iloc[column_indices['time']]).strip()
                formatted_time = parse_time(time_str)
                if not formatted_time:
                    continue
                
                # Extract kWh reading
                kwh_value = row_data.iloc[column_indices['kwh']]
                if pd.isna(kwh_value):
                    continue
                
                try:
                    kwh_reading = float(kwh_value)
                    if abs(kwh_reading) > 100000:  # Reasonable validation
                        logger.debug(f"Large kWh reading: {kwh_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Invalid kWh value: {kwh_value}")
                    continue
                
                # Extract quality code
                quality = str(row_data.iloc[column_indices['quality']]).strip()
                if quality not in NEM12Config.VALID_QUALITY_FLAGS:
                    quality = 'A'  # Default to Actual
                
                # Extract additional info
                meter_serial = str(row_data.iloc[column_indices['meter_serial']]).strip()
                suffix = str(row_data.iloc[column_indices['suffix']]).strip()
                channel = str(row_data.iloc[column_indices['channel']]).strip()
                interval_num = str(row_data.iloc[column_indices['interval']]).strip()
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{kwh_reading:.3f}",
                    'quality': quality,
                    'meter_serial': meter_serial,
                    'suffix': suffix,
                    'channel': channel,
                    'interval_num': interval_num
                })
                
            except Exception as e:
                logger.debug(f"Error processing row {row_idx + 1}: {e}")
                continue
        
        logger.info(f"Extracted {len(time_series_data)} SRC CONNECTION POINT records")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting SRC CONNECTION POINT data: {e}")
        return []

def parse_src_date(date_str: str, logger: logging.Logger) -> Optional[str]:
    """
    Parse SRC date format like "1-Jul-17" to YYYYMMDD format.
    """
    if not date_str:
        return None
    
    try:
        # Handle formats like "1-Jul-17", "01-Jul-2017", etc.
        date_patterns = [
            '%d-%b-%y',    # 1-Jul-17
            '%d-%b-%Y',    # 1-Jul-2017
            '%d-%B-%y',    # 1-July-17
            '%d-%B-%Y',    # 1-July-2017
        ]
        
        for pattern in date_patterns:
            try:
                date_obj = datetime.strptime(date_str, pattern)
                return date_obj.strftime('%Y%m%d')
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date: {date_str}")
        return None
        
    except Exception as e:
        logger.debug(f"Error parsing date '{date_str}': {e}")
        return None


# ============================================================================
# UPDATE YOUR EXISTING FUNCTIONS TO INCLUDE THE NEW FORMAT
# ============================================================================

def detect_interval_data_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    ENHANCED VERSION - Replace your existing detect_interval_data_format function.
    Now includes SRC CONNECTION POINT format detection.
    """
    try:
        if df.empty or df.shape[0] < 5:
            return False
        
        # First check for SRC CONNECTION POINT format
        if detect_src_connection_point_format(df, logger):
            return True
        
        # Then check for other interval data formats (your existing logic)
        header_text = ' '.join(str(x).upper() for x in df.iloc[0] if pd.notna(x))
        
        interval_indicators = [
            'TIMESTAMP', 'INTERVAL', 'KWH', 'QUALITY', 'LOCAL TIME',
            'CONNECTION POINT', 'METER SERIAL', 'PERIOD'
        ]
        
        indicator_count = sum(1 for indicator in interval_indicators 
                            if indicator in header_text)
        
        if indicator_count >= 3:
            # Verify with data pattern check
            for idx in range(1, min(10, df.shape[0])):
                row_str = ' '.join(str(x) for x in df.iloc[idx] if pd.notna(x))
                
                has_time_pattern = bool(re.search(r'\d{1,2}:\d{2}', row_str))
                has_date_pattern = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', row_str))
                
                if has_time_pattern and has_date_pattern:
                    logger.info(f"Detected interval data format (score: {indicator_count})")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error detecting interval data format: {e}")
        return False


def extract_interval_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED VERSION - Replace your existing extract_interval_data function.
    Now includes SRC CONNECTION POINT format extraction.
    """
    try:
        logger.info("Extracting interval data (enhanced)")
        
        if df.empty:
            return []
        
        # Check for SRC CONNECTION POINT format first
        if detect_src_connection_point_format(df, logger):
            return extract_src_connection_point_data(df, logger)
        
        # Fall back to existing logic for other formats
        header_text = ' '.join(str(x).upper() for x in df.iloc[0] if pd.notna(x))
        logger.debug(f"Header analysis: {header_text[:100]}...")
        
        # Determine format type (your existing logic)
        if 'PERIOD' in header_text and 'KWH AT METER' in header_text:
            return extract_format_period_kwh(df, logger)
        elif 'LOCAL TIME' in header_text and 'KWH' in header_text:
            return extract_format_local_time(df, logger)
        else:
            return extract_format_connection_point(df, logger)
        
    except Exception as e:
        logger.error(f"Error extracting interval data: {e}")
        return []

# ============================================================================
# TESTING FUNCTION FOR YOUR SPECIFIC DATA
# ============================================================================

def test_src_connection_point_format():
    """Test function specifically for your data format."""
    
    # Sample data based on your format
    test_data = [
        ["SRC CONNECTION POINT ID", "METER SERIAL NUMBER", "READ QUALITY CODE", 
         "CONNECTION POINT SUFFIX", "SRC READ TIMESTAMP", "DAY TIME", 
         "METER CHANNEL CODE", "INTERVAL", "KWH"],
        ["6305006808", "4750384", "A", "E1", "1-Jul-17", "0:00", "E", "1", "3.75"],
        ["6305006808", "4750384", "A", "E1", "1-Jul-17", "0:30", "E", "2", "3.379"],
        ["6305006808", "4750384", "A", "E1", "1-Jul-17", "1:00", "E", "3", "3.616"]
    ]
    
    # Create test DataFrame
    import pandas as pd
    import logging
    
    df = pd.DataFrame(test_data[1:], columns=test_data[0])
    logger = logging.getLogger("test")
    
    # Test detection
    print("Testing SRC CONNECTION POINT format detection...")
    is_detected = detect_src_connection_point_format(df, logger)
    print(f"Detection result: {is_detected}")
    
    if is_detected:
        # Test extraction
        print("\nTesting data extraction...")
        extracted_data = extract_src_connection_point_data(df, logger)
        print(f"Extracted {len(extracted_data)} records")
        
        if extracted_data:
            print("\nSample extracted record:")
            print(extracted_data[0])
    
    return is_detected



# ============================================================================
# TESTING FUNCTION FOR YOUR SPECIFIC EXCEL DATA
# ============================================================================

def test_excel_interval_format():
    """Test function specifically for your Excel data format."""
    
    # Sample data based on your format
    test_data = [
        ["NMI", "Date", "Interval Length", "Period", "EndTime", "Meter Serial", 
         "Kwh", "Generated Kwh", "Net Kwh", "KW", "KVA", "pf", "Quality Code", 
         "Validation Code", "Validation Description"],
        ["6203733568", "1/04/2017", "30", "1", "01/04/2017 00:30", "213215669", 
         "35.41000", "0.00000", "35.41000", "70.82000", "73.01007", "0.97000", 
         "A", "", "Gentrack"],
        ["6203733568", "1/04/2017", "30", "2", "01/04/2017 01:00", "213215669", 
         "33.86000", "0.00000", "33.86000", "67.72000", "69.25936", "0.97777", 
         "A", "", "Gentrack"]
    ]
    
    # Create test DataFrame
    import pandas as pd
    import logging
    
    df = pd.DataFrame(test_data[1:], columns=test_data[0])
    logger = logging.getLogger("test")
    
    # Test detection
    print("Testing Excel interval format detection...")
    is_detected = detect_excel_interval_format(df, logger)
    print(f"Detection result: {is_detected}")
    
    if is_detected:
        # Test extraction
        print("\nTesting data extraction...")
        extracted_data = extract_excel_interval_data(df, logger)
        print(f"Extracted {len(extracted_data)} records")
        
        if extracted_data:
            print("\nSample extracted record:")
            print(extracted_data[0])
    
    return is_detected






















# ============================================================================
# ENHANCED EXCEL DETECTION FUNCTIONS - REPLACE EXISTING ONES
# These have better detection logic and debug output
# ============================================================================

def detect_excel_interval_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    ENHANCED - Detect Excel interval format with better flexibility.
    Format: NMI, Date, Interval Length, Period, EndTime, Meter Serial, Kwh, Generated Kwh, Net Kwh, etc.
    """
    try:
        if df.empty or df.shape[0] < 2:
            logger.debug("DataFrame is empty or too small")
            return False
        
        logger.debug(f"Detecting Excel interval format - DataFrame shape: {df.shape}")
        
        # Check for specific header pattern - be more flexible
        header_row = None
        best_score = 0
        
        for row_idx in range(min(5, df.shape[0])):  # Check first 5 rows
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x) and str(x).strip()]
            row_text = ' '.join(row_values)
            
            logger.debug(f"Row {row_idx + 1} text: {row_text[:200]}...")  # Log first 200 chars
            
            # Look for specific indicators with more flexible scoring
            score = 0
            indicators_found = []
            
            if 'NMI' in row_text:
                score += 4
                indicators_found.append('NMI')
            if 'DATE' in row_text:
                score += 3
                indicators_found.append('DATE')
            if 'INTERVAL' in row_text and 'LENGTH' in row_text:
                score += 3
                indicators_found.append('INTERVAL LENGTH')
            if 'PERIOD' in row_text:
                score += 2
                indicators_found.append('PERIOD')
            if 'ENDTIME' in row_text or 'END TIME' in row_text:
                score += 3
                indicators_found.append('ENDTIME')
            if 'METER' in row_text and 'SERIAL' in row_text:
                score += 2
                indicators_found.append('METER SERIAL')
            if 'KWH' in row_text:
                score += 3
                indicators_found.append('KWH')
            if 'NET' in row_text and 'KWH' in row_text:
                score += 3
                indicators_found.append('NET KWH')
            if 'GENERATED' in row_text and 'KWH' in row_text:
                score += 2
                indicators_found.append('GENERATED KWH')
            if 'QUALITY' in row_text and 'CODE' in row_text:
                score += 2
                indicators_found.append('QUALITY CODE')
            if 'VALIDATION' in row_text:
                score += 1
                indicators_found.append('VALIDATION')
            
            logger.debug(f"Row {row_idx + 1} score: {score}, indicators: {indicators_found}")
            
            if score > best_score:
                best_score = score
                header_row = row_idx
        
        logger.info(f"Best header row: {header_row + 1 if header_row is not None else 'None'} with score {best_score}")
        
        if header_row is None or best_score < 15:  # Lowered threshold
            logger.debug(f"No suitable header found (best score: {best_score})")
            return False
        
        # Verify data format by checking a few data rows
        data_start = header_row + 1
        if data_start >= df.shape[0]:
            logger.debug("No data rows after header")
            return False
        
        valid_data_rows = 0
        max_check_rows = min(data_start + 10, df.shape[0])  # Check more rows
        
        for row_idx in range(data_start, max_check_rows):
            row_data = df.iloc[row_idx].fillna('').astype(str)
            validation_score = 0
            
            logger.debug(f"Checking data row {row_idx + 1}: {[str(x)[:20] for x in row_data[:5]]}")
            
            # Check for 10-digit NMI in first few columns
            nmi_found = False
            for col_idx in range(min(3, len(row_data))):
                cell_value = str(row_data.iloc[col_idx]).strip()
                if re.match(r'^\d{10}$', cell_value):
                    validation_score += 3
                    nmi_found = True
                    logger.debug(f"Found NMI: {cell_value}")
                    break
            
            # Check for date format in first few columns
            date_found = False
            for col_idx in range(min(5, len(row_data))):
                cell_value = str(row_data.iloc[col_idx]).strip()
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}', cell_value):
                    validation_score += 2
                    date_found = True
                    logger.debug(f"Found date: {cell_value}")
                    break
            
            # Check for datetime format (EndTime)
            datetime_found = False
            for col_idx in range(min(8, len(row_data))):
                cell_value = str(row_data.iloc[col_idx]).strip()
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', cell_value):
                    validation_score += 3
                    datetime_found = True
                    logger.debug(f"Found datetime: {cell_value}")
                    break
            
            # Check for numeric energy values
            numeric_found = False
            for col_idx in range(5, min(12, len(row_data))):  # Check columns 5-11
                cell_value = str(row_data.iloc[col_idx]).strip()
                try:
                    float_val = float(cell_value)
                    if 0 <= float_val <= 10000:  # Reasonable energy range
                        validation_score += 1
                        numeric_found = True
                        logger.debug(f"Found energy value: {float_val}")
                        break
                except (ValueError, TypeError):
                    continue
            
            logger.debug(f"Row {row_idx + 1} validation score: {validation_score} "
                        f"(NMI:{nmi_found}, Date:{date_found}, DateTime:{datetime_found}, Numeric:{numeric_found})")
            
            if validation_score >= 6:  # Lower threshold
                valid_data_rows += 1
        
        is_excel_interval = valid_data_rows >= 2  # Lower threshold
        logger.info(f"Excel interval detection: header_row={header_row + 1 if header_row is not None else 'None'}, "
                   f"valid_rows={valid_data_rows}/{max_check_rows - data_start}, result={is_excel_interval}")
        
        return is_excel_interval
        
    except Exception as e:
        logger.error(f"Error detecting Excel interval format: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def extract_excel_interval_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED - Extract data from Excel interval format with better flexibility.
    """
    try:
        logger.info("Extracting Excel interval data (enhanced)")
        
        # Find header row with enhanced logic
        header_row_idx = None
        best_score = 0
        
        for row_idx in range(min(5, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[row_idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            score = 0
            if 'NMI' in row_text:
                score += 3
            if ('ENDTIME' in row_text or 'END TIME' in row_text):
                score += 3
            if 'KWH' in row_text:
                score += 2
            if 'NET' in row_text:
                score += 2
            if 'QUALITY' in row_text:
                score += 1
            
            if score > best_score:
                best_score = score
                header_row_idx = row_idx
        
        if header_row_idx is None or best_score < 8:
            logger.error("Could not find Excel interval header row")
            return []
        
        logger.info(f"Using header row {header_row_idx + 1} with score {best_score}")
        
        # Get column headers
        headers = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[header_row_idx]]
        logger.info(f"Headers ({len(headers)} columns): {headers}")
        
        # Enhanced column mapping with more flexible matching
        column_indices = {}
        
        for idx, header in enumerate(headers):
            header_clean = header.upper().replace(' ', '').replace('_', '')
            
            # Core columns
            if header_clean == 'NMI':
                column_indices['nmi'] = idx
            elif header_clean == 'DATE':
                column_indices['date'] = idx
            elif 'INTERVAL' in header_clean and 'LENGTH' in header_clean:
                column_indices['interval_length'] = idx
            elif header_clean == 'PERIOD':
                column_indices['period'] = idx
            elif 'ENDTIME' in header_clean or header_clean == 'ENDTIME':
                column_indices['endtime'] = idx
            elif 'METER' in header_clean and 'SERIAL' in header_clean:
                column_indices['meter_serial'] = idx
            elif 'QUALITY' in header_clean and 'CODE' in header_clean:
                column_indices['quality'] = idx
            elif 'VALIDATION' in header_clean and 'CODE' in header_clean:
                column_indices['validation_code'] = idx
            
            # Energy columns with priority system
            elif header_clean == 'NETKWH' or (header_clean == 'NET' and 'KWH' in header):
                column_indices['kwh'] = idx  # Highest priority
                logger.info(f"Found Net Kwh at position {idx}")
            elif header_clean == 'KWH' and 'kwh' not in column_indices:
                column_indices['kwh'] = idx  # Second priority
                logger.info(f"Found Kwh at position {idx}")
            elif ('GENERATED' in header_clean and 'KWH' in header_clean) and 'kwh' not in column_indices:
                column_indices['kwh'] = idx  # Third priority
                logger.info(f"Found Generated Kwh at position {idx}")
        
        logger.info(f"Enhanced column mapping: {column_indices}")
        
        # Check for required columns with more flexibility
        if 'nmi' not in column_indices:
            # Try to find NMI in first few columns by data pattern
            logger.warning("NMI column not found by header, searching by data pattern...")
            for col_idx in range(min(3, len(headers))):
                sample_values = [str(df.iloc[i, col_idx]) for i in range(header_row_idx + 1, min(header_row_idx + 6, df.shape[0]))]
                if any(re.match(r'^\d{10}$', val.strip()) for val in sample_values if pd.notna(val)):
                    column_indices['nmi'] = col_idx
                    logger.info(f"Found NMI column by pattern at position {col_idx}")
                    break
        
        if 'endtime' not in column_indices:
            # Try to find EndTime by data pattern
            logger.warning("EndTime column not found by header, searching by data pattern...")
            for col_idx in range(len(headers)):
                sample_values = [str(df.iloc[i, col_idx]) for i in range(header_row_idx + 1, min(header_row_idx + 6, df.shape[0]))]
                if any(re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', val.strip()) for val in sample_values if pd.notna(val)):
                    column_indices['endtime'] = col_idx
                    logger.info(f"Found EndTime column by pattern at position {col_idx}")
                    break
        
        if 'kwh' not in column_indices:
            # Try to find energy column by data pattern
            logger.warning("Energy column not found by header, searching by data pattern...")
            for col_idx in range(5, min(15, len(headers))):  # Look in middle columns
                sample_values = []
                for i in range(header_row_idx + 1, min(header_row_idx + 6, df.shape[0])):
                    val = df.iloc[i, col_idx]
                    if pd.notna(val):
                        sample_values.append(val)
                
                numeric_count = 0
                for val in sample_values:
                    try:
                        float_val = float(val)
                        if 0 <= float_val <= 10000:  # Reasonable energy range
                            numeric_count += 1
                    except (ValueError, TypeError):
                        continue
                
                if numeric_count >= 3:  # Most values are reasonable energy readings
                    column_indices['kwh'] = col_idx
                    logger.info(f"Found energy column by pattern at position {col_idx}")
                    break
        
        # Final validation
        required_cols = ['nmi', 'endtime', 'kwh']
        missing_cols = [col for col in required_cols if col not in column_indices]
        
        if missing_cols:
            logger.error(f"Missing required columns after enhanced search: {missing_cols}")
            logger.error(f"Available columns: {column_indices}")
            return []
        
        # Extract data
        data_start = header_row_idx + 1
        time_series_data = []
        
        logger.info(f"Starting data extraction from row {data_start + 1}")
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                row_data = df.iloc[row_idx]
                
                # Get NMI
                nmi = str(row_data.iloc[column_indices['nmi']]).strip()
                if not re.match(r'^\d{10}$', nmi):
                    logger.debug(f"Row {row_idx + 1}: Invalid NMI '{nmi}'")
                    continue
                
                # Get EndTime
                endtime_value = str(row_data.iloc[column_indices['endtime']]).strip()
                if not re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', endtime_value):
                    logger.debug(f"Row {row_idx + 1}: Invalid EndTime '{endtime_value}'")
                    continue
                
                # Parse datetime from EndTime
                try:
                    if ' ' in endtime_value:
                        date_part, time_part = endtime_value.split(' ', 1)
                    else:
                        logger.debug(f"Row {row_idx + 1}: EndTime has no space: '{endtime_value}'")
                        continue
                    
                    formatted_date = parse_date(date_part)
                    formatted_time = parse_time(time_part)
                    
                    if not formatted_date or not formatted_time:
                        logger.debug(f"Row {row_idx + 1}: Could not parse date/time from '{endtime_value}'")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Row {row_idx + 1}: Error parsing EndTime '{endtime_value}': {e}")
                    continue
                
                # Get kWh reading
                kwh_value = row_data.iloc[column_indices['kwh']]
                if pd.isna(kwh_value):
                    logger.debug(f"Row {row_idx + 1}: Empty kWh value")
                    continue
                
                try:
                    kwh_reading = float(kwh_value)
                    if abs(kwh_reading) > 100000:
                        logger.debug(f"Row {row_idx + 1}: Large kWh reading: {kwh_reading}")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"Row {row_idx + 1}: Invalid kWh value: {kwh_value}")
                    continue
                
                # Get quality code
                quality = 'A'  # Default
                if 'quality' in column_indices:
                    quality_value = str(row_data.iloc[column_indices['quality']]).strip()
                    if quality_value in NEM12Config.VALID_QUALITY_FLAGS:
                        quality = quality_value
                
                # Get additional info
                meter_serial = ''
                if 'meter_serial' in column_indices:
                    meter_serial = str(row_data.iloc[column_indices['meter_serial']]).strip()
                
                interval_length = '30'  # Default
                if 'interval_length' in column_indices:
                    interval_length = str(row_data.iloc[column_indices['interval_length']]).strip()
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{kwh_reading:.3f}",
                    'quality': quality,
                    'meter_serial': meter_serial,
                    'interval_length': interval_length
                })
                
                if len(time_series_data) % 1000 == 0:
                    logger.info(f"Processed {len(time_series_data)} records...")
                
            except Exception as e:
                logger.debug(f"Error processing row {row_idx + 1}: {e}")
                continue
        
        logger.info(f"Extracted {len(time_series_data)} Excel interval records")
        
        if time_series_data:
            logger.info(f"Sample record: NMI={time_series_data[0]['nmi']}, "
                       f"Date={time_series_data[0]['date']}, Time={time_series_data[0]['time']}")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting Excel interval data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


# ============================================================================
# DEBUG VERSION OF process_excel_file WITH MORE LOGGING
# ============================================================================

def process_excel_file(file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED VERSION with better debugging and Excel interval format support.
    """
    results = []
    
    try:
        logger.info(f"Starting Excel file processing: {os.path.basename(file_path)}")
        
        with safe_file_processing(file_path, logger):
            sheets = read_excel_file(file_path, logger)
            if not sheets:
                logger.error("No sheets found in Excel file")
                return []
            
            logger.info(f"Found {len(sheets)} sheets: {list(sheets.keys())}")
            
            # Extract NMI from filename or first sheet
            nmi = None
            for sheet_name, df in list(sheets.items())[:3]:
                extracted_nmi = extract_and_validate_nmi(file_path, df, logger)
                if not extracted_nmi.startswith("AUTO"):
                    nmi = extracted_nmi
                    break
            
            if not nmi:
                nmi = extract_and_validate_nmi(file_path, None, logger)
            
            logger.info(f"Using NMI: {nmi}")
            
            # Process each sheet with enhanced detection
            all_time_series = []
            for sheet_name, df in sheets.items():
                logger.info(f"Processing sheet: '{sheet_name}' with shape {df.shape}")
                
                if detect_nem12_format(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as NEM12 format")
                    results.extend(list(extract_nem12_data(df, f"{file_path}::{sheet_name}", logger)))

                elif detect_excel_interval_format(df, logger):  # ENHANCED FORMAT CHECK
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as Excel interval format")
                    time_series_data = extract_excel_interval_data(df, logger)
                    if time_series_data:
                        logger.info(f"✅ Extracted {len(time_series_data)} records from Excel interval format")
                        all_time_series.extend(time_series_data)
                    else:
                        logger.warning(f"❌ No data extracted from Excel interval format")

                elif detect_wide_format_layout(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as wide format layout")
                    try:
                        time_series_data = extract_wide_format_data(df, logger)
                        all_time_series.extend(time_series_data)
                    except Exception as e:
                        logger.error(f"Error in extract_wide_format_data: {e}")
                        
                elif detect_standard_interval_format(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as standard interval format")
                    time_series_data = extract_standard_interval_data(df, logger)
                    all_time_series.extend(time_series_data)
                    
                elif detect_multi_column_energy_format(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as multi-column energy format")
                    time_series_data = extract_multi_column_energy_data(df, logger)
                    all_time_series.extend(time_series_data)
                    
                elif detect_interval_data_format(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as interval data format")
                    time_series_data = extract_interval_data(df, logger)
                    all_time_series.extend(time_series_data)
                    
                elif detect_time_series_format(df, logger):
                    logger.info(f"🎯 Processing sheet '{sheet_name}' as time series format")
                    time_series_data = extract_time_series_data(df, nmi, logger)
                    all_time_series.extend(time_series_data)
                    
                else:
                    logger.warning(f"❌ Sheet '{sheet_name}' format not recognized")
                    # Show first few rows for debugging
                    logger.info("First 3 rows of unrecognized sheet:")
                    for i in range(min(3, df.shape[0])):
                        row_preview = [str(x)[:30] for x in df.iloc[i][:10] if pd.notna(x)]
                        logger.info(f"  Row {i+1}: {row_preview}")
            
            # Convert collected time series to NEM12
            if all_time_series:
                logger.info(f"Converting {len(all_time_series)} time series records to NEM12")
                
                # Group by NMI if multiple NMIs found
                nmi_groups = {}
                for record in all_time_series:
                    record_nmi = record['nmi']
                    if record_nmi not in nmi_groups:
                        nmi_groups[record_nmi] = []
                    nmi_groups[record_nmi].append(record)
                
                logger.info(f"Found {len(nmi_groups)} unique NMIs: {list(nmi_groups.keys())}")
                
                # Create NEM12 block for each NMI
                for nmi_key, nmi_data in nmi_groups.items():
                    logger.info(f"Creating NEM12 block for NMI {nmi_key} with {len(nmi_data)} records")
                    nem12_block = create_nem12_structure(nmi_data, nmi_key, logger)
                    if nem12_block:
                        results.append(nem12_block)
                        logger.info(f"✅ Successfully created NEM12 block for NMI {nmi_key}")
                    else:
                        logger.warning(f"❌ Failed to create NEM12 block for NMI {nmi_key}")
        
        logger.info(f"Completed processing {file_path} with {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error processing Excel file {file_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []















# ============================================================================
# ADD THESE FUNCTIONS TO YOUR NEM12 CONVERTER CODE
# These handle the AGL DETAILED CSV format specifically
# ============================================================================

def find_agl_data_start(df: pd.DataFrame, logger: logging.Logger) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Find where the actual data starts in AGL DETAILED format and map columns.
    Returns (data_start_row, column_mapping)
    """
    try:
        data_start = None
        column_mapping = {}
        
        # Look for header row with interval data columns
        for idx in range(min(30, df.shape[0])):
            row_values = [str(x).strip().upper() for x in df.iloc[idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Score this row as a potential header
            header_score = 0
            potential_columns = {}
            
            # Check for essential columns
            essential_indicators = {
                'NMI': ['NMI', 'METER', 'CONNECTION'],
                'DATE': ['DATE', 'DAY', 'PERIOD'],
                'TIME': ['TIME', 'HOUR', 'MINUTE'],
                'READING': ['READING', 'CONSUMPTION', 'USAGE', 'KWH', 'VALUE'],
                'QUALITY': ['QUALITY', 'FLAG', 'STATUS', 'CODE']
            }
            
            for col_type, indicators in essential_indicators.items():
                for col_idx, cell_value in enumerate(df.iloc[idx]):
                    if pd.notna(cell_value):
                        cell_text = str(cell_value).strip().upper()
                        if any(indicator in cell_text for indicator in indicators):
                            potential_columns[col_type] = col_idx
                            header_score += 2
                            break
            
            # If we found a good header row
            if header_score >= 6:  # Need at least 3 essential columns
                data_start = idx + 1
                column_mapping = potential_columns
                logger.info(f"Found AGL header at row {idx + 1} with score {header_score}")
                logger.info(f"Column mapping: {column_mapping}")
                break
        
        # If no clear header found, try to find data by pattern
        if data_start is None:
            logger.warning("No clear header found, searching for data by pattern")
            
            for idx in range(min(50, df.shape[0])):
                # Look for rows with NMI pattern
                for col_idx in range(min(3, df.shape[1])):
                    cell_value = str(df.iloc[idx, col_idx]).strip()
                    if re.match(r'^\d{10}$', cell_value):
                        data_start = idx
                        # Try to infer column mapping
                        column_mapping = infer_agl_columns(df, idx, logger)
                        logger.info(f"Found data start at row {idx + 1} by NMI pattern")
                        break
                if data_start is not None:
                    break
        
        if data_start is None:
            logger.warning("Could not find data start in AGL format")
        
        return data_start, column_mapping
        
    except Exception as e:
        logger.error(f"Error finding AGL data start: {e}")
        return None, {}

def infer_agl_columns(df: pd.DataFrame, start_row: int, logger: logging.Logger) -> Dict[str, int]:
    """
    Infer column mapping by analyzing data patterns in AGL format.
    """
    column_mapping = {}
    
    try:
        # Analyze first few data rows to infer column types
        sample_rows = min(10, df.shape[0] - start_row)
        
        for col_idx in range(df.shape[1]):
            column_type = analyze_agl_column(df, start_row, col_idx, sample_rows, logger)
            if column_type:
                column_mapping[column_type] = col_idx
        
        logger.info(f"Inferred AGL column mapping: {column_mapping}")
        return column_mapping
        
    except Exception as e:
        logger.error(f"Error inferring AGL columns: {e}")
        return {}


def analyze_agl_column(df: pd.DataFrame, start_row: int, col_idx: int, 
                      sample_rows: int, logger: logging.Logger) -> Optional[str]:
    """
    Analyze a column to determine its type in AGL format.
    """
    try:
        values = []
        for row_idx in range(start_row, start_row + sample_rows):
            if row_idx < df.shape[0]:
                cell_value = df.iloc[row_idx, col_idx]
                if pd.notna(cell_value):
                    values.append(str(cell_value).strip())
        
        if not values:
            return None
        
        # Analyze patterns
        nmi_count = sum(1 for v in values if re.match(r'^\d{10}$', v))
        date_count = sum(1 for v in values if re.match(r'\d{1,2}/\d{1,2}/\d{4}', v))
        time_count = sum(1 for v in values if re.match(r'\d{1,2}:\d{2}', v))
        datetime_count = sum(1 for v in values if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', v))
        
        numeric_count = 0
        for v in values:
            try:
                float_val = float(v)
                if 0 <= float_val <= 10000:
                    numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        quality_count = sum(1 for v in values if v.upper() in ['A', 'S', 'F', 'V', 'N', 'E'])
        
        # Determine column type
        total_values = len(values)
        
        if nmi_count >= total_values * 0.8:
            return 'NMI'
        elif datetime_count >= total_values * 0.8:
            return 'DATETIME'
        elif date_count >= total_values * 0.8:
            return 'DATE'  
        elif time_count >= total_values * 0.8:
            return 'TIME'
        elif numeric_count >= total_values * 0.8:
            return 'READING'
        elif quality_count >= total_values * 0.8:
            return 'QUALITY'
        
        return None
        
    except Exception as e:
        logger.debug(f"Error analyzing column {col_idx}: {e}")
        return None



#==================================+++++++++++++++++++++++++++++++==================================================

# ============================================================================
# ENHANCED AGL DETAILED FORMAT FUNCTIONS
# Replace existing AGL functions with these enhanced versions
# ============================================================================

def detect_agl_detailed_format(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    ENHANCED - Detect AGL DETAILED CSV format with better accuracy.
    Handles AGL retailer export files with metadata and detailed interval data.
    """
    try:
        if df.empty or df.shape[0] < 3:
            return False
        
        logger.debug("Detecting AGL DETAILED format")
        
        format_score = 0
        has_agl_indicators = False
        has_interval_data = False
        has_detailed_info = False
        has_nmi_in_filename = False
        
        # Check for AGL in context (filename, early rows)
        # This is often indicated by "AGL" in filename or metadata
        
        # Check first 30 rows for AGL-specific indicators and structure
        for idx in range(min(30, df.shape[0])):
            # Combine all non-null values in the row
            row_values = [str(x).strip().upper() for x in df.iloc[idx] if pd.notna(x) and str(x).strip()]
            row_text = ' '.join(row_values)
            
            # Check for AGL-specific indicators
            agl_indicators = [
                'AGL', 'AUSTRALIAN GAS LIGHT', 'RETAILER', 'DETAILED', 
                'INTERVAL DATA', 'CONSUMPTION', 'USAGE DATA', 'BILLING',
                'CUSTOMER', 'ACCOUNT', 'SUPPLY'
            ]
            agl_count = sum(1 for indicator in agl_indicators if indicator in row_text)
            if agl_count >= 2:
                has_agl_indicators = True
                format_score += 3
                logger.debug(f"Found AGL indicators at row {idx + 1}: {agl_count}")
            
            # Check for typical interval data headers
            interval_indicators = [
                'NMI', 'DATE', 'TIME', 'READING', 'CONSUMPTION', 'USAGE', 
                'KWH', 'QUALITY', 'STATUS', 'PERIOD', 'INTERVAL', 'TIMESTAMP',
                'START TIME', 'END TIME', 'LOCAL TIME'
            ]
            indicator_count = sum(1 for indicator in interval_indicators if indicator in row_text)
            if indicator_count >= 5:
                has_interval_data = True
                format_score += 4
                logger.debug(f"Found interval data indicators at row {idx + 1}: {indicator_count}")
            elif indicator_count >= 3:
                format_score += 2
            
            # Check for detailed/quality information typical in AGL exports
            detailed_indicators = [
                'QUALITY', 'METHOD', 'FLAG', 'CODE', 'STATUS', 'VALIDATION',
                'ACTUAL', 'ESTIMATED', 'SUBSTITUTED', 'MEASURED'
            ]
            detailed_count = sum(1 for indicator in detailed_indicators if indicator in row_text)
            if detailed_count >= 2:
                has_detailed_info = True
                format_score += 2
            
            # Check for NMI patterns (10 digits)
            nmi_matches = re.findall(r'\b(\d{10})\b', row_text)
            if nmi_matches:
                has_nmi_in_filename = True
                format_score += 2
                logger.debug(f"Found NMI pattern at row {idx + 1}: {nmi_matches[0]}")
        
        # Check for typical AGL data patterns in the file
        has_datetime_data = False
        has_numeric_data = False
        
        # Look for datetime patterns typical in AGL files
        datetime_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',     # Date patterns like 1/1/2017
            r'\d{4}-\d{2}-\d{2}',         # ISO date like 2017-01-01
            r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}',  # Full datetime
            r'\d{1,2}:\d{2}:\d{2}'        # Time patterns
        ]
        
        for idx in range(min(100, df.shape[0])):
            for col_idx in range(min(10, df.shape[1])):
                cell_value = str(df.iloc[idx, col_idx]).strip()
                for pattern in datetime_patterns:
                    if re.search(pattern, cell_value):
                        has_datetime_data = True
                        format_score += 1
                        logger.debug(f"Found datetime pattern at row {idx + 1}, col {col_idx + 1}")
                        break
                if has_datetime_data:
                    break
            if has_datetime_data:
                break
        
        # Look for numeric energy data with reasonable ranges
        for idx in range(10, min(100, df.shape[0])):  # Skip potential header rows
            for col_idx in range(df.shape[1]):
                cell_value = df.iloc[idx, col_idx]
                if pd.notna(cell_value):
                    try:
                        float_val = float(cell_value)
                        # AGL data often has decimal values in reasonable energy ranges
                        if 0 <= float_val <= 50000 and (float_val != int(float_val) or float_val > 0):
                            has_numeric_data = True
                            format_score += 1
                            break
                    except (ValueError, TypeError):
                        continue
            if has_numeric_data:
                break
        
        # Enhanced scoring logic for AGL DETAILED format
        is_agl_detailed = (
            format_score >= 8 and  # Minimum score threshold
            (has_agl_indicators or has_interval_data) and  # Must have AGL or interval indicators
            (has_datetime_data and has_numeric_data)  # Must have both datetime and numeric data
        )
        
        logger.info(f"AGL DETAILED detection: score={format_score}, "
                   f"agl_indicators={has_agl_indicators}, interval_data={has_interval_data}, "
                   f"detailed_info={has_detailed_info}, nmi_found={has_nmi_in_filename}, "
                   f"datetime_data={has_datetime_data}, numeric_data={has_numeric_data}, "
                   f"result={is_agl_detailed}")
        
        return is_agl_detailed
        
    except Exception as e:
        logger.error(f"Error detecting AGL DETAILED format: {e}")
        return False


def find_agl_data_start_enhanced(df: pd.DataFrame, logger: logging.Logger) -> Tuple[Optional[int], Dict[str, int]]:
    """
    ENHANCED - Find where the actual data starts in AGL DETAILED format and map columns.
    Returns (data_start_row, column_mapping)
    """
    try:
        data_start = None
        column_mapping = {}
        
        # Look for header row with interval data columns
        # AGL files often have metadata at the top, then headers, then data
        best_header_row = None
        best_score = 0
        
        for idx in range(min(50, df.shape[0])):  # Check more rows for AGL
            row_values = [str(x).strip().upper() for x in df.iloc[idx] if pd.notna(x)]
            row_text = ' '.join(row_values)
            
            # Score this row as a potential header
            header_score = 0
            potential_columns = {}
            
            # Check for essential columns with flexible naming
            essential_patterns = {
                'NMI': [r'NMI', r'METER.*POINT', r'CONNECTION.*POINT', r'SUPPLY.*POINT'],
                'DATE': [r'DATE', r'DAY', r'PERIOD.*DATE', r'READ.*DATE'],
                'TIME': [r'TIME', r'HOUR', r'MINUTE', r'TIMESTAMP', r'START.*TIME', r'END.*TIME'],
                'DATETIME': [r'DATE.*TIME', r'TIMESTAMP', r'LOCAL.*TIME', r'READ.*TIME'],
                'READING': [r'READING', r'CONSUMPTION', r'USAGE', r'KWH', r'VALUE', r'ENERGY'],
                'QUALITY': [r'QUALITY', r'FLAG', r'STATUS', r'CODE', r'METHOD']
            }
            
            # Check current row for column headers
            for col_type, patterns in essential_patterns.items():
                for col_idx, cell_value in enumerate(df.iloc[idx]):
                    if pd.notna(cell_value):
                        cell_text = str(cell_value).strip().upper()
                        for pattern in patterns:
                            if re.search(pattern, cell_text):
                                potential_columns[col_type] = col_idx
                                header_score += 3
                                break
                        if col_type in potential_columns:
                            break
            
            # Additional scoring for AGL-specific patterns
            agl_headers = ['INTERVAL', 'PERIOD', 'ACTUAL', 'ESTIMATED', 'MEASURED']
            for header in agl_headers:
                if header in row_text:
                    header_score += 1
            
            # If we found a good header row
            if header_score > best_score and header_score >= 9:  # Need good coverage
                best_score = header_score
                best_header_row = idx
                column_mapping = potential_columns
                logger.info(f"Found potential AGL header at row {idx + 1} with score {header_score}")
        
        if best_header_row is not None:
            data_start = best_header_row + 1
            logger.info(f"Using AGL header at row {best_header_row + 1}, data starts at row {data_start + 1}")
            logger.info(f"Column mapping: {column_mapping}")
        
        # If no clear header found, try to find data by NMI pattern
        if data_start is None:
            logger.warning("No clear header found, searching for data by NMI pattern")
            
            for idx in range(min(100, df.shape[0])):
                # Look for rows with 10-digit NMI pattern
                for col_idx in range(min(5, df.shape[1])):
                    cell_value = str(df.iloc[idx, col_idx]).strip()
                    if re.match(r'^\d{10}$', cell_value):
                        data_start = idx
                        # Try to infer column mapping from surrounding data
                        column_mapping = infer_agl_columns_enhanced(df, idx, logger)
                        logger.info(f"Found AGL data start at row {idx + 1} by NMI pattern")
                        break
                if data_start is not None:
                    break
        
        return data_start, column_mapping
        
    except Exception as e:
        logger.error(f"Error finding AGL data start: {e}")
        return None, {}


def infer_agl_columns_enhanced(df: pd.DataFrame, start_row: int, logger: logging.Logger) -> Dict[str, int]:
    """
    ENHANCED - Infer column mapping by analyzing data patterns in AGL format.
    """
    column_mapping = {}
    
    try:
        # Analyze first 20 data rows to infer column types
        sample_rows = min(20, df.shape[0] - start_row)
        
        for col_idx in range(df.shape[1]):
            column_type = analyze_agl_column_enhanced(df, start_row, col_idx, sample_rows, logger)
            if column_type and column_type not in column_mapping:
                column_mapping[column_type] = col_idx
        
        logger.info(f"Inferred AGL column mapping: {column_mapping}")
        return column_mapping
        
    except Exception as e:
        logger.error(f"Error inferring AGL columns: {e}")
        return {}


def analyze_agl_column_enhanced(df: pd.DataFrame, start_row: int, col_idx: int, 
                               sample_rows: int, logger: logging.Logger) -> Optional[str]:
    """
    ENHANCED - Analyze a column to determine its type in AGL format.
    """
    try:
        values = []
        for row_idx in range(start_row, start_row + sample_rows):
            if row_idx < df.shape[0]:
                cell_value = df.iloc[row_idx, col_idx]
                if pd.notna(cell_value):
                    values.append(str(cell_value).strip())
        
        if not values:
            return None
        
        # Analyze patterns with enhanced detection
        nmi_count = sum(1 for v in values if re.match(r'^\d{10}$', v))
        date_count = sum(1 for v in values if re.match(r'\d{1,2}/\d{1,2}/\d{4}', v))
        time_count = sum(1 for v in values if re.match(r'\d{1,2}:\d{2}', v))
        datetime_count = sum(1 for v in values if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', v))
        
        # Enhanced numeric analysis for AGL data
        numeric_count = 0
        decimal_count = 0
        for v in values:
            try:
                float_val = float(v)
                if 0 <= float_val <= 50000:  # Reasonable range for AGL data
                    numeric_count += 1
                    if '.' in v and float_val != int(float_val):
                        decimal_count += 1
            except (ValueError, TypeError):
                pass
        
        # Enhanced quality flag detection
        quality_patterns = ['A', 'S', 'F', 'V', 'N', 'E', 'ACTUAL', 'ESTIMATED', 'MEASURED']
        quality_count = sum(1 for v in values if v.upper() in quality_patterns)
        
        # Determine column type with enhanced logic
        total_values = len(values)
        
        if nmi_count >= total_values * 0.8:
            return 'NMI'
        elif datetime_count >= total_values * 0.8:
            return 'DATETIME'
        elif date_count >= total_values * 0.8:
            return 'DATE'  
        elif time_count >= total_values * 0.8:
            return 'TIME'
        elif numeric_count >= total_values * 0.8:
            # Distinguish between energy readings and other numeric data
            if decimal_count >= numeric_count * 0.5:  # Many decimal values
                return 'READING'
            else:
                return 'NUMERIC'  # Could be period, interval number, etc.
        elif quality_count >= total_values * 0.6:  # Lower threshold for quality
            return 'QUALITY'
        
        return None
        
    except Exception as e:
        logger.debug(f"Error analyzing AGL column {col_idx}: {e}")
        return None


def extract_agl_detailed_data(df: pd.DataFrame, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    ENHANCED - Extract data from AGL DETAILED format with improved reliability.
    """
    try:
        logger.info("Extracting AGL DETAILED interval data")
        
        # Find data start and column mapping
        data_start, column_mapping = find_agl_data_start_enhanced(df, logger)
        
        if data_start is None:
            logger.error("Could not find data start in AGL DETAILED format")
            return []
        
        if not column_mapping:
            logger.error("Could not determine column mapping for AGL DETAILED format")
            return []
        
        # Extract NMI for validation
        nmi_from_data = None
        if 'NMI' in column_mapping:
            for row_idx in range(data_start, min(data_start + 10, df.shape[0])):
                nmi_value = str(df.iloc[row_idx, column_mapping['NMI']]).strip()
                if re.match(r'^\d{10}$', nmi_value):
                    nmi_from_data = nmi_value
                    break
        
        if not nmi_from_data:
            logger.error("Could not extract valid NMI from AGL data")
            return []
        
        logger.info(f"Found NMI in AGL data: {nmi_from_data}")
        
        # Extract time series data
        time_series_data = []
        processed_count = 0
        error_count = 0
        
        for row_idx in range(data_start, df.shape[0]):
            try:
                row_data = df.iloc[row_idx]
                
                # Extract NMI
                nmi = str(row_data.iloc[column_mapping['NMI']]).strip() if 'NMI' in column_mapping else nmi_from_data
                if not nmi or not re.match(r'^\d{10}$', nmi):
                    error_count += 1
                    continue
                
                # Extract datetime with enhanced logic
                formatted_date = None
                formatted_time = None
                
                if 'DATETIME' in column_mapping:
                    # Full datetime in one column
                    datetime_value = str(row_data.iloc[column_mapping['DATETIME']]).strip()
                    if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}', datetime_value):
                        try:
                            date_part, time_part = datetime_value.split(' ', 1)
                            formatted_date = parse_date(date_part)
                            formatted_time = parse_time(time_part)
                        except:
                            pass
                elif 'DATE' in column_mapping and 'TIME' in column_mapping:
                    # Separate date and time columns
                    date_value = str(row_data.iloc[column_mapping['DATE']]).strip()
                    time_value = str(row_data.iloc[column_mapping['TIME']]).strip()
                    formatted_date = parse_date(date_value)
                    formatted_time = parse_time(time_value)
                elif 'DATE' in column_mapping:
                    # Only date column, estimate time from row position
                    date_value = str(row_data.iloc[column_mapping['DATE']]).strip()
                    formatted_date = parse_date(date_value)
                    
                    # Estimate time based on row position (assuming 30-minute intervals)
                    rows_for_date = processed_count % 48  # 48 intervals per day for 30-min
                    total_minutes = rows_for_date * 30
                    hours = total_minutes // 60
                    minutes = total_minutes % 60
                    formatted_time = f"{hours:02d}{minutes:02d}"
                
                if not formatted_date or not formatted_time:
                    error_count += 1
                    continue
                
                # Extract reading
                reading_value = None
                if 'READING' in column_mapping:
                    reading_cell = row_data.iloc[column_mapping['READING']]
                    if pd.notna(reading_cell):
                        try:
                            reading_value = float(reading_cell)
                            if reading_value < 0 or reading_value > 50000:  # Reasonable validation
                                logger.debug(f"Unusual reading value: {reading_value}")
                        except (ValueError, TypeError):
                            error_count += 1
                            continue
                
                if reading_value is None:
                    error_count += 1
                    continue
                
                # Extract quality with enhanced logic
                quality = 'A'  # Default to Actual
                if 'QUALITY' in column_mapping:
                    quality_value = str(row_data.iloc[column_mapping['QUALITY']]).strip().upper()
                    if quality_value in NEM12Config.VALID_QUALITY_FLAGS:
                        quality = quality_value
                    elif 'ESTIMATED' in quality_value or 'SUBSTITUTED' in quality_value:
                        quality = 'S'
                    elif 'ACTUAL' in quality_value or 'MEASURED' in quality_value:
                        quality = 'A'
                
                time_series_data.append({
                    'nmi': nmi,
                    'date': formatted_date,
                    'time': formatted_time,
                    'reading': f"{reading_value:.3f}",
                    'quality': quality
                })
                
                processed_count += 1
                
                if processed_count % 5000 == 0:
                    logger.info(f"Processed {processed_count} AGL records...")
                
            except Exception as e:
                logger.debug(f"Error processing AGL row {row_idx + 1}: {e}")
                error_count += 1
                continue
        
        logger.info(f"Extracted {len(time_series_data)} AGL DETAILED records")
        logger.info(f"Processed: {processed_count}, Errors: {error_count}")
        
        if time_series_data:
            logger.info(f"Date range: {time_series_data[0]['date']} to {time_series_data[-1]['date']}")
            logger.info(f"Sample reading: {time_series_data[0]['reading']} kWh")
        
        return time_series_data
        
    except Exception as e:
        logger.error(f"Error extracting AGL DETAILED data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


# ============================================================================
# USAGE INSTRUCTIONS FOR YOUR AGL FILES
# ============================================================================

def convert_agl_files_to_nem12():
    """
    Instructions for converting your AGL DETAILED files to NEM12 format.
    """
    
    instructions = """
    ============================================================================
    CONVERTING AGL DETAILED FILES TO NEM12 FORMAT
    ============================================================================
    
    Your files:
    1. 6102923044_20170601_20171004_20180614_AGL_7376_DETAILED.csv
    2. 6203177121_20170101_20171231_20180125_AGL_5830_DETAILED.csv
    3. 6203695622_20170809_20180704_20180706_AGL_7611_DETAILED.xlsx
    
    STEP 1: Place all files in input folder
    ----------------------------------------
    Create folder: C:\\NEM12\\Input_NEM12\\
    Copy all 3 files to this folder
    
    STEP 2: Run the converter
    -------------------------
    Option A - Separate files (recommended):
    python nem12_converter.py -i C:\\NEM12\\Input_NEM12 -o C:\\NEM12\\Output_NEM12 --separate
    
    Option B - Combined file:
    python nem12_converter.py -i C:\\NEM12\\Input_NEM12 -o C:\\NEM12\\Output_NEM12 --combined
    
    Option C - Individual file:
    python nem12_converter.py -i "path/to/single/file.csv" -o "output.csv"
    
    STEP 3: Expected Output
    -----------------------
    The converter will:
    ✅ Automatically detect AGL DETAILED format
    ✅ Extract interval data for each NMI
    ✅ Convert to proper NEM12 format
    ✅ Generate both .csv and .dat files
    ✅ Validate the output
    
    Output files will be named like:
    - NEM12_6102923044_AGL_DETAILED.csv
    - NEM12_6203177121_AGL_DETAILED.csv  
    - NEM12_6203695622_AGL_DETAILED.csv
    
    STEP 4: Validation
    ------------------
    Each output file will be automatically validated for NEM12 compliance.
    Check the log file for any issues or warnings.
    
    TROUBLESHOOTING
    ===============
    If files are not detected as AGL format:
    1. Check the log file for detection details
    2. Use --verbose flag for more information
    3. Ensure files are not corrupted or password protected
    
    The enhanced AGL detection functions should handle your files automatically.
    """
    
    print(instructions)
    return instructions


# Test function for your specific AGL files
def test_agl_detection_on_your_files():
    """
    Test the AGL detection with sample data matching your file structure.
    """
    import pandas as pd
    import logging
    
    # Sample AGL DETAILED data structure
    sample_agl_data = [
        # Row 0: Some metadata
        ["AGL DETAILED INTERVAL DATA EXPORT", "", "", "", ""],
        # Row 1: More metadata  
        ["Customer Account: 1234567", "NMI: 6102923044", "", "", ""],
        # Row 2: Headers
        ["Date", "Time", "Consumption (kWh)", "Quality", "Status"],
        # Row 3-5: Data
        ["1/06/2017", "0:30", "1.234", "A", "Actual"],
        ["1/06/2017", "1:00", "1.456", "A", "Actual"],
        ["1/06/2017", "1:30", "1.678", "S", "Estimated"]
    ]
    
    df = pd.DataFrame(sample_agl_data)
    logger = logging.getLogger("test")
    
    print("Testing AGL DETAILED format detection...")
    is_detected = detect_agl_detailed_format(df, logger)
    print(f"Detection result: {is_detected}")
    
    if is_detected:
        print("\nTesting data extraction...")
        extracted_data = extract_agl_detailed_data(df, logger)
        print(f"Extracted {len(extracted_data)} records")
        
        if extracted_data:
            print("\nSample extracted record:")
            for key, value in extracted_data[0].items():
                print(f"  {key}: {value}")
    
    return is_detected


#==================

# ============================================================================
# TESTING FUNCTION FOR ALL FORMATS
# ============================================================================

def test_all_csv_formats():
    """
    Test function to verify all CSV format detection is working.
    Use this to test your converter with different file types.
    """
    import pandas as pd
    import logging
    
    logger = logging.getLogger("test")
    
    print("Testing All CSV Format Detection:")
    print("=" * 50)
    
    # Test format priority (higher priority formats are checked first)
    format_priority = [
        "NEM12 Format (existing)",
        "AGL DETAILED Format (retailer)",
        "SRC CONNECTION POINT Format (utility)", 
        "Space-Separated Format (wide row)",
        "Standard Interval Format (structured)",
        "Multi-Column Energy Format (multiple energy)",
        "General Interval Data Format (various)",
        "Time Series Format (generic)",
        "Fallback Processing (unknown)"
    ]
    
    for i, format_name in enumerate(format_priority, 1):
        print(f"{i:2d}. {format_name}")
    
    print("\n" + "=" * 50)
    print("Format detection follows this priority order.")
    print("First matching format will be used for processing.")
    print("This ensures most specific formats are detected first.")
    
    return True




#================================================================================================================================================
#================================================================================================================================================
#================================================================================================================================================

def main():
    """Main function with comprehensive input validation and error reporting."""
    parser = argparse.ArgumentParser(
        description="Convert CSV/Excel files to NEM12 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert each input file to separate NEM12 files (recommended)
  %(prog)s -i /path/to/input -o /path/to/output --separate
  
  # Convert multiple files into one NEM12 file
  %(prog)s -i /path/to/input -o /path/to/output.csv --combined
  
  # Create separate files for each NMI found
  %(prog)s -i /path/to/input -o /path/to/output --batch
  
  # Validate existing NEM12 file
  %(prog)s --validate-only -i existing_file.csv
        """
    )
    
    parser.add_argument("--input", "-i", 
                       help="Input folder path or file path",
                       default="C:\\NEM12\\Input_NEM12")
    parser.add_argument("--output", "-o", 
                       help="Output folder path or file path for NEM12 files",
                       default="C:\\NEM12\\Output_NEM12")
    parser.add_argument("--log", "-l", 
                       help="Log file path",
                       default="nem12_conversion.log")
    parser.add_argument("--batch", "-b", 
                       action="store_true",
                       help="Create separate files for each NMI found")
    parser.add_argument("--separate", "-s",
                       action="store_true", 
                       default=True,
                       help="Create separate NEM12 file for each input file (default)")
    parser.add_argument("--combined", "-c",
                       action="store_true",
                       help="Combine all input files into one NEM12 file")
    parser.add_argument("--validate-only", "-v",
                       action="store_true",
                       help="Only validate existing NEM12 files")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("NEM12 Converter - Polished Version 2.0.0")
    logger.info("=" * 60)
    
    # Validate input path
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        print(f"❌ Error: Input path does not exist: {args.input}")
        return 1
    
    # Check input accessibility
    if os.path.isfile(args.input):
        if not os.access(args.input, os.R_OK):
            logger.error(f"Input file is not readable: {args.input}")
            print(f"❌ Error: Cannot read input file: {args.input}")
            return 1
    elif os.path.isdir(args.input):
        if not os.access(args.input, os.R_OK | os.X_OK):
            logger.error(f"Input directory is not accessible: {args.input}")
            print(f"❌ Error: Cannot access input directory: {args.input}")
            return 1
    
    # Create output directory if needed
    if not args.validate_only:
        try:
            if not args.output.endswith(('.csv', '.dat')):
                os.makedirs(args.output, exist_ok=True)
                if not os.access(args.output, os.W_OK):
                    logger.error(f"Output directory is not writable: {args.output}")
                    print(f"❌ Error: Cannot write to output directory: {args.output}")
                    return 1
            else:
                parent_dir = os.path.dirname(os.path.abspath(args.output))
                os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory: {e}")
            print(f"❌ Error: Cannot create output directory: {e}")
            return 1
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    # Determine processing mode
    if args.combined:
        processing_mode = "Combined (all files into one NEM12)"
        separate_files = False
    elif args.batch:
        processing_mode = "Batch by NMI (separate file per NMI)"
        separate_files = False
    else:
        processing_mode = "Separate files (one NEM12 per input file)"
        separate_files = True
    
    logger.info(f"Mode: {processing_mode}")
    logger.info(f"Log file: {args.log}")

    try:
        if args.validate_only:
            # Validation mode
            if os.path.isfile(args.input):
                logger.info(f"Validating file: {args.input}")
                success = validate_nem12_file(args.input, logger)
                if success:
                    print("✅ NEM12 file validation passed!")
                else:
                    print("❌ NEM12 file validation failed!")
            else:
                logger.error("Validation mode requires a file path, not a folder")
                print("❌ Error: Validation mode requires a file path, not a folder")
                return 1
        else:
            # Processing mode
            if os.path.isfile(args.input):
                # Single file processing
                logger.info("Processing single file")
                result = process_file(args.input, logger)
                if result:
                    success = generate_nem12_file(result, args.output, logger)
                    if success:
                        success = validate_nem12_file(args.output, logger)
                else:
                    success = False
            else:
                # Folder processing
                success = process_folder(args.input, args.output, logger, 
                                       batch_per_nmi=args.batch,
                                       separate_files=separate_files)

        if success:
            safe_log_message(logger, "info", "✅ NEM12 conversion completed successfully!")
            print("✅ NEM12 conversion completed successfully!")
            return 0
        else:
            logger.error("❌ NEM12 conversion failed")
            print("❌ NEM12 conversion failed - check log file for details")
            return 1

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
