import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NEM12Converter:
    def __init__(self, input_dir="C:\\NEM12\\Input_NEM12", output_dir="C:\\NEM12\\Output_NEM12"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NEM12 standard configuration
        self.version_header = "NEM12"
        self.creation_date = datetime.now().strftime("%Y%m%d%H%M%S")
        self.from_participant = "AEMO"
        self.to_participant = "RETAILER"
        
        # Default meter configuration
        self.default_nmi = "6001425887"
        self.default_suffix = "E1"
        self.default_register = "E1"
        self.default_stream = "E1"
        self.default_meter_serial = "17435764"
        self.default_uom = "KWH"
        self.default_interval_length = "30"
        self.default_next_scheduled_read = (datetime.now() + timedelta(days=30)).strftime("%Y%m%d")
        
    def _format_date_yyyymmdd(self, date_obj):
        """Format date as YYYYMMDD"""
        return date_obj.strftime("%Y%m%d")
    
    def _write_nem12_file(self, data_dict, output_file, nmi_number=None, suffix=None):
        """Write data in proper NEM12 standard format"""
        try:
            # Use defaults if not provided
            nmi_number = nmi_number or self.default_nmi
            suffix = suffix or self.default_suffix
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
                
                # 100 Record - NEM12 Header
                header_100 = [
                    "100",
                    self.version_header,
                    self.creation_date,
                    self.from_participant,
                    self.to_participant
                ]
                writer.writerow(header_100)
                
                # 200 Record - NMI Data Details
                record_200 = [
                    "200",
                    nmi_number,
                    suffix,
                    self.default_register,
                    self.default_stream,
                    self.default_meter_serial,
                    self.default_uom,
                    self.default_interval_length,
                    self.default_next_scheduled_read
                ]
                writer.writerow(record_200)
                
                # 300 Records - Interval Data
                for date_str in sorted(data_dict.keys()):
                    values = data_dict[date_str]
                    
                    # Ensure exactly 48 values for 30-minute intervals
                    while len(values) < 48:
                        values.append(0.0)
                    values = values[:48]
                    
                    # Format values - NEM12 standard precision
                    formatted_values = []
                    for val in values:
                        if val == 0 or val == 0.0:
                            formatted_values.append("0")
                        else:
                            # Format to appropriate decimal places
                            formatted_values.append(f"{float(val):.3f}")
                    
                    # Create 300 record
                    record_300 = ["300", date_str] + formatted_values + ["A", "", f"{date_str}000000"]
                    writer.writerow(record_300)
                
                # 900 Record - End of File
                writer.writerow(["900"])
            
            logger.info(f"Successfully wrote NEM12 file: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing NEM12 file {output_file}: {str(e)}")
            return False

    def detect_input1_format(self, file_path):
        """Detect if this is the Input_1 horizontal NEM12 format"""
        try:
            if file_path.name.lower() != 'input_1.csv':
                return False
                
            # Read first row to check structure - handle the parsing error
            try:
                df = pd.read_csv(file_path, nrows=1, on_bad_lines='skip')
            except:
                # If normal CSV reading fails, try reading as raw text
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                columns = first_line.split(',')
            else:
                columns = list(df.columns)
            
            # Check for NEM12 indicators in columns
            has_100 = '100' in columns
            has_nem12 = 'NEM12' in columns
            has_scientific_timestamp = any('E+' in str(col) for col in columns)
            has_200 = '200' in columns
            
            is_input1_format = has_100 and has_nem12 and has_scientific_timestamp and has_200
            
            if is_input1_format:
                logger.info(f"Detected Input_1 horizontal NEM12 format")
            
            return is_input1_format
            
        except Exception as e:
            logger.warning(f"Error detecting Input_1 format: {str(e)}")
            return False

    def convert_input1_horizontal_nem12_format(self, file_path):
        """
        SPECIFIC FIX for Input_1.csv format
        This file contains NEM12 data spread horizontally across CSV columns
        with timestamps in scientific notation that need to be converted
        """
        try:
            logger.info(f"Converting Input_1 horizontal NEM12 format: {file_path.name}")
            
            # Read the CSV file with special handling for malformed CSV
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
            except:
                # If pandas fails, read manually
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                row_data = first_line.split(',')
            else:
                # Convert to list for easier processing
                row_data = []
                for col in df.columns:
                    # Handle scientific notation timestamps
                    if 'E+' in str(col):
                        try:
                            # Convert scientific notation to regular number
                            timestamp = int(float(col))
                            row_data.append(str(timestamp))
                        except:
                            row_data.append(str(col))
                    else:
                        row_data.append(str(col))
            
            logger.info(f"Converted columns to row data: {len(row_data)} elements")
            logger.info(f"First 10 elements: {row_data[:10]}")
            
            data_dict = {}
            
            # Find all 300 record positions (these contain the daily data)
            i = 0
            while i < len(row_data):
                if row_data[i] == '300':
                    try:
                        # Next element should be the date
                        if i + 1 < len(row_data):
                            date_str = row_data[i + 1]
                            
                            # Validate date format (should be YYYYMMDD)
                            if len(date_str) == 8 and date_str.isdigit():
                                logger.debug(f"Processing date: {date_str}")
                                
                                # Extract 48 interval values (next 48 elements)
                                values = []
                                for j in range(i + 2, i + 50):  # 48 intervals
                                    if j < len(row_data):
                                        try:
                                            # Handle the 'A' quality flag that appears in the data
                                            val_str = row_data[j]
                                            if val_str == 'A' or val_str == '':
                                                break  # End of intervals for this day
                                            val = float(val_str)
                                            values.append(val)
                                        except ValueError:
                                            # Hit non-numeric value, probably end of intervals
                                            break
                                    else:
                                        break
                                
                                # Ensure we have exactly 48 intervals
                                while len(values) < 48:
                                    values.append(0.0)
                                
                                if len(values) >= 48:
                                    data_dict[date_str] = values[:48]
                                    logger.debug(f"Added {date_str}: {len(values)} intervals, total: {sum(values[:48]):.3f} kWh")
                                
                                # Skip past this record's data
                                i += 52  # 300 + date + 48 intervals + quality flag + timestamp
                            else:
                                i += 1
                        else:
                            i += 1
                    except Exception as e:
                        logger.warning(f"Error processing 300 record at position {i}: {str(e)}")
                        i += 1
                else:
                    i += 1
            
            logger.info(f"Converted {len(data_dict)} days from Input_1 horizontal NEM12 format")
            
            if data_dict:
                # Log some statistics
                total_energy = sum(sum(day_data) for day_data in data_dict.values())
                avg_daily_energy = total_energy / len(data_dict)
                logger.info(f"Total energy: {total_energy:.1f} kWh, Average daily: {avg_daily_energy:.1f} kWh")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting Input_1 horizontal NEM12 format {file_path}: {str(e)}")
            return None

    def detect_excel_nem12_row_format(self, file_path):
        """Detect if this is Excel file with NEM12 data in row format (Input_4, Input_6, Input_7 style)"""
        try:
            # Read first few rows to check structure
            df = pd.read_excel(file_path, header=None, nrows=5)
            
            if df.empty:
                return False
            
            # Check if first column contains 300 records
            has_300_records = any(str(val) == '300' or val == 300 for val in df.iloc[:, 0])
            
            # Check if we have numeric columns that could be dates
            has_date_like_values = False
            for _, row in df.iterrows():
                if len(row) > 1 and str(row.iloc[0]) == '300':
                    # Check if second column looks like a date (8 digits)
                    date_val = row.iloc[1]
                    try:
                        date_str = str(int(date_val)) if pd.notna(date_val) else ""
                        if len(date_str) == 8 and date_str.isdigit():
                            has_date_like_values = True
                            break
                    except:
                        continue
            
            # Reduced column requirement to handle Input_4/6
            has_enough_columns = len(df.columns) >= 25
            
            is_excel_nem12_row = has_300_records and has_date_like_values and has_enough_columns
            
            if is_excel_nem12_row:
                logger.info(f"Detected Excel NEM12 row format in {file_path.name}")
            
            return is_excel_nem12_row
            
        except Exception as e:
            logger.warning(f"Error detecting Excel NEM12 row format: {str(e)}")
            return False

    def convert_excel_nem12_row_format(self, file_path):
        """
        Convert Excel file with NEM12 data in row format
        Each row: 300 | DATE | 48 interval values | quality flags
        """
        try:
            logger.info(f"Converting Excel NEM12 row format: {file_path.name}")
            
            # Read Excel file without headers
            df = pd.read_excel(file_path, header=None)
            
            if df.empty:
                logger.warning(f"No data found in Excel file: {file_path.name}")
                return None
            
            data_dict = {}
            
            # Process each row
            for row_idx, row in df.iterrows():
                try:
                    # Check if this row starts with 300
                    if len(row) > 1 and (str(row.iloc[0]) == '300' or row.iloc[0] == 300):
                        # Extract date from second column
                        date_val = row.iloc[1]
                        
                        try:
                            # Handle different date formats
                            if pd.isna(date_val):
                                continue
                            
                            # Convert to string and validate
                            date_str = str(int(date_val))
                            
                            # Validate date format (should be YYYYMMDD)
                            if len(date_str) == 8 and date_str.isdigit():
                                logger.debug(f"Processing date: {date_str}")
                                
                                # Extract 48 interval values (columns 2 to 49)
                                values = []
                                for col_idx in range(2, min(50, len(row))):  # Start from column 2, get up to 48 values
                                    if col_idx < len(row):
                                        val = row.iloc[col_idx]
                                        try:
                                            if pd.isna(val):
                                                values.append(0.0)
                                            else:
                                                # Check if it's a quality flag or timestamp (usually at the end)
                                                if isinstance(val, str) and val in ['A', 'E', 'F', 'N']:
                                                    break  # Hit quality flag, stop reading intervals
                                                values.append(float(val))
                                        except (ValueError, TypeError):
                                            values.append(0.0)
                                    else:
                                        values.append(0.0)
                                
                                # Ensure we have exactly 48 intervals
                                while len(values) < 48:
                                    values.append(0.0)
                                
                                # Store the data
                                data_dict[date_str] = values[:48]
                                logger.debug(f"Added {date_str}: 48 intervals, total energy: {sum(values[:48]):.3f} kWh")
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing date in row {row_idx}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing row {row_idx}: {str(e)}")
                    continue
            
            logger.info(f"Converted {len(data_dict)} days from Excel NEM12 row format")
            
            if data_dict:
                # Log some statistics
                total_energy = sum(sum(day_data) for day_data in data_dict.values())
                avg_daily_energy = total_energy / len(data_dict)
                logger.info(f"Total energy: {total_energy:.1f} kWh, Average daily: {avg_daily_energy:.1f} kWh")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting Excel NEM12 row format {file_path}: {str(e)}")
            return None
    
    def convert_timeseries_format(self, file_path):
        """Convert time-series CSV format (multiple timestamps per day) - FIXED FOR INPUT_10"""
        try:
            df = pd.read_csv(file_path)
            
            # Enhanced column detection - same logic as analyzer
            datetime_col = None
            power_col = None
            
            # Print available columns for debugging
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Look for datetime columns
            datetime_keywords = [
                'read_datetime', 'read datetime', 'datetime', 'timestamp', 
                'endtime', 'end time', 'local time', 'time', 'date'
            ]
            
            for keyword in datetime_keywords:
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if keyword in col_lower:
                        datetime_col = col
                        logger.info(f"Found datetime column: '{col}'")
                        break
                if datetime_col:
                    break
            
            # Look for power/energy columns with better prioritization - FIXED LOGIC
            energy_priority_keywords = ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'generated kwh']
            power_priority_keywords = ['average kw', 'kw', 'power']
            general_keywords = ['e', 'energy']
            
            # First try energy columns (preferred)
            for keyword in energy_priority_keywords:
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if keyword in col_lower and not any(exclude in col_lower for exclude in 
                        ['serial', 'nmi', 'id', 'identifier', 'point', 'time', 'date', 'type', 'code']):
                        power_col = col
                        logger.info(f"Found energy column (priority): '{col}'")
                        break
                if power_col:
                    break
            
            # Then try power columns
            if power_col is None:
                for keyword in power_priority_keywords:
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower == keyword or (keyword in col_lower and not any(exclude in col_lower for exclude in 
                            ['serial', 'nmi', 'id', 'identifier', 'point', 'time', 'date', 'type', 'code'])):
                            power_col = col
                            logger.info(f"Found power column: '{col}'")
                            break
                    if power_col:
                        break
            
            # Finally try general energy terms (but be more selective)
            if power_col is None:
                for keyword in general_keywords:
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower == keyword:  # Only exact matches for 'E' to avoid false positives
                            power_col = col
                            logger.info(f"Found energy column (general): '{col}'")
                            break
                    if power_col:
                        break
            
            if datetime_col is None or power_col is None:
                logger.warning(f"Could not find required columns in {file_path}")
                logger.info(f"Datetime column: {datetime_col}, Power column: {power_col}")
                return self.convert_csv_format2(file_path)  # Fallback
            
            logger.info(f"Using datetime column: '{datetime_col}', power column: '{power_col}'")
            
            # Parse datetime and extract date/time
            df['datetime_parsed'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['datetime_parsed'])  # Remove invalid dates
            
            if df.empty:
                logger.warning(f"No valid dates found in {file_path}")
                return None
            
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['minute'] = df['datetime_parsed'].dt.minute
            
            # Convert to 30-minute intervals (0-47 intervals per day)
            df['interval_index'] = df['hour'] * 2 + (df['minute'] // 30)
            
            # Clean power values
            df[power_col] = pd.to_numeric(df[power_col], errors='coerce').fillna(0)
            
            # CRITICAL FIX: Determine if values are already energy or need conversion
            col_lower = power_col.lower()
            is_energy_column = any(keyword in col_lower for keyword in ['kwh', 'energy', 'usage'])
            is_power_column = any(keyword in col_lower for keyword in ['kw', 'power']) and 'kwh' not in col_lower
            
            logger.info(f"Column '{power_col}' analysis: is_energy={is_energy_column}, is_power={is_power_column}")
            
            data_dict = {}
            
            # Group by date
            for date, group in df.groupby('date'):
                date_str = date.strftime("%Y%m%d")
                
                # Initialize 48 intervals with zeros
                intervals = [0.0] * 48
                
                # Aggregate by interval (average if multiple readings per interval)
                interval_groups = group.groupby('interval_index')[power_col].mean()
                
                # Fill intervals with values - CORRECTED LOGIC
                for interval_idx, avg_value in interval_groups.items():
                    if 0 <= interval_idx < 48:
                        if is_energy_column:
                            # Already energy (kWh) - use directly
                            energy_value = float(avg_value)
                        elif is_power_column:
                            # Power (kW) - convert to energy for 30-minute interval
                            energy_value = float(avg_value) * 0.5
                        else:
                            # Unknown - assume energy if values are reasonable for energy
                            if 0.01 <= abs(avg_value) <= 1000:
                                energy_value = float(avg_value)  # Assume energy
                            else:
                                energy_value = float(avg_value) * 0.5  # Assume power
                        
                        intervals[int(interval_idx)] = energy_value
                
                # Only add if we have some data
                if any(val > 0 for val in intervals):
                    data_dict[date_str] = intervals
                    logger.debug(f"Date {date_str}: {len(group)} readings, {len(interval_groups)} intervals")
            
            logger.info(f"Converted {len(data_dict)} days of time-series data")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting time-series format {file_path}: {str(e)}")
            return None
    
    def convert_timeseries_excel(self, df):
        """Convert time-series Excel format - FIXED FOR INPUT_10"""
        try:
            # Enhanced column detection - same logic as CSV version
            datetime_col = None
            power_col = None
            
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Look for datetime columns
            datetime_keywords = [
                'readingdatetime', 'read_datetime', 'read datetime', 'datetime', 'timestamp', 
                'endtime', 'end time', 'local time', 'time', 'date'
            ]
            
            for keyword in datetime_keywords:
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if keyword in col_lower:
                        datetime_col = col
                        logger.info(f"Found datetime column: '{col}'")
                        break
                if datetime_col:
                    break
            
            # Look for power/energy columns with better prioritization - FIXED LOGIC
            energy_priority_keywords = ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'generated kwh']
            power_priority_keywords = ['average kw', 'kw', 'power']
            general_keywords = ['e', 'energy']
            
            # First try energy columns (preferred)
            for keyword in energy_priority_keywords:
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if keyword in col_lower and not any(exclude in col_lower for exclude in 
                        ['serial', 'nmi', 'id', 'identifier', 'point', 'time', 'date', 'type', 'code']):
                        power_col = col
                        logger.info(f"Found energy column (priority): '{col}'")
                        break
                if power_col:
                    break
            
            # Then try power columns
            if power_col is None:
                for keyword in power_priority_keywords:
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower == keyword or (keyword in col_lower and not any(exclude in col_lower for exclude in 
                            ['serial', 'nmi', 'id', 'identifier', 'point', 'time', 'date', 'type', 'code'])):
                            power_col = col
                            logger.info(f"Found power column: '{col}'")
                            break
                    if power_col:
                        break
            
            # Finally try general energy terms (but be more selective)
            if power_col is None:
                for keyword in general_keywords:
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower == keyword:  # Only exact matches for 'E' to avoid false positives
                            power_col = col
                            logger.info(f"Found energy column (general): '{col}'")
                            break
                    if power_col:
                        break
            
            if datetime_col is None or power_col is None:
                logger.warning(f"Could not find datetime or power columns in Excel")
                logger.info(f"Datetime column: {datetime_col}, Power column: {power_col}")
                return None
            
            logger.info(f"Using datetime column: '{datetime_col}', power column: '{power_col}'")
            
            # Parse datetime and extract date/time
            df['datetime_parsed'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['datetime_parsed'])  # Remove invalid dates
            
            if df.empty:
                logger.warning(f"No valid dates found in Excel file")
                return None
            
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['minute'] = df['datetime_parsed'].dt.minute
            
            # Convert to 30-minute intervals (0-47 intervals per day)
            df['interval_index'] = df['hour'] * 2 + (df['minute'] // 30)
            
            # Clean power values
            df[power_col] = pd.to_numeric(df[power_col], errors='coerce').fillna(0)
            
            # CRITICAL FIX: Determine if values are already energy or need conversion
            col_lower = power_col.lower()
            is_energy_column = any(keyword in col_lower for keyword in ['kwh', 'energy', 'usage'])
            is_power_column = any(keyword in col_lower for keyword in ['kw', 'power']) and 'kwh' not in col_lower
            
            logger.info(f"Column '{power_col}' analysis: is_energy={is_energy_column}, is_power={is_power_column}")
            
            data_dict = {}
            
            # Group by date
            for date, group in df.groupby('date'):
                date_str = date.strftime("%Y%m%d")
                
                # Initialize 48 intervals with zeros
                intervals = [0.0] * 48
                
                # Aggregate by interval (average if multiple readings per interval)
                interval_groups = group.groupby('interval_index')[power_col].mean()
                
                # Fill intervals with values - CORRECTED LOGIC
                for interval_idx, avg_value in interval_groups.items():
                    if 0 <= interval_idx < 48:
                        if is_energy_column:
                            # Already energy (kWh) - use directly
                            energy_value = float(avg_value)
                            logger.debug(f"Using energy directly: {avg_value} kWh")
                        elif is_power_column:
                            # Power (kW) - convert to energy for 30-minute interval
                            energy_value = float(avg_value) * 0.5
                            logger.debug(f"Converting power to energy: {avg_value} kW -> {energy_value} kWh")
                        else:
                            # Unknown - assume energy if values are reasonable for energy
                            if 0.01 <= abs(avg_value) <= 1000:
                                energy_value = float(avg_value)  # Assume energy
                                logger.debug(f"Assuming energy: {avg_value} kWh")
                            else:
                                energy_value = float(avg_value) * 0.5  # Assume power
                                logger.debug(f"Assuming power, converting: {avg_value} -> {energy_value} kWh")
                        
                        intervals[int(interval_idx)] = energy_value
                
                # Only add if we have some data
                if any(val > 0 for val in intervals):
                    data_dict[date_str] = intervals
            
            logger.info(f"Converted {len(data_dict)} days of time-series Excel data")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting time-series Excel format: {str(e)}")
            return None
    
    def convert_excel_format(self, file_path):
        """Convert Excel file with date and interval data"""
        try:
            # Try reading first sheet
            df = pd.read_excel(file_path, sheet_name=0, header=None)
            
            # Check if it's already in NEM12-like format with 200/300 records
            if len(df.columns) > 2 and (df.iloc[0, 0] == 200 or df.iloc[0, 0] == "200"):
                return self.convert_existing_nem12_excel(df)
            
            # Try with headers for time-series format
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Check if it's time-series format
            has_datetime_col = any(any(keyword in str(col).lower() for keyword in 
                                     ['readingdatetime', 'read_datetime', 'datetime', 'timestamp', 'read datetime', 'endtime', 'local time']) 
                                 for col in df.columns)
            has_power_col = any(any(keyword in str(col).lower() for keyword in 
                                  ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'kw', 'power']) and 
                              not any(exclude in str(col).lower() for exclude in 
                                     ['serial', 'nmi', 'id', 'identifier', 'point'])
                              for col in df.columns)
            
            if has_datetime_col and has_power_col:
                logger.info(f"Detected time-series format in Excel file")
                return self.convert_timeseries_excel(df)
            
            # Regular format processing
            date_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['date', 'time', 'day', 'timestamp']):
                    date_col = col
                    break
            
            if date_col is None:
                date_col = df.columns[0]  # Use first column as date
            
            data_dict = {}
            for _, row in df.iterrows():
                try:
                    date_obj = pd.to_datetime(row[date_col])
                    date_str = self._format_date_yyyymmdd(date_obj)
                    
                    # Extract numerical values (skip date column)
                    values = []
                    for col in df.columns:
                        if col != date_col:
                            val = row[col] if pd.notna(row[col]) else 0
                            try:
                                values.append(float(val))
                            except:
                                values.append(0.0)
                    
                    # Ensure 48 intervals
                    while len(values) < 48:
                        values.append(0.0)
                    
                    data_dict[date_str] = values[:48]
                    
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {str(e)}")
                    continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting Excel format {file_path}: {str(e)}")
            return None
    
    def convert_csv_format2(self, file_path):
        """Convert CSV with date column and value columns"""
        try:
            df = pd.read_csv(file_path)
            
            # Look for date column
            date_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['date', 'time', 'day', 'timestamp']):
                    date_col = col
                    break
            
            if date_col is None:
                date_col = df.columns[0]  # Use first column as date
            
            data_dict = {}
            for _, row in df.iterrows():
                try:
                    date_obj = pd.to_datetime(row[date_col])
                    date_str = self._format_date_yyyymmdd(date_obj)
                    
                    # Extract numerical values (skip date column)
                    values = []
                    for col in df.columns:
                        if col != date_col:
                            val = row[col] if pd.notna(row[col]) else 0
                            try:
                                values.append(float(val))
                            except:
                                values.append(0.0)
                    
                    # Ensure 48 intervals
                    while len(values) < 48:
                        values.append(0.0)
                    
                    data_dict[date_str] = values[:48]
                    
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {str(e)}")
                    continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting CSV format2 {file_path}: {str(e)}")
            return None
    
    def convert_existing_nem12_csv(self, file_path):
        """Convert existing NEM12 CSV format"""
        try:
            data_dict = {}
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > 2 and row[0] == '300':
                        try:
                            date_str = str(int(float(row[1])))
                            values = []
                            
                            # Extract values from columns 2 onwards (up to 48 values)
                            for i in range(2, min(50, len(row))):
                                if i < len(row) and row[i] and row[i] != '':
                                    try:
                                        values.append(float(row[i]))
                                    except:
                                        values.append(0.0)
                                else:
                                    values.append(0.0)
                            
                            # Ensure 48 intervals
                            while len(values) < 48:
                                values.append(0.0)
                            
                            data_dict[date_str] = values[:48]
                            
                        except Exception as e:
                            logger.warning(f"Skipping row due to error: {str(e)}")
                            continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting existing NEM12 CSV {file_path}: {str(e)}")
            return None
    
    def convert_existing_nem12_excel(self, df):
        """Convert existing NEM12-like Excel format"""
        try:
            data_dict = {}
            
            for _, row in df.iterrows():
                if len(row) > 2 and (row.iloc[0] == 300 or row.iloc[0] == "300"):
                    try:
                        date_str = str(int(float(row.iloc[1])))
                        values = []
                        
                        # Extract values from columns 2 onwards (up to 48 values)
                        for i in range(2, min(50, len(row))):
                            val = row.iloc[i] if pd.notna(row.iloc[i]) else 0
                            values.append(float(val))
                        
                        # Ensure 48 intervals
                        while len(values) < 48:
                            values.append(0.0)
                        
                        data_dict[date_str] = values[:48]
                        
                    except Exception as e:
                        logger.warning(f"Skipping row due to error: {str(e)}")
                        continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting existing NEM12 Excel format: {str(e)}")
            return None
        
    def detect_interval_energy_format(self, file_path):
        """
        Detect if this is interval energy format:
        - Each row represents one 30-minute interval
        - Has 'kWh Value' or similar energy column
        - Has period/datetime column showing 30-min intervals
        """
        try:
            # Read sample to check format
            df_sample = pd.read_csv(file_path, nrows=50)
            
            # Check for key indicators
            has_kwh_value = 'kWh Value' in df_sample.columns
            has_period = 'Period' in df_sample.columns or any('time' in str(col).lower() for col in df_sample.columns)
            
            # Check if rows represent individual intervals (should be many rows)
            total_rows = len(pd.read_csv(file_path))
            has_many_intervals = total_rows > 1000  # Expect thousands of 30-min intervals
            
            # Check if kWh values are reasonable for 30-min intervals (0.1 to 500 kWh per interval)
            kwh_reasonable = False
            if has_kwh_value:
                sample_values = pd.to_numeric(df_sample['kWh Value'], errors='coerce').dropna()
                if len(sample_values) > 0:
                    avg_val = sample_values.mean()
                    kwh_reasonable = 0.1 <= avg_val <= 500  # Reasonable for 30-min intervals
            
            is_interval_energy_format = has_kwh_value and has_period and has_many_intervals and kwh_reasonable
            
            if is_interval_energy_format:
                logger.info(f"Detected interval energy format in {file_path.name}")
                logger.info(f"  Rows: {total_rows}, Avg kWh: {avg_val:.2f}")
            
            return is_interval_energy_format
            
        except Exception as e:
            logger.warning(f"Error detecting interval energy format: {str(e)}")
            return False

    def convert_interval_energy_format(self, file_path):
        """
        Convert interval energy format where each row = one 30-minute interval
        Format: Each row has date/time and kWh Value for that specific interval
        """
        try:
            logger.info(f"Converting interval energy format: {file_path.name}")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info(f"Total intervals: {len(df)}")
            
            # Find the datetime and energy columns
            datetime_col = None
            energy_col = None
            
            # Look for datetime column
            datetime_candidates = ['Period', 'End Time', 'Start Time', 'DateTime', 'Time']
            for col in datetime_candidates:
                if col in df.columns:
                    datetime_col = col
                    logger.info(f"Found datetime column: '{col}'")
                    break
            
            # Look for energy column
            energy_candidates = ['kWh Value', 'kWh Actual', 'Energy', 'kWh']
            for col in energy_candidates:
                if col in df.columns:
                    energy_col = col
                    logger.info(f"Found energy column: '{col}'")
                    break
            
            if datetime_col is None or energy_col is None:
                logger.error(f"Could not find required columns. DateTime: {datetime_col}, Energy: {energy_col}")
                return None
            
            # Parse datetime
            df['datetime_parsed'] = pd.to_datetime(df[datetime_col], errors='coerce')
            df = df.dropna(subset=['datetime_parsed'])
            
            if df.empty:
                logger.warning(f"No valid dates found")
                return None
            
            # Extract date and time components
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['minute'] = df['datetime_parsed'].dt.minute
            
            # Calculate 30-minute interval index (0-47)
            df['interval_index'] = df['hour'] * 2 + (df['minute'] // 30)
            
            # Clean energy values - these are already kWh, use directly!
            df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce').fillna(0)
            
            logger.info(f"Energy column '{energy_col}' - using values directly as kWh")
            logger.info(f"Sample energy values: {df[energy_col].head(5).tolist()}")
            
            data_dict = {}
            
            # Group by date and build daily interval data
            for date, group in df.groupby('date'):
                date_str = date.strftime("%Y%m%d")
                
                # Initialize 48 intervals with zeros
                intervals = [0.0] * 48
                
                # Fill intervals with kWh values (use directly, no conversion!)
                for _, row in group.iterrows():
                    interval_idx = int(row['interval_index'])
                    kwh_value = row[energy_col]
                    
                    # Ensure interval index is valid
                    if 0 <= interval_idx < 48:
                        intervals[interval_idx] = float(kwh_value)
                
                # Only add if we have some non-zero data
                if any(val > 0 for val in intervals):
                    data_dict[date_str] = intervals
                    daily_total = sum(intervals)
                    logger.debug(f"Date {date_str}: {len(group)} intervals, daily total: {daily_total:.1f} kWh")
            
            logger.info(f"Converted {len(data_dict)} days from interval energy format")
            
            # Log conversion statistics
            if data_dict:
                total_energy = sum(sum(day_data) for day_data in data_dict.values())
                avg_daily_energy = total_energy / len(data_dict)
                logger.info(f"Total energy: {total_energy:.1f} kWh, Average daily: {avg_daily_energy:.1f} kWh")
                
                # Calculate expected annual energy
                annual_energy = avg_daily_energy * 365
                logger.info(f"Projected annual energy: {annual_energy:.0f} kWh/year")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting interval energy format {file_path}: {str(e)}")
            return None

    def detect_and_convert_file(self, file_path):
        """Enhanced detection with all format handlers - CORRECTED PRIORITIES"""
        file_ext = file_path.suffix.lower()
        
        # PRIORITY 1: Check for Input_1 horizontal NEM12 format
        if self.detect_input1_format(file_path):
            logger.info(f"Using Input_1 horizontal NEM12 handler for {file_path.name}")
            return self.convert_input1_horizontal_nem12_format(file_path)
        
        # PRIORITY 2: Check for Excel NEM12 row format (Input_4, Input_6, Input_7 style)
        if file_ext in ['.xlsx', '.xls'] and self.detect_excel_nem12_row_format(file_path):
            logger.info(f"Using Excel NEM12 row format handler for {file_path.name}")
            return self.convert_excel_nem12_row_format(file_path)
        
        # PRIORITY 3: Try to detect existing NEM12 format
        if file_ext in ['.csv', '.txt']:
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                if first_line.startswith('100,NEM12') or first_line.startswith('200,'):
                    logger.info(f"Detected existing NEM12 CSV format in {file_path.name}")
                    return self.convert_existing_nem12_csv(file_path)
            except:
                pass
        
        # PRIORITY 4: Check for interval energy format (NEW!)
        if file_ext in ['.csv'] and self.detect_interval_energy_format(file_path):
            logger.info(f"Using interval energy format handler for {file_path.name}")
            return self.convert_interval_energy_format(file_path)
        
        # PRIORITY 5: Check if it's time-series format (multiple timestamps)
        if file_ext in ['.csv']:
            try:
                df_sample = pd.read_csv(file_path, nrows=10)
                
                # Look for time-series indicators - more specific detection
                has_datetime_col = any(any(keyword in str(col).lower() for keyword in 
                                        ['readingdatetime', 'read_datetime', 'datetime', 'timestamp', 'read datetime', 'endtime', 'local time']) 
                                    for col in df_sample.columns)
                has_power_col = any(any(keyword in str(col).lower() for keyword in 
                                    ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'kw', 'power']) and 
                                not any(exclude in str(col).lower() for exclude in 
                                        ['serial', 'nmi', 'id', 'identifier', 'point'])
                                for col in df_sample.columns)
                
                # Also check if we have multiple rows (time-series characteristic)
                total_rows = len(pd.read_csv(file_path))
                has_multiple_timestamps = total_rows > 50  # Likely time-series if many rows
                
                if has_datetime_col and has_power_col and has_multiple_timestamps:
                    logger.info(f"Detected time-series format in {file_path.name} ({total_rows} rows)")
                    return self.convert_timeseries_format(file_path)
                    
            except Exception as e:
                logger.warning(f"Error detecting time-series format: {str(e)}")
        
        # PRIORITY 6: Check Excel files for time-series format
        if file_ext in ['.xlsx', '.xls']:
            try:
                df_sample = pd.read_excel(file_path, nrows=10)
                
                has_datetime_col = any(any(keyword in str(col).lower() for keyword in 
                                        ['readingdatetime', 'read_datetime', 'datetime', 'timestamp', 'read datetime', 'endtime', 'local time']) 
                                    for col in df_sample.columns)
                has_power_col = any(any(keyword in str(col).lower() for keyword in 
                                    ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'kw', 'power']) and 
                                not any(exclude in str(col).lower() for exclude in 
                                        ['serial', 'nmi', 'id', 'identifier', 'point'])
                                for col in df_sample.columns)
                
                if has_datetime_col and has_power_col:
                    logger.info(f"Detected time-series format in Excel file {file_path.name}")
                    return self.convert_excel_format(file_path)  # This will handle time-series
                    
            except Exception as e:
                logger.warning(f"Error detecting Excel time-series format: {str(e)}")
        
        # PRIORITY 7: Extension-based detection (fallback)
        conversion_map = {
            '.csv': self.convert_csv_format2,
            '.xlsx': self.convert_excel_format,
            '.xls': self.convert_excel_format
        }
        
        converter_func = conversion_map.get(file_ext, self.convert_csv_format2)
        logger.info(f"Converting {file_path.name} using {converter_func.__name__}")
        return converter_func(file_path)
    
    def process_all_files(self):
        """Process all files in input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return False
        
        processed_count = 0
        error_count = 0
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file():
                logger.info(f"Processing file: {file_path.name}")
                
                try:
                    # Convert file to data dictionary
                    data_dict = self.detect_and_convert_file(file_path)
                    
                    if data_dict and len(data_dict) > 0:
                        # Generate output filename
                        output_filename = f"{file_path.stem}_NEM12.csv"
                        output_path = self.output_dir / output_filename
                        
                        # Write proper NEM12 format file
                        if self._write_nem12_file(data_dict, output_path):
                            processed_count += 1
                            logger.info(f"Successfully converted: {file_path.name} -> {output_filename}")
                        else:
                            error_count += 1
                    else:
                        error_count += 1
                        logger.error(f"No valid data found in: {file_path.name}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
        
        logger.info(f"Processing complete. Success: {processed_count}, Errors: {error_count}")
        return processed_count > 0

    def process_folder(folder_path, output_path, logger, batch_per_nmi=False, separate_files=True):
        """Process folder function for Streamlit integration"""
        try:
            converter = NEM12Converter(input_dir=folder_path, output_dir=output_path)
            success = converter.process_all_files()
            return success
        except Exception as e:
            logger.error(f"Error in process_folder: {e}")
            return False

    def validate_nem12_file(file_path, logger):
        """Validation function for Streamlit integration"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                logger.error("File is empty")
                return False

            # Check basic NEM12 structure
            first_line = lines[0].strip()
            last_line = lines[-1].strip()
            
            starts_ok = first_line.startswith('200') or first_line.startswith('100')
            ends_ok = last_line.startswith('900')
            
            return starts_ok and ends_ok
                
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    def main():
        """Main execution function"""
        try:
            # Initialize converter
            converter = NEM12Converter()
            
            # Process all files in input directory
            logger.info("Starting NEM12 conversion process...")
            success = converter.process_all_files()
            if success:
                logger.info("NEM12 conversion process completed successfully.")
            else:
                logger.warning("NEM12 conversion process completed with issues.")
            
        except Exception as e:
            logger.error(f"Fatal error in main: {str(e)}")

    if __name__ == "__main__":
        main()
