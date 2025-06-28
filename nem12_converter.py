import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import json
import xml.etree.ElementTree as ET
import logging
import io
import zipfile
from pathlib import Path

# Setup logging for Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlitNEM12Converter:
    def __init__(self):
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
    
    def _write_nem12_file(self, data_dict, nmi_number=None, suffix=None):
        """Write data in proper NEM12 standard format - returns string content"""
        try:
            # Use defaults if not provided
            nmi_number = nmi_number or self.default_nmi
            suffix = suffix or self.default_suffix
            
            # Create string buffer instead of file
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_NONE, escapechar='\\')
            
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
            
            content = output.getvalue()
            output.close()
            
            logger.info(f"Successfully generated NEM12 content with {len(data_dict)} days")
            return content
            
        except Exception as e:
            logger.error(f"Error generating NEM12 content: {str(e)}")
            return None

    def detect_input1_format(self, file_content, filename):
        """Detect if this is the Input_1 horizontal NEM12 format"""
        try:
            if filename.lower() != 'input_1.csv':
                return False
                
            # Read first row to check structure
            first_line = file_content.decode('utf-8').split('\n')[0].strip()
            columns = first_line.split(',')
            
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

    def convert_input1_horizontal_nem12_format(self, file_content, filename):
        """Convert Input_1.csv horizontal NEM12 format"""
        try:
            logger.info(f"Converting Input_1 horizontal NEM12 format: {filename}")
            
            # Convert bytes to string and split into elements
            content_str = file_content.decode('utf-8')
            first_line = content_str.split('\n')[0].strip()
            row_data = first_line.split(',')
            
            # Convert scientific notation timestamps
            for i, col in enumerate(row_data):
                if 'E+' in str(col):
                    try:
                        timestamp = int(float(col))
                        row_data[i] = str(timestamp)
                    except:
                        pass
            
            logger.info(f"Converted columns to row data: {len(row_data)} elements")
            
            data_dict = {}
            
            # Find all 300 record positions
            i = 0
            while i < len(row_data):
                if row_data[i] == '300':
                    try:
                        if i + 1 < len(row_data):
                            date_str = row_data[i + 1]
                            
                            if len(date_str) == 8 and date_str.isdigit():
                                logger.debug(f"Processing date: {date_str}")
                                
                                # Extract 48 interval values
                                values = []
                                for j in range(i + 2, i + 50):
                                    if j < len(row_data):
                                        try:
                                            val_str = row_data[j]
                                            if val_str == 'A' or val_str == '':
                                                break
                                            val = float(val_str)
                                            values.append(val)
                                        except ValueError:
                                            break
                                    else:
                                        break
                                
                                while len(values) < 48:
                                    values.append(0.0)
                                
                                if len(values) >= 48:
                                    data_dict[date_str] = values[:48]
                                
                                i += 52
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
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting Input_1 horizontal NEM12 format: {str(e)}")
            return None

    def detect_excel_nem12_row_format(self, file_content, filename):
        """Detect Excel NEM12 row format"""
        try:
            df = pd.read_excel(io.BytesIO(file_content), header=None, nrows=5)
            
            if df.empty:
                return False
            
            has_300_records = any(str(val) == '300' or val == 300 for val in df.iloc[:, 0])
            
            has_date_like_values = False
            for _, row in df.iterrows():
                if len(row) > 1 and str(row.iloc[0]) == '300':
                    date_val = row.iloc[1]
                    try:
                        date_str = str(int(date_val)) if pd.notna(date_val) else ""
                        if len(date_str) == 8 and date_str.isdigit():
                            has_date_like_values = True
                            break
                    except:
                        continue
            
            has_enough_columns = len(df.columns) >= 25
            
            is_excel_nem12_row = has_300_records and has_date_like_values and has_enough_columns
            
            if is_excel_nem12_row:
                logger.info(f"Detected Excel NEM12 row format in {filename}")
            
            return is_excel_nem12_row
            
        except Exception as e:
            logger.warning(f"Error detecting Excel NEM12 row format: {str(e)}")
            return False

    def convert_excel_nem12_row_format(self, file_content, filename):
        """Convert Excel NEM12 row format - Enhanced to match original logic"""
        try:
            logger.info(f"Converting Excel NEM12 row format: {filename}")
            
            df = pd.read_excel(io.BytesIO(file_content), header=None)
            
            if df.empty:
                logger.warning(f"No data found in Excel file: {filename}")
                return None
            
            data_dict = {}
            
            for row_idx, row in df.iterrows():
                try:
                    if len(row) > 1 and (str(row.iloc[0]) == '300' or row.iloc[0] == 300):
                        date_val = row.iloc[1]
                        
                        try:
                            if pd.isna(date_val):
                                continue
                            
                            date_str = str(int(date_val))
                            
                            if len(date_str) == 8 and date_str.isdigit():
                                logger.debug(f"Processing date: {date_str}")
                                
                                values = []
                                for col_idx in range(2, min(50, len(row))):
                                    if col_idx < len(row):
                                        val = row.iloc[col_idx]
                                        try:
                                            if pd.isna(val):
                                                values.append(0.0)
                                            else:
                                                if isinstance(val, str) and val in ['A', 'E', 'F', 'N']:
                                                    break
                                                values.append(float(val))
                                        except (ValueError, TypeError):
                                            values.append(0.0)
                                    else:
                                        values.append(0.0)
                                
                                while len(values) < 48:
                                    values.append(0.0)
                                
                                data_dict[date_str] = values[:48]
                                logger.debug(f"Added {date_str}: 48 intervals, total energy: {sum(values[:48]):.3f} kWh")
                            
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing date in row {row_idx}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing row {row_idx}: {str(e)}")
                    continue
            
            logger.info(f"Converted {len(data_dict)} days from Excel NEM12 row format")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting Excel NEM12 row format: {str(e)}")
            return None

    def convert_excel_format(self, file_content, filename):
        """Convert Excel file with date and interval data - Enhanced to match original"""
        try:
            # Try reading first sheet
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0, header=None)
            
            # Check if it's already in NEM12-like format with 200/300 records
            if len(df.columns) > 2 and (df.iloc[0, 0] == 200 or df.iloc[0, 0] == "200"):
                return self.convert_existing_nem12_excel(df)
            
            # Try with headers for time-series format
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)
            
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
            logger.error(f"Error converting Excel format: {str(e)}")
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
    
    def convert_timeseries_format(self, file_content, filename):
        """Convert time-series CSV format"""
        try:
            # Read CSV from bytes
            df = pd.read_csv(io.BytesIO(file_content))
            
            datetime_col = None
            power_col = None
            
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
            
            # Look for energy/power columns
            energy_priority_keywords = ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'generated kwh']
            power_priority_keywords = ['average kw', 'kw', 'power']
            general_keywords = ['e', 'energy']
            
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
            
            if power_col is None:
                for keyword in general_keywords:
                    for col in df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower == keyword:
                            power_col = col
                            logger.info(f"Found energy column (general): '{col}'")
                            break
                    if power_col:
                        break
            
            if datetime_col is None or power_col is None:
                logger.warning(f"Could not find required columns in {filename}")
                logger.info(f"Datetime column: {datetime_col}, Power column: {power_col}")
                return self.convert_csv_format2(file_content, filename)
            
            logger.info(f"Using datetime column: '{datetime_col}', power column: '{power_col}'")
            
            # Parse datetime and extract date/time
            df['datetime_parsed'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['datetime_parsed'])
            
            if df.empty:
                logger.warning(f"No valid dates found in {filename}")
                return None
            
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['minute'] = df['datetime_parsed'].dt.minute
            
            # Convert to 30-minute intervals
            df['interval_index'] = df['hour'] * 2 + (df['minute'] // 30)
            
            # Clean power values
            df[power_col] = pd.to_numeric(df[power_col], errors='coerce').fillna(0)
            
            # Determine if values are energy or power
            col_lower = power_col.lower()
            is_energy_column = any(keyword in col_lower for keyword in ['kwh', 'energy', 'usage'])
            is_power_column = any(keyword in col_lower for keyword in ['kw', 'power']) and 'kwh' not in col_lower
            
            logger.info(f"Column '{power_col}' analysis: is_energy={is_energy_column}, is_power={is_power_column}")
            
            data_dict = {}
            
            # Group by date
            for date, group in df.groupby('date'):
                date_str = date.strftime("%Y%m%d")
                
                intervals = [0.0] * 48
                
                interval_groups = group.groupby('interval_index')[power_col].mean()
                
                for interval_idx, avg_value in interval_groups.items():
                    if 0 <= interval_idx < 48:
                        if is_energy_column:
                            energy_value = float(avg_value)
                        elif is_power_column:
                            energy_value = float(avg_value) * 0.5
                        else:
                            if 0.01 <= abs(avg_value) <= 1000:
                                energy_value = float(avg_value)
                            else:
                                energy_value = float(avg_value) * 0.5
                        
                        intervals[int(interval_idx)] = energy_value
                
                if any(val > 0 for val in intervals):
                    data_dict[date_str] = intervals
            
            logger.info(f"Converted {len(data_dict)} days of time-series data")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting time-series format: {str(e)}")
            return None
    
    def convert_csv_format2(self, file_content, filename):
        """Convert CSV with date column and value columns"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Look for date column
            date_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['date', 'time', 'day', 'timestamp']):
                    date_col = col
                    break
            
            if date_col is None:
                date_col = df.columns[0]
            
            data_dict = {}
            for _, row in df.iterrows():
                try:
                    date_obj = pd.to_datetime(row[date_col])
                    date_str = self._format_date_yyyymmdd(date_obj)
                    
                    values = []
                    for col in df.columns:
                        if col != date_col:
                            val = row[col] if pd.notna(row[col]) else 0
                            try:
                                values.append(float(val))
                            except:
                                values.append(0.0)
                    
                    while len(values) < 48:
                        values.append(0.0)
                    
                    data_dict[date_str] = values[:48]
                    
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {str(e)}")
                    continue
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting CSV format2: {str(e)}")
            return None

    def detect_interval_energy_format(self, file_content, filename):
        """Detect interval energy format"""
        try:
            df_sample = pd.read_csv(io.BytesIO(file_content), nrows=50)
            
            has_kwh_value = 'kWh Value' in df_sample.columns
            has_period = 'Period' in df_sample.columns or any('time' in str(col).lower() for col in df_sample.columns)
            
            # Check total rows
            df_full = pd.read_csv(io.BytesIO(file_content))
            total_rows = len(df_full)
            has_many_intervals = total_rows > 1000
            
            kwh_reasonable = False
            if has_kwh_value:
                sample_values = pd.to_numeric(df_sample['kWh Value'], errors='coerce').dropna()
                if len(sample_values) > 0:
                    avg_val = sample_values.mean()
                    kwh_reasonable = 0.1 <= avg_val <= 500
            
            is_interval_energy_format = has_kwh_value and has_period and has_many_intervals and kwh_reasonable
            
            if is_interval_energy_format:
                logger.info(f"Detected interval energy format in {filename}")
                logger.info(f"  Rows: {total_rows}, Avg kWh: {avg_val:.2f}")
            
            return is_interval_energy_format
            
        except Exception as e:
            logger.warning(f"Error detecting interval energy format: {str(e)}")
            return False

    def convert_interval_energy_format(self, file_content, filename):
        """Convert interval energy format"""
        try:
            logger.info(f"Converting interval energy format: {filename}")
            
            df = pd.read_csv(io.BytesIO(file_content))
            
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info(f"Total intervals: {len(df)}")
            
            datetime_col = None
            energy_col = None
            
            datetime_candidates = ['Period', 'End Time', 'Start Time', 'DateTime', 'Time']
            for col in datetime_candidates:
                if col in df.columns:
                    datetime_col = col
                    logger.info(f"Found datetime column: '{col}'")
                    break
            
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
            
            df['date'] = df['datetime_parsed'].dt.date
            df['hour'] = df['datetime_parsed'].dt.hour
            df['minute'] = df['datetime_parsed'].dt.minute
            
            df['interval_index'] = df['hour'] * 2 + (df['minute'] // 30)
            
            df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce').fillna(0)
            
            logger.info(f"Energy column '{energy_col}' - using values directly as kWh")
            
            data_dict = {}
            
            for date, group in df.groupby('date'):
                date_str = date.strftime("%Y%m%d")
                
                intervals = [0.0] * 48
                
                for _, row in group.iterrows():
                    interval_idx = int(row['interval_index'])
                    kwh_value = row[energy_col]
                    
                    if 0 <= interval_idx < 48:
                        intervals[interval_idx] = float(kwh_value)
                
                if any(val > 0 for val in intervals):
                    data_dict[date_str] = intervals
                    daily_total = sum(intervals)
                    logger.debug(f"Date {date_str}: {len(group)} intervals, daily total: {daily_total:.1f} kWh")
            
            logger.info(f"Converted {len(data_dict)} days from interval energy format")
            return data_dict
            
        except Exception as e:
            logger.error(f"Error converting interval energy format: {str(e)}")
            return None

    def detect_and_convert_file(self, file_content, filename):
        """Enhanced detection with all format handlers - Match original priorities"""
        file_ext = Path(filename).suffix.lower()
        
        # Priority 1: Check for Input_1 horizontal NEM12 format
        if self.detect_input1_format(file_content, filename):
            logger.info(f"Using Input_1 horizontal NEM12 handler for {filename}")
            return self.convert_input1_horizontal_nem12_format(file_content, filename)
        
        # Priority 2: Check for Excel NEM12 row format (Input_4, Input_6, Input_7 style)
        if file_ext in ['.xlsx', '.xls'] and self.detect_excel_nem12_row_format(file_content, filename):
            logger.info(f"Using Excel NEM12 row format handler for {filename}")
            return self.convert_excel_nem12_row_format(file_content, filename)
        
        # Priority 3: Check for existing NEM12 format
        if file_ext in ['.csv', '.txt']:
            try:
                first_line = file_content.decode('utf-8').split('\n')[0].strip()
                if first_line.startswith('100,NEM12') or first_line.startswith('200,'):
                    logger.info(f"Detected existing NEM12 CSV format in {filename}")
                    return self.convert_csv_format2(file_content, filename)
            except:
                pass
        
        # Priority 4: Check for interval energy format
        if file_ext in ['.csv'] and self.detect_interval_energy_format(file_content, filename):
            logger.info(f"Using interval energy format handler for {filename}")
            return self.convert_interval_energy_format(file_content, filename)
        
        # Priority 5: Check for time-series format (CSV)
        if file_ext in ['.csv']:
            try:
                df_sample = pd.read_csv(io.BytesIO(file_content), nrows=10)
                
                has_datetime_col = any(any(keyword in str(col).lower() for keyword in 
                                        ['readingdatetime', 'read_datetime', 'datetime', 'timestamp', 'read datetime', 'endtime', 'local time']) 
                                    for col in df_sample.columns)
                has_power_col = any(any(keyword in str(col).lower() for keyword in 
                                    ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'kw', 'power']) and 
                                not any(exclude in str(col).lower() for exclude in 
                                        ['serial', 'nmi', 'id', 'identifier', 'point'])
                                for col in df_sample.columns)
                
                df_full = pd.read_csv(io.BytesIO(file_content))
                total_rows = len(df_full)
                has_multiple_timestamps = total_rows > 50
                
                if has_datetime_col and has_power_col and has_multiple_timestamps:
                    logger.info(f"Detected time-series format in {filename} ({total_rows} rows)")
                    return self.convert_timeseries_format(file_content, filename)
                    
            except Exception as e:
                logger.warning(f"Error detecting time-series format: {str(e)}")
        
        # Priority 6: Excel files (enhanced detection)
        if file_ext in ['.xlsx', '.xls']:
            try:
                # First check for NEM12-like format without headers
                df_noheader = pd.read_excel(io.BytesIO(file_content), header=None, nrows=5)
                
                # Check if first column contains 200/300 records (existing NEM12)
                has_200_300_records = any(str(val) in ['200', '300'] or val in [200, 300] for val in df_noheader.iloc[:, 0])
                
                if has_200_300_records:
                    logger.info(f"Detected existing NEM12 Excel format in {filename}")
                    return self.convert_excel_format(file_content, filename)
                
                # Check for time-series format with headers
                df_sample = pd.read_excel(io.BytesIO(file_content), nrows=10)
                
                has_datetime_col = any(any(keyword in str(col).lower() for keyword in 
                                        ['readingdatetime', 'read_datetime', 'datetime', 'timestamp', 'read datetime', 'endtime', 'local time']) 
                                    for col in df_sample.columns)
                has_power_col = any(any(keyword in str(col).lower() for keyword in 
                                    ['e (usage kwh)', 'kwh value', 'kwh', 'e kwh at meter', 'net kwh', 'kw', 'power']) and 
                                not any(exclude in str(col).lower() for exclude in 
                                        ['serial', 'nmi', 'id', 'identifier', 'point'])
                                for col in df_sample.columns)
                
                if has_datetime_col and has_power_col:
                    logger.info(f"Detected time-series format in Excel file {filename}")
                    return self.convert_excel_format(file_content, filename)
                
                # Regular Excel format (date + interval columns)
                logger.info(f"Detected regular Excel format in {filename}")
                return self.convert_excel_format(file_content, filename)
                    
            except Exception as e:
                logger.warning(f"Error detecting Excel format: {str(e)}")
        
        # Default fallback
        logger.info(f"Converting {filename} using fallback method")
        return self.convert_csv_format2(file_content, filename)

# Streamlit App
def main():
    st.set_page_config(
        page_title="NEM12 Converter",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ NEM12 Energy Data Converter")
    st.markdown("Convert various energy data formats to NEM12 standard format")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # NMI Configuration
    st.sidebar.subheader("NMI Settings")
    nmi_number = st.sidebar.text_input("NMI Number", value="6001425887")
    suffix = st.sidebar.text_input("Suffix", value="E1")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose energy data files",
        type=['csv', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Upload CSV, Excel, or text files containing energy interval data"
    )
    
    if uploaded_files:
        converter = StreamlitNEM12Converter()
        
        # Update configuration
        converter.default_nmi = nmi_number
        converter.default_suffix = suffix
        
        results = []
        
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            try:
                # Read file content
                file_content = uploaded_file.read()
                
                # Convert file
                with st.spinner(f"Converting {uploaded_file.name}..."):
                    data_dict = converter.detect_and_convert_file(file_content, uploaded_file.name)
                
                if data_dict and len(data_dict) > 0:
                    # Generate NEM12 content
                    nem12_content = converter._write_nem12_file(data_dict, nmi_number, suffix)
                    
                    if nem12_content:
                        # Display statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Days Converted", len(data_dict))
                        
                        with col2:
                            total_energy = sum(sum(day_data) for day_data in data_dict.values())
                            st.metric("Total Energy (kWh)", f"{total_energy:.1f}")
                        
                        with col3:
                            avg_daily = total_energy / len(data_dict) if len(data_dict) > 0 else 0
                            st.metric("Avg Daily (kWh)", f"{avg_daily:.1f}")
                        
                        # Store result for download
                        output_filename = f"{Path(uploaded_file.name).stem}_NEM12.csv"
                        results.append({
                            'filename': output_filename,
                            'content': nem12_content,
                            'original_name': uploaded_file.name
                        })
                        
                        st.success(f"✅ Successfully converted {uploaded_file.name}")
                        
                        # Show preview of first few days
                        with st.expander("Preview Data"):
                            preview_data = []
                            for date_str in sorted(list(data_dict.keys())[:5]):  # Show first 5 days
                                day_total = sum(data_dict[date_str])
                                preview_data.append({
                                    'Date': date_str,
                                    'Daily Total (kWh)': f"{day_total:.3f}",
                                    'First Interval': f"{data_dict[date_str][0]:.3f}",
                                    'Peak Interval': f"{max(data_dict[date_str]):.3f}"
                                })
                            
                            df_preview = pd.DataFrame(preview_data)
                            st.dataframe(df_preview, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Failed to generate NEM12 content for {uploaded_file.name}")
                
                else:
                    st.error(f"❌ No valid data found in {uploaded_file.name}")
                    st.info("Please check that your file contains energy interval data with proper date/time columns")
            
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Download section
        if results:
            st.header("📥 Download Converted Files")
            
            if len(results) == 1:
                # Single file download
                result = results[0]
                st.download_button(
                    label=f"📄 Download {result['filename']}",
                    data=result['content'],
                    file_name=result['filename'],
                    mime='text/csv',
                    use_container_width=True
                )
            
            else:
                # Multiple files - create zip
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for result in results:
                        zip_file.writestr(result['filename'], result['content'])
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label=f"📦 Download All Files ({len(results)} files)",
                    data=zip_buffer.getvalue(),
                    file_name="NEM12_converted_files.zip",
                    mime='application/zip',
                    use_container_width=True
                )
                
                # Individual download buttons
                st.subheader("Individual Downloads")
                for result in results:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(result['filename'])
                    with col2:
                        st.download_button(
                            label="📄 Download",
                            data=result['content'],
                            file_name=result['filename'],
                            mime='text/csv',
                            key=f"download_{result['filename']}"
                        )
    
    # Information section
    with st.expander("ℹ️ About NEM12 Format"):
        st.markdown("""
        **NEM12** is the Australian standard format for electricity interval data. This converter supports:
        
        **Input Formats:**
        - **Time-series CSV**: Files with datetime and energy/power columns
        - **Excel files**: With interval data in rows or time-series format
        - **Interval energy CSV**: Each row represents one 30-minute interval
        - **Horizontal NEM12**: Input_1 style with data spread across columns
        - **Existing NEM12**: Files already in partial NEM12 format
        
        **Output Format:**
        - Standard NEM12 CSV with 100, 200, 300, and 900 records
        - 48 intervals per day (30-minute intervals)
        - Proper date formatting (YYYYMMDD)
        - Quality flags and timestamps
        
        **Key Features:**
        - Automatic format detection
        - Energy/power unit conversion
        - Data validation and cleaning
        - Multiple file processing
        - Configurable NMI settings
        """)
    
    with st.expander("📋 Usage Instructions"):
        st.markdown("""
        1. **Upload Files**: Select one or more energy data files (CSV, Excel, or TXT)
        2. **Configure NMI**: Set your National Metering Identifier and suffix in the sidebar
        3. **Process**: The app will automatically detect the format and convert your data
        4. **Download**: Get your converted NEM12 files individually or as a ZIP archive
        
        **Supported File Types:**
        - CSV files with datetime and energy columns
        - Excel files with interval data
        - Files with kWh values per 30-minute interval
        - Existing partial NEM12 formats
        
        **Tips:**
        - Ensure your files have clear date/time columns
        - Energy values should be in kWh or kW (will be auto-converted)
        - For best results, use consistent date formats
        - The app handles various column naming conventions
        """)

if __name__ == "__main__":
    main()
