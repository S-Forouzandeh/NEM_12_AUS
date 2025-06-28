import streamlit as st
import pandas as pd
import os
import io
import tempfile
import logging
import zipfile
from datetime import datetime
import shutil
import sys
import traceback
from pathlib import Path
import re

# Add the current directory to Python path to import nem12_converter
sys.path.append(os.path.dirname(__file__))

# Set up page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="NEM12 File Converter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import the converter with better error handling
try:
    import nem12_converter as nc
    CONVERTER_AVAILABLE = True
except ImportError as e:
    CONVERTER_AVAILABLE = False
    st.error(f"❌ Error importing nem12_converter: {e}")
    st.error("Please ensure nem12_converter.py is in the same directory as app.py")
    st.info("Upload both files to the same location for the app to work properly.")

# Session state initialization for better state management
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'output_files' not in st.session_state:
    st.session_state.output_files = []
if 'processing_logs' not in st.session_state:
    st.session_state.processing_logs = ""

def reset_session_state():
    """Reset session state for new processing"""
    st.session_state.processing_complete = False
    st.session_state.output_files = []
    st.session_state.processing_logs = ""

# Setup logging for Streamlit with better configuration
@st.cache_resource
def setup_streamlit_logging():
    """Set up and return a configured logger for the Streamlit app."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    
    # Create the logger
    logger = logging.getLogger("nem12_streamlit")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    return logger, log_formatter

def create_log_stream():
    """Create a new log stream for capturing logs"""
    log_stream = io.StringIO()
    return log_stream

def setup_logger_with_stream(logger, formatter, log_stream, debug_mode=False):
    """Setup logger with stream handler"""
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    # Create and add stream handler
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def sanitize_filename(filename):
    """Sanitize filename for cross-platform compatibility"""
    # Remove path traversal attempts and invalid characters
    filename = os.path.basename(filename)
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return filename

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return paths with better error handling."""
    temp_dir = tempfile.mkdtemp(prefix="nem12_input_")
    saved_paths = []
    errors = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Sanitize filename
            safe_filename = sanitize_filename(uploaded_file.name)
            
            # Add extension if missing
            if not any(safe_filename.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.txt']):
                # Try to determine extension from type
                if 'excel' in uploaded_file.type.lower():
                    safe_filename += '.xlsx'
                elif 'csv' in uploaded_file.type.lower() or 'text' in uploaded_file.type.lower():
                    safe_filename += '.csv'
                else:
                    safe_filename += '.csv'  # Default to CSV
            
            file_path = os.path.join(temp_dir, safe_filename)
            
            # Save file with error handling
            try:
                file_content = uploaded_file.getbuffer()
                with open(file_path, "wb") as f:
                    f.write(file_content)
                saved_paths.append(file_path)
                st.success(f"✅ Saved: {uploaded_file.name} → {safe_filename}")
            except Exception as write_error:
                errors.append(f"Failed to write {uploaded_file.name}: {write_error}")
                st.error(f"❌ Error saving file {uploaded_file.name}: {write_error}")
                
        except Exception as e:
            errors.append(f"Failed to process {uploaded_file.name}: {e}")
            st.error(f"❌ Error processing file {uploaded_file.name}: {e}")
    
    return temp_dir, saved_paths, errors

def validate_nem12_file_enhanced(file_path: str, logger: logging.Logger) -> dict:
    """Enhanced validation that returns detailed results"""
    result = {
        'is_valid': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        if not os.path.exists(file_path):
            result['errors'].append(f"File does not exist: {file_path}")
            return result

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            result['errors'].append("File is empty")
            return result

        # Basic structure validation
        first_line = lines[0].strip()
        first_record_type = first_line.split(',')[0] if ',' in first_line else first_line
        
        # Check start (should be 200 since we remove 100)
        if first_record_type != "200":
            result['warnings'].append(f"File starts with '{first_record_type}' instead of '200' (header row 100 was removed)")
        
        # Check end
        last_line = lines[-1].strip()
        last_record_type = last_line.split(',')[0] if ',' in last_line else last_line
        if last_record_type != "900":
            result['errors'].append(f"File must end with 900 record, found: {last_record_type}")

        # Count record types and validate structure
        record_counts = {'200': 0, '300': 0, '400': 0, '900': 0}
        problematic_300_records = 0
        total_energy = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            fields = line.split(',')
            if not fields:
                continue
                
            record_type = fields[0].strip()
            
            if record_type in record_counts:
                record_counts[record_type] += 1
                
                # Special validation for 300 records
                if record_type == "300":
                    if len(fields) < 50:
                        problematic_300_records += 1
                        result['warnings'].append(f"Line {line_num}: 300 record has {len(fields)} fields (expected ~51)")
                    elif len(fields) > 55:
                        problematic_300_records += 1
                        result['warnings'].append(f"Line {line_num}: 300 record has {len(fields)} fields (too many)")
                    
                    # Try to calculate energy
                    try:
                        if len(fields) >= 50:
                            interval_values = [float(f) for f in fields[2:50] if f.replace('.', '').replace('-', '').isdigit()]
                            total_energy += sum(interval_values)
                    except:
                        pass

        # Store statistics
        result['stats'] = {
            'record_counts': dict(record_counts),
            'problematic_300_records': problematic_300_records,
            'total_energy': total_energy,
            'file_size': os.path.getsize(file_path)
        }

        # Determine if valid
        has_basic_structure = (
            record_counts['200'] > 0 and
            record_counts['300'] > 0 and
            record_counts['900'] > 0
        )
        
        result['is_valid'] = has_basic_structure and len(result['errors']) == 0
        
        if result['is_valid']:
            logger.info(f"✅ Enhanced validation passed for {file_path}")
        else:
            logger.error(f"❌ Enhanced validation failed for {file_path}")
            
        return result

    except Exception as e:
        result['errors'].append(f"Validation error: {e}")
        logger.error(f"Error in enhanced validation for {file_path}: {e}")
        return result

def process_files_streamlit(input_dir, output_dir, logger):
    """Process files using the nem12_converter module with better error handling"""
    try:
        if not CONVERTER_AVAILABLE:
            logger.error("nem12_converter module not available")
            return False
            
        success = nc.process_folder(
            folder_path=input_dir,
            output_path=output_dir,
            logger=logger,
            batch_per_nmi=False,
            separate_files=True
        )
        return success
    except Exception as e:
        logger.error(f"Error in process_files_streamlit: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def collect_output_files(output_dir):
    """Collect all output files from the output directory."""
    output_files = []
    
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.csv', '.dat')):
                    file_path = os.path.join(root, file)
                    output_files.append(file_path)
    
    return output_files

# Main UI starts here
def main():
    # Title and description
    st.title("⚡ NEM12 File Converter")
    st.markdown("""
    Upload your CSV, Excel, or text files to convert them to NEM12 format. 
    Each input file will generate one NEM12 output file (**header row 100 removed** for platform compatibility).
    """)
    
    # Show converter status
    if not CONVERTER_AVAILABLE:
        st.error("⚠️ NEM12 converter module not loaded. Please check file setup.")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Processing options
        st.subheader("Processing Options")
        st.info("📄 Each input file → one NEM12 output file (header row 100 removed)")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            debug_mode = st.checkbox("Enable debug logging", value=False)
            enhanced_validation = st.checkbox("Enhanced validation", value=True, help="Recommended for better error detection")
            show_file_preview = st.checkbox("Show file preview", value=True)
            auto_clear_cache = st.checkbox("Auto-clear cache", value=True, help="Clear session state after processing")
        
        # Reset button
        if st.button("🔄 Reset Session", help="Clear all session data and start fresh"):
            reset_session_state()
            st.rerun()
    
    # File upload section
    st.header("📁 File Upload")
    
    # Clear files button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Files", help="Clear uploaded files"):
            reset_session_state()
            st.rerun()
    
    uploaded_files = st.file_uploader(
        "Choose your files to convert",
        type=["csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,
        help="Upload CSV, Excel, or text files containing time series energy data",
        key="file_uploader"
    )
    
    # Display uploaded files info
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        
        # Show file details
        with st.expander("📋 Uploaded Files Details", expanded=True):
            for i, file in enumerate(uploaded_files):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{file.name}**")
                with col2:
                    st.write(f"{file.size:,} bytes")
                with col3:
                    st.write(file.type or "Unknown")
                with col4:
                    # File preview
                    if show_file_preview:
                        try:
                            if file.type and 'excel' in file.type.lower():
                                preview_df = pd.read_excel(file, nrows=3)
                            else:
                                preview_df = pd.read_csv(file, nrows=3)
                            st.write(f"{len(preview_df.columns)} cols")
                        except:
                            st.write("Preview N/A")
        
        # Output configuration
        st.subheader("📤 Output Configuration")
        output_filename = st.text_input(
            "Output filename prefix:",
            value=f"NEM12_Output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Files will be named with this prefix"
        )

    # Processing section
    if uploaded_files and st.button("🚀 Convert to NEM12", type="primary", disabled=not CONVERTER_AVAILABLE):
        
        # Reset previous results
        reset_session_state()
        
        # Setup logging
        logger, formatter = setup_streamlit_logging()
        log_stream = create_log_stream()
        logger = setup_logger_with_stream(logger, formatter, log_stream, debug_mode)
        
        # Create containers for status updates
        status_container = st.container()
        
        with status_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
        
        temp_input_dir = None
        temp_output_dir = None
        
        try:
            # Step 1: Save uploaded files
            status_text.info("💾 Saving uploaded files...")
            progress_bar.progress(10)
            
            temp_input_dir, saved_files, save_errors = save_uploaded_files(uploaded_files)
            
            if save_errors:
                st.warning(f"⚠️ Some files had issues: {len(save_errors)} errors")
                with st.expander("File Save Errors"):
                    for error in save_errors:
                        st.error(error)
            
            if not saved_files:
                st.error("❌ No files were successfully saved")
                return
            
            logger.info(f"Saved {len(saved_files)} files to temporary directory")
            
            # Step 2: Create output directory
            status_text.info("📁 Creating output directory...")
            progress_bar.progress(20)
            
            temp_output_dir = tempfile.mkdtemp(prefix="nem12_output_")
            
            # Step 3: Process files
            status_text.info("⚙️ Processing files (generating separate NEM12 files without header row 100)...")
            progress_bar.progress(30)
            
            logger.info("Starting processing in separate files mode (header row 100 will be removed)")
            success = process_files_streamlit(
                input_dir=temp_input_dir,
                output_dir=temp_output_dir,
                logger=logger
            )
            
            progress_bar.progress(80)
            
            # Step 4: Collect results
            status_text.info("📦 Collecting output files...")
            
            output_files = collect_output_files(temp_output_dir)
            
            progress_bar.progress(90)
            
            if success and output_files:
                status_text.success("✅ Conversion completed successfully!")
                progress_bar.progress(100)
                
                # Store results in session state
                st.session_state.output_files = output_files
                st.session_state.processing_complete = True
                st.session_state.processing_logs = log_stream.getvalue()
                
                # Display results
                st.success(f"🎉 Successfully converted {len(uploaded_files)} input file(s) to {len(output_files)} NEM12 file(s)")
                
                # Show output files with enhanced validation
                with st.expander("📋 Generated Files", expanded=True):
                    validation_results = []
                    
                    for output_file in output_files:
                        filename = os.path.basename(output_file)
                        filesize = os.path.getsize(output_file)
                        
                        if enhanced_validation:
                            validation_result = validate_nem12_file_enhanced(output_file, logger)
                            validation_results.append(validation_result)
                            
                            is_valid = validation_result['is_valid']
                            validation_icon = "✅" if is_valid else "⚠️"
                            validation_text = "Valid" if is_valid else "Issues Found"
                            
                            # Show detailed info
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"{validation_icon} **{filename}**")
                            with col2:
                                st.write(f"{filesize:,} bytes")
                            with col3:
                                st.write(validation_text)
                            
                            # Show warnings/errors if any
                            if validation_result['warnings'] or validation_result['errors']:
                                with st.expander(f"Details for {filename}"):
                                    if validation_result['errors']:
                                        st.error("Errors:")
                                        for error in validation_result['errors']:
                                            st.write(f"- {error}")
                                    if validation_result['warnings']:
                                        st.warning("Warnings:")
                                        for warning in validation_result['warnings']:
                                            st.write(f"- {warning}")
                                    
                                    # Show stats
                                    stats = validation_result['stats']
                                    if stats:
                                        st.info(f"Records: {stats.get('record_counts', {})}")
                                        if stats.get('total_energy', 0) > 0:
                                            st.info(f"Total Energy: {stats['total_energy']:.1f} kWh")
                        else:
                            # Use simple validation
                            is_valid = nc.validate_nem12_file(output_file, logger) if CONVERTER_AVAILABLE else True
                            validation_icon = "✅" if is_valid else "❌"
                            validation_text = "Valid" if is_valid else "Invalid"
                            st.write(f"{validation_icon} **{filename}** - {filesize:,} bytes - {validation_text}")
                
                # Create download package
                status_text.info("🗜️ Creating download package...")
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for output_file in output_files:
                        filename = os.path.basename(output_file)
                        zip_file.write(output_file, filename)
                
                zip_buffer.seek(0)
                
                # Download button
                st.download_button(
                    label="📥 Download NEM12 Files (ZIP)",
                    data=zip_buffer,
                    file_name=f"{output_filename}.zip",
                    mime="application/zip",
                    type="primary"
                )
                
            else:
                st.error("❌ Conversion failed")
                if not output_files:
                    st.error("No output files were generated")
                
                logger.error(f"Processing failed. Success: {success}, Output files: {len(output_files)}")
            
        except Exception as e:
            st.error(f"❌ An error occurred during processing: {str(e)}")
            logger.error(f"Exception in main processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Store logs in session state
            st.session_state.processing_logs = log_stream.getvalue()
            
            # Cleanup temporary directories
            try:
                if temp_input_dir and os.path.exists(temp_input_dir):
                    shutil.rmtree(temp_input_dir, ignore_errors=True)
                if temp_output_dir and os.path.exists(temp_output_dir):
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
            except Exception as cleanup_error:
                logger.debug(f"Cleanup error: {cleanup_error}")
    
    # Show processing logs if available
    if st.session_state.processing_logs:
        with st.expander("📋 Processing Logs", expanded=not st.session_state.processing_complete):
            st.code(st.session_state.processing_logs, language="text")
    
    # Information section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload Files** - Select your CSV, Excel, or text files
        2. **Configure** - Adjust settings in the sidebar if needed
        3. **Convert** - Click convert (each input = one NEM12 output)
        4. **Download** - Get your NEM12 files as a ZIP package
        
        **Note**: Header rows (record type 100) are automatically removed for platform compatibility.
        """)
    
    with col2:
        st.markdown("### 📊 Supported Formats")
        st.markdown("""
        - **Time Series Data** - Date/time + energy readings
        - **Excel Interval Format** - Structured interval data
        - **AGL DETAILED** - Retailer export format  
        - **Standard Interval** - Various CSV structures
        - **Multi-Column Energy** - Wide format data
        - **Existing NEM12** - For conversion (header removed)
        
        **Output**: Valid NEM12 files without header row (100).
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        NEM12 Converter v2.1 | Enhanced Error Handling | Built with Streamlit<br>
        <small>⚠️ Header row (100) removed for platform compatibility</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
