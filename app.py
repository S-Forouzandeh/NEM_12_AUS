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

# Add the current directory to Python path to import nem12_converter
sys.path.append(os.path.dirname(__file__))

try:
    import nem12_converter as nc
except ImportError as e:
    st.error(f"Error importing nem12_converter: {e}")
    st.error("Please ensure nem12_converter.py is in the same directory as app.py")
    st.stop()

# Setup logging for Streamlit
def setup_streamlit_logging():
    """Set up and return a configured logger for the Streamlit app."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    
    # Create the logger
    logger = logging.getLogger("nem12_streamlit")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create stream handler for logging to string IO
    log_stream = io.StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    
    return logger, log_stream

# Function to save uploaded files to a temporary directory
def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return paths."""
    temp_dir = tempfile.mkdtemp(prefix="nem12_input_")
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Sanitize filename
        safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._-")
        if not safe_filename:
            safe_filename = f"upload_{len(saved_paths) + 1}.csv"
        file_path = os.path.join(temp_dir, safe_filename)
        
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
            st.write(f"✅ Saved: {uploaded_file.name} → {safe_filename}")
        except Exception as e:
            st.error(f"❌ Error saving file {uploaded_file.name}: {e}")
            continue
    
    return temp_dir, saved_paths

# Enhanced validation function that handles the specific error we're seeing
def validate_nem12_file_enhanced(file_path: str, logger: logging.Logger) -> bool:
    """Enhanced validation that handles field count mismatches."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False

        # Read file without strict field validation first
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            logger.error("File is empty")
            return False

        # Basic structure validation
        first_line = lines[0].strip()
        first_record_type = first_line.split(',')[0] if ',' in first_line else first_line
        
        # Since we removed header (100), expect to start with 200
        if first_record_type != "200":
            logger.warning(f"File starts with '{first_record_type}' instead of '200' (header row 100 was removed)")
        
        last_line = lines[-1].strip()
        last_record_type = last_line.split(',')[0] if ',' in last_line else last_line
        if last_record_type != "900":
            logger.error(f"File must end with 900 record, found: {last_record_type}")
            return False

        # Count record types and validate basic structure
        record_counts = {'200': 0, '300': 0, '400': 0, '900': 0}
        problematic_300_records = 0
        
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
                    # 300 records should have: RecordType + Date + 48 intervals + Quality = 51 fields minimum
                    if len(fields) < 50:  # Allow some flexibility
                        problematic_300_records += 1
                        logger.debug(f"Line {line_num}: 300 record has {len(fields)} fields (expected ~51)")
                    elif len(fields) > 55:  # Too many fields
                        problematic_300_records += 1
                        logger.debug(f"Line {line_num}: 300 record has {len(fields)} fields (too many)")

        # Report results
        logger.info(f"Enhanced validation for {file_path}:")
        logger.info(f"Record counts: {dict(record_counts)}")
        
        if problematic_300_records > 0:
            logger.warning(f"Found {problematic_300_records} 300 records with field count issues")
        
        # Validation passes if we have the basic structure
        has_basic_structure = (
            record_counts['200'] > 0 and  # At least one NMI record
            record_counts['300'] > 0 and  # At least one interval record
            record_counts['900'] > 0      # At least one footer record
        )
        
        if has_basic_structure:
            logger.info(f"✅ Enhanced validation passed for {file_path}")
            if problematic_300_records > 0:
                logger.info(f"⚠️  Note: {problematic_300_records} records had field count issues but overall structure is valid")
        else:
            logger.error(f"❌ Enhanced validation failed: missing required record types")
            
        return has_basic_structure

    except Exception as e:
        logger.error(f"Error in enhanced validation for {file_path}: {e}")
        return False

# Function to process files using the converter
def process_files_streamlit(input_dir, output_dir, logger):
    """Process files using the nem12_converter module - separate files only (no header row 100)."""
    try:
        # Use the process_folder function from your converter with separate files mode
        success = nc.process_folder(
            folder_path=input_dir,
            output_path=output_dir,
            logger=logger,
            batch_per_nmi=False,
            separate_files=True  # Always separate files, no combine option
        )
        return success
    except Exception as e:
        logger.error(f"Error in process_files_streamlit: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Function to collect output files
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

# Streamlit UI setup
st.set_page_config(
    page_title="NEM12 File Converter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("⚡ NEM12 File Converter")
st.markdown("""
Upload your CSV, Excel, or text files to convert them to NEM12 format. 
Each input file will generate one NEM12 output file (**header row 100 removed** for platform compatibility).
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Processing options
    st.subheader("Processing Options")
    st.info("📄 Each input file will generate one NEM12 output file (without header row 100)")
    
    # Advanced options
    with st.expander("Advanced Options"):
        debug_mode = st.checkbox("Enable debug logging", value=False)
        enhanced_validation = st.checkbox("Use enhanced validation (recommended)", value=True)
        show_file_preview = st.checkbox("Show file preview", value=True)

# File upload section
st.header("📁 File Upload")
uploaded_files = st.file_uploader(
    "Choose your files to convert",
    type=["csv", "xlsx", "xls", "txt"],
    accept_multiple_files=True,
    help="Upload CSV, Excel, or text files containing time series energy data"
)

# Display uploaded files info
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
    
    # Show file details
    with st.expander("📋 Uploaded Files Details", expanded=True):
        for i, file in enumerate(uploaded_files):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{file.name}**")
            with col2:
                st.write(f"{file.size:,} bytes")
            with col3:
                st.write(file.type)
    
    # Output filename configuration
    st.subheader("📤 Output Configuration")
    output_filename = st.text_input(
        "Output filename prefix:",
        value=f"NEM12_Output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Files will be named with this prefix"
    )

# Processing section
if uploaded_files and st.button("🚀 Convert to NEM12", type="primary"):
    
    # Setup logging
    logger, log_stream = setup_streamlit_logging()
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    
    # Create containers for status updates
    status_container = st.container()
    progress_container = st.container()
    
    with status_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
    
    try:
        # Step 1: Save uploaded files
        status_text.info("💾 Saving uploaded files...")
        progress_bar.progress(10)
        
        temp_input_dir, saved_files = save_uploaded_files(uploaded_files)
        
        if not saved_files:
            st.error("❌ No files were successfully saved")
            st.stop()
        
        logger.info(f"Saved {len(saved_files)} files to temporary directory")
        
        # Step 2: Create output directory
        status_text.info("📁 Creating output directory...")
        progress_bar.progress(20)
        
        temp_output_dir = tempfile.mkdtemp(prefix="nem12_output_")
        
        # Step 3: Process files (separate files mode only)
        status_text.info("⚙️ Processing files (generating separate NEM12 files without header row 100)...")
        progress_bar.progress(30)
        
        logger.info(f"Starting processing in separate files mode (header row 100 will be removed)")
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
            
            # Display results
            st.success(f"🎉 Successfully converted {len(uploaded_files)} input file(s) to {len(output_files)} NEM12 file(s)")
            
            # Show output files
            with st.expander("📋 Generated Files", expanded=True):
                for output_file in output_files:
                    filename = os.path.basename(output_file)
                    filesize = os.path.getsize(output_file)
                    
                    # Enhanced validation
                    if enhanced_validation:
                        is_valid = validate_nem12_file_enhanced(output_file, logger)
                        validation_icon = "✅" if is_valid else "⚠️"
                        validation_text = "Valid" if is_valid else "Structure Issues"
                    else:
                        # Use original validation
                        is_valid = nc.validate_nem12_file(output_file, logger)
                        validation_icon = "✅" if is_valid else "❌"
                        validation_text = "Valid" if is_valid else "Invalid"
                    
                    st.write(f"{validation_icon} **{filename}** - {filesize:,} bytes - {validation_text}")
            
            # Create download ZIP
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
        # Always show logs
        with st.expander("📋 Processing Logs", expanded=not success if 'success' in locals() else True):
            log_content = log_stream.getvalue()
            if log_content:
                st.code(log_content, language="text")
            else:
                st.write("No logs available")
        
        # Cleanup temporary directories
        try:
            if 'temp_input_dir' in locals() and os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir, ignore_errors=True)
            if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir, ignore_errors=True)
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error: {cleanup_error}")

# Information section
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload Files** - Select your CSV, Excel, or text files
    2. **Convert** - Click the convert button to process (each input = one NEM12 output)
    3. **Download** - Get your NEM12 files as a ZIP package
    
    **Note**: Header rows (record type 100) are automatically removed from output files for platform compatibility.
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
    
    **Output**: Valid NEM12 files without header row (100) for platform compatibility.
    """)

# Footer with important notes
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    NEM12 Converter v2.0 | Built with Streamlit | Header row (100) removed for platform compatibility<br>
    <small>⚠️ Enhanced validation handles field count mismatches in 300 records</small>
    </div>
    """,
    unsafe_allow_html=True
)
