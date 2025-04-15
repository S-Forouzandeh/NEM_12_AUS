import streamlit as st
import pandas as pd
import os
import io
import tempfile
import logging
import zipfile
from datetime import datetime
import nem12_converter as nc  # Import your module

# Setup logging
def setup_logging():
    """Set up and return a configured logger for the Streamlit app."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
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
    """Save uploaded files to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return temp_dir, saved_paths

# Function to process files
def process_files(input_dir, output_file, logger, batch_per_nmi):
    """Process files using the nem12_converter module."""
    success = nc.process_folder(input_dir, output_file, logger, batch_per_nmi=batch_per_nmi)
    return success

# Streamlit UI setup
st.set_page_config(
    page_title="NEM12 File Converter",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ NEM12 File Converter")
st.write("""
Upload your CSV, Excel, or text files to convert them to NEM12 format. 
This app handles time series data and can export either a single NEM12 file or separate files per NMI.
""")

# File upload
uploaded_files = st.file_uploader("Upload your files", 
                                 type=["csv", "xlsx", "xls", "txt"], 
                                 accept_multiple_files=True)

# Output options
st.subheader("Output Options")
col1, col2 = st.columns(2)
with col1:
    batch_per_nmi = st.checkbox("Create separate files for each NMI", value=False)
with col2:
    output_filename = st.text_input("Output filename (without extension)", 
                                    value=f"NEM12_Output_{datetime.now().strftime('%Y%m%d')}")

# Process button
if st.button("Process Files") and uploaded_files:
    # Set up logging
    logger, log_stream = setup_logging()
    
    # Create progress bar
    progress_text = "Processing files..."
    progress_bar = st.progress(0)
    
    # Process files
    with st.spinner(progress_text):
        # Save uploaded files
        temp_dir, saved_files = save_uploaded_files(uploaded_files)
        
        # Set output file
        temp_output_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_output_dir, f"{output_filename}.csv")
        
        # Process the files
        st.info(f"Processing {len(saved_files)} file(s)...")
        success = process_files(temp_dir, output_file, logger, batch_per_nmi)
        
        # Update progress
        progress_bar.progress(100)
    
    # Display results
    if success:
        st.success("✅ Processing completed successfully!")
        
        # Create a ZIP of output files to download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files in temp_output_dir
            for root, _, files in os.walk(temp_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_output_dir)
                    zip_file.write(file_path, arcname)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Offer download button
        st.download_button(
            label="Download NEM12 Files",
            data=zip_buffer,
            file_name=f"{output_filename}.zip",
            mime="application/zip"
        )
    else:
        st.error("❌ Processing failed. Check the logs for details.")
    
    # Display logs
    with st.expander("View processing logs", expanded=not success):
        st.code(log_stream.getvalue(), language="text")

# Footer
st.markdown("---")
st.markdown("### How to use this converter")
st.markdown("""
1. **Upload files** - Select multiple CSV, Excel, or text files containing time series data or NEM12 formatted data
2. **Configure options** - Choose to create a single file or separate files per NMI
3. **Process** - Click the Process Files button to start conversion
4. **Download** - Once processing is complete, download the resulting NEM12 files as a ZIP
""")

st.markdown("### Supported Input Formats")
st.markdown("""
- **Time Series Data** - Files with date/time columns and corresponding energy readings
- **Existing NEM12 Files** - Already in NEM12 format for merging or validation
- **Excel Workbooks** - With time series data in one or more sheets
""")