# NEM12 File Converter

A Streamlit web application for converting various file formats (CSV, Excel, text) to NEM12 format for the Australian energy market.

## Features

- Convert time series data to NEM12 format
- Process multiple files at once
- Extract NMI identifiers automatically from file names or data
- Option to create a single NEM12 file or separate files per NMI
- Support for 15-minute and 30-minute interval data
- Validation of generated NEM12 files
- Comprehensive logging

## Usage

1. Upload one or more files (CSV, Excel, or text)
2. Choose output options (single file or separate files per NMI)
3. Click "Process Files" to convert
4. Download the resulting NEM12 file(s) as a ZIP

## Input File Requirements

The converter supports the following input formats:

- CSV or text files with time series data
- Excel work