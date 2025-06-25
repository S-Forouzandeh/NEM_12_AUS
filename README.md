# NEM12 File Converter

A powerful tool to convert various energy data formats to NEM12 format, with both command-line and web interface options.

## 🚀 Features

- **Multiple Input Formats**: CSV, Excel, text files with time series data
- **Automatic Detection**: Smart format detection for different data layouts
- **NEM12 Compliance**: Generates valid NEM12 files without headers (row 100 removed)
- **Web Interface**: Easy-to-use Streamlit web app
- **Batch Processing**: Handle multiple files at once
- **Quality Validation**: Built-in NEM12 format validation

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nem12-converter.git
cd nem12-converter
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🖥️ Usage

### Web Interface (Recommended)

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Interface

```bash
# Convert single file
python nem12_converter.py -i input_file.csv -o output_file.csv

# Convert folder (separate files)
python nem12_converter.py -i input_folder/ -o output_folder/ --separate

# Convert folder (combined file)
python nem12_converter.py -i input_folder/ -o combined_output.csv --combined

# Batch by NMI
python nem12_converter.py -i input_folder/ -o output_folder/ --batch
```

## 📁 File Structure

```
nem12-converter/
├── nem12_converter.py    # Main converter module
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Supported Input Formats

- **Time Series Data**: Files with datetime and energy reading columns
- **AGL DETAILED**: Retailer export format
- **SRC CONNECTION POINT**: Utility format
- **Standard Interval**: Structured interval data
- **Multi-Column Energy**: Wide format with multiple energy columns
- **Excel Workbooks**: Multiple sheets supported
- **Existing NEM12**: For validation and merging

## 📊 Output

- Generates valid NEM12 format files
- Header row (100) removed for platform compatibility
- Both CSV and DAT formats created
- Automatic validation included

## 🛠️ Configuration

The converter automatically detects input formats and handles:
- NMI extraction from filenames or data
- Date/time parsing in multiple formats
- Quality flag mapping
- Interval length detection (5, 15, 30 minute intervals)

## 📝 Example

Input file with time series data:
```csv
Date,Time,Reading,Quality
1/01/2023,00:30,1.234,A
1/01/2023,01:00,1.456,A
```

Output NEM12 format:
```csv
200,1234567890,E1,1,E1,N,,KWH,30,20231201
300,20230101,1.234,1.456,...,A
900
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- The converter removes header rows (record type 100) from output files for platform compatibility
- Input files can contain headers during processing
- All temporary files are automatically cleaned up
- Large files are processed efficiently with progress tracking

## 🆘 Support

If you encounter any issues:
1. Check the processing logs in the web interface
2. Verify your input file format
3. Ensure all required dependencies are installed
4. Open an issue on GitHub with sample data (anonymized)

---

**Built with Python, Pandas, and Streamlit** ⚡
