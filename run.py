"""Run script for the Streamlit application."""
import subprocess
import sys
from pathlib import Path
import os
# import pytesseract

# Get the conda environment path
conda_prefix = os.getenv('CONDA_PREFIX')

# Update the path to where Tesseract is installed on your system
# pytesseract.pytesseract.tesseract_cmd = fr"{conda_prefix}\Library\bin\tesseract.exe"
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

def main():
    """Run the Streamlit application."""
    app_path = Path("src/app/main.py")
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)

    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
