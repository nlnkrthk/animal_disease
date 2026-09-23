"""
Convenience launcher to execute the Streamlit application from the project root.
Usage: python run_app.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "frontend" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
