import os
import sys
import subprocess

def main():
    if not os.path.exists("pii_validator_env"):
        subprocess.check_call([sys.executable, "-m", "venv", "pii_validator_env"])
    pip = os.path.join("pii_validator_env", "bin", "pip")
    subprocess.check_call([pip, "install", "--upgrade", "pip"])
    subprocess.check_call([pip, "install", "pandas", "requests", "openpyxl"])

if __name__ == "__main__":
    main()
