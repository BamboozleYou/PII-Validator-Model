#!/usr/bin/env python3
"""
Setup script for PII Validator
Run this once to set up your environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 PII Validator Setup")
    print("=" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python packages"):
        sys.exit(1)
    
    # Create sample data file if it doesn't exist
    if not os.path.exists("pii_test_data.csv"):
        print("📄 Creating sample test data file...")
        sample_data = """Table Name,Column Name,Data Type,Column Width,Guessed PII classification
users,user_id,INTEGER,11,NONE
users,email_address,VARCHAR,255,HIGH
users,first_name,VARCHAR,50,MEDIUM
users,last_name,VARCHAR,50,MEDIUM
users,phone_number,VARCHAR,15,HIGH
users,date_of_birth,DATE,10,SENSITIVE
employees,ssn,VARCHAR,11,SENSITIVE
employees,salary,DECIMAL,10,SENSITIVE
customers,credit_card_number,VARCHAR,19,SENSITIVE
customers,credit_card_last4,VARCHAR,4,MEDIUM"""
        
        with open("pii_test_data.csv", "w") as f:
            f.write(sample_data)
        print("✅ Sample data file created: pii_test_data.csv")
    else:
        print("✅ Test data file already exists")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Make sure your local model is running (Ollama, LM Studio, etc.)")
    print("2. Edit the model settings in run_validation.py if needed")
    print("3. Run the validation:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
        print("   python run_validation.py")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
        print("   python run_validation.py")

if __name__ == "__main__":
    main()