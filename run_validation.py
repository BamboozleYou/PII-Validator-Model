#!/usr/bin/env python3
"""
Targeted PII Validation Runner Script
Run this script to validate PII classifications against your specific types
"""

import sys
import os
from datetime import datetime
from pii_validator import TargetedPIIValidator

def main():
    print("🎯 Targeted PII Validation System")
    print("=" * 50)
    
    # ⚙️ CONFIGURATION - EDIT THESE SETTINGS ⚙️
    # ===========================================
    
    # 📄 CHANGE THIS TO YOUR CSV FILE:
    INPUT_FILE = "pii_dataset_test.csv"   # <-- PUT YOUR FILE NAME HERE
    
    # 🤖 Model settings (usually don't need to change):
    MODEL_URL = "http://localhost:11434"  # Ollama default
    MODEL_NAME = "deepseek-r1:1.5b"     # DeepSeek-R1 model
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f"targeted_validation_results_{timestamp}.csv"
    
    print(f"📊 Input file: {INPUT_FILE}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🌐 Model URL: {MODEL_URL}")
    print()
    
    # Show expected PII types
    pii_types = [
        "Address", "Address 2", "Age", "Bank account", "City", "Country",
        "Date of birth", "Drivers license number", "Email address", 
        "First name", "Full name", "Insurance number", "Last name",
        "Medical records", "Medicare ID", "National Identifier", 
        "Phone", "Postal Code", "NOT_PII"
    ]
    print("🏷️  Supported PII Types:")
    for i, pii_type in enumerate(pii_types, 1):
        print(f"   {i:2d}. {pii_type}")
    print()
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found!")
        print("Please ensure your CSV file is in the current directory.")
        print("\nExpected CSV format:")
        print("Table Name,Column Name,Column Datatype,Column Length,Guessed Classification")
        sys.exit(1)
    
    # Initialize targeted validator
    print("🔧 Initializing targeted validator...")
    validator = TargetedPIIValidator(
        model_url=MODEL_URL,
        model_name=MODEL_NAME
    )
    
    # Test model connection
    print("🔌 Testing DeepSeek-R1 model connection...")
    if not validator.test_connection():
        print("❌ Failed to connect to DeepSeek-R1 model!")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Download DeepSeek-R1: ollama pull deepseek-r1")
        print("3. Verify the model is loaded: ollama list")
        print("4. Check the MODEL_URL and MODEL_NAME settings above")
        print(f"5. Try accessing {MODEL_URL} in your browser")
        sys.exit(1)
    
    print("✅ DeepSeek-R1 model connection successful!")
    print()
    
    # Process the spreadsheet
    try:
        print("🚀 Starting targeted PII validation process...")
        print("📋 The system will classify each column and compare with your guess...")
        print("⚡ Responses: True Positive | Negative | Unsure")
        print("This may take a few minutes depending on your data size...")
        print()
        
        results = validator.process_spreadsheet(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE
        )
        
        # Display results summary
        print("\n" + "=" * 60)
        print("🎯 TARGETED VALIDATION SUMMARY")
        print("=" * 60)
        
        total_rows = len(results)
        true_positives = (results['Validation_Result'] == 'True Positive').sum()
        negatives = (results['Validation_Result'] == 'Negative').sum()
        unsure = (results['Validation_Result'] == 'Unsure').sum()
        errors = (results['Validation_Result'] == 'Error').sum()
        
        print(f"✅ Total rows processed: {total_rows}")
        print(f"🎯 True Positives: {true_positives} ({true_positives/total_rows*100:.1f}%)")
        print(f"❌ Negatives: {negatives} ({negatives/total_rows*100:.1f}%)")
        print(f"❓ Unsure: {unsure} ({unsure/total_rows*100:.1f}%)")
        if errors > 0:
            print(f"⚠️  Errors: {errors} ({errors/total_rows*100:.1f}%)")
        print(f"📄 Results saved to: {OUTPUT_FILE}")
        
        # Show accuracy metrics
        accuracy = true_positives / total_rows * 100
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Agreement Rate: {accuracy:.1f}%")
        print(f"   Disagreement Rate: {negatives/total_rows*100:.1f}%")
        print(f"   Uncertainty Rate: {unsure/total_rows*100:.1f}%")
        
        # Show specific disagreements (Negatives)
        if negatives > 0:
            print(f"\n❌ DISAGREEMENTS ({negatives} items):")
            print("-" * 60)
            
            negative_items = results[results['Validation_Result'] == 'Negative'].head(10)
            
            for idx, row in negative_items.iterrows():
                table_col = f"{row['Table Name']}.{row['Column Name']}"
                guessed = row['Guessed Classification']
                slm_guess = row['SLM_Classification']
                
                print(f"🔄 {table_col}")
                print(f"   Your Guess: {guessed}")
                print(f"   SLM Says: {slm_guess}")
                print(f"   Reason: {row['Reasoning']}")
                print()
            
            if negatives > 10:
                print(f"... and {negatives - 10} more disagreements (see full results in CSV)")
        
        # Show uncertain classifications
        if unsure > 0:
            print(f"\n❓ UNCERTAIN CLASSIFICATIONS ({unsure} items):")
            print("-" * 60)
            
            unsure_items = results[results['Validation_Result'] == 'Unsure'].head(5)
            
            for idx, row in unsure_items.iterrows():
                table_col = f"{row['Table Name']}.{row['Column Name']}"
                guessed = row['Guessed Classification']
                
                print(f"❓ {table_col}")
                print(f"   Your Guess: {guessed}")
                print(f"   Reason: {row['Reasoning']}")
                print()
            
            if unsure > 5:
                print(f"... and {unsure - 5} more uncertain items (see full results in CSV)")
        
        # Show validation insights
        print(f"\n💡 INSIGHTS:")
        print("-" * 30)
        if true_positives >= total_rows * 0.8:
            print("✓ High agreement - Your initial classifications are very good!")
        elif true_positives >= total_rows * 0.6:
            print("✓ Good agreement - Most classifications are accurate")
        else:
            print("⚠️  Low agreement - Consider reviewing your classification approach")
            
        if negatives > 0:
            print(f"✓ Found {negatives} potential classification improvements")
        if unsure > total_rows * 0.2:
            print("⚠️  Many uncertain classifications - consider more descriptive column names")
        if unsure == 0 and negatives == 0:
            print("🎉 Perfect! All classifications validated successfully")
        
        print("\n🎉 Targeted PII validation completed!")
        print(f"📂 Open '{OUTPUT_FILE}' to review all results")
        print("\n📋 Output CSV columns:")
        print("  - Validation_Result: True Positive | Negative | Unsure")
        print("  - SLM_Classification: SLM's classification (for Negative cases)")
        print("  - Reasoning: Explanation of the decision")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("\nPlease check:")
        print("1. Your CSV file format matches the expected columns:")
        print("   Table Name,Column Name,Column Datatype,Column Length,Guessed Classification")
        print("2. DeepSeek-R1 model is still running and accessible")
        print("3. You have write permissions in this directory")
        print("4. Your CSV file is properly formatted")
        sys.exit(1)

def show_classification_examples():
    """Show examples of how the system classifies"""
    print("\n🎯 CLASSIFICATION EXAMPLES:")
    print("=" * 40)
    
    examples = [
        ("email_address", "Email address", "True Positive", "Clear email field"),
        ("user_email", "Phone", "Negative: Email address", "Mislabeled as phone"),
        ("contact_info", "Email address", "Unsure", "Could be email or phone"),
        ("first_name", "First name", "True Positive", "Clear first name field"),
        ("personal_data", "First name", "Unsure", "Too vague to determine"),
        ("ssn", "National Identifier", "True Positive", "Clear SSN field"),
        ("phone_number", "Phone", "True Positive", "Clear phone field"),
        ("user_id", "NOT_PII", "True Positive", "Non-personal identifier"),
    ]
    
    for column, guess, result, reason in examples:
        print(f"Column: {column:15} | Guess: {guess:18} | Result: {result:25} | {reason}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        show_classification_examples()
    else:
        main()
