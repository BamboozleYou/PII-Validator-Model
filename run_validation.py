#!/usr/bin/env python3
"""
PII Validation Runner Script
Run this script to validate PII classifications in your spreadsheet
"""

import sys
import os
from datetime import datetime
from pii_validator import LocalPIIValidator

def main():
    print("🔒 Local PII Validation System")
    print("=" * 50)
    
    # Configuration - Edit these settings as needed
    MODEL_URL = "http://localhost:11434"  # Ollama default
    MODEL_NAME = "gemma3:4b"           # Change to your model
    INPUT_FILE = "pii_test_data.csv"     # Your input file
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f"pii_validation_results_{timestamp}.csv"
    
    print(f"📊 Input file: {INPUT_FILE}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🌐 Model URL: {MODEL_URL}")
    print()
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file '{INPUT_FILE}' not found!")
        print("Please ensure your CSV file is in the current directory.")
        sys.exit(1)
    
    # Initialize validator
    print("🔧 Initializing validator...")
    validator = LocalPIIValidator(
        model_url=MODEL_URL,
        model_name=MODEL_NAME
    )
    
    # Test model connection
    print("🔌 Testing model connection...")
    if not validator.test_connection():
        print("❌ Failed to connect to local model!")
        print("\nTroubleshooting steps:")
        print("1. Make sure your model app (Ollama/LM Studio) is running")
        print("2. Verify the model is loaded and accessible")
        print("3. Check the MODEL_URL and MODEL_NAME settings above")
        print(f"4. Try accessing {MODEL_URL} in your browser")
        sys.exit(1)
    
    print("✅ Model connection successful!")
    print()
    
    # Process the spreadsheet
    try:
        print("🚀 Starting PII validation process...")
        print("This may take a few minutes depending on your data size...")
        print()
        
        results = validator.process_spreadsheet(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE
        )
        
        # Display results summary
        print("\n" + "=" * 50)
        print("📈 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_rows = len(results)
        needs_review = results['Needs_Review'].sum()
        classification_changes = (results['validated_classification'] != results['Guessed PII classification']).sum()
        avg_confidence = results['confidence'].astype(float).mean()
        
        print(f"✅ Total rows processed: {total_rows}")
        print(f"⚠️  Rows needing review: {needs_review}")
        print(f"🔄 Classification changes: {classification_changes}")
        print(f"🎯 Average confidence: {avg_confidence:.2f}")
        print(f"📄 Results saved to: {OUTPUT_FILE}")
        
        # Show specific issues that need attention
        if needs_review > 0:
            print(f"\n🔍 ITEMS NEEDING REVIEW ({needs_review} items):")
            print("-" * 50)
            
            review_items = results[results['Needs_Review']].head(10)  # Show first 10
            
            for idx, row in review_items.iterrows():
                original = row['Guessed PII classification']
                validated = row['validated_classification']
                confidence = float(row['confidence'])
                table_col = f"{row['Table Name']}.{row['Column Name']}"
                
                if original != validated:
                    print(f"🔄 {table_col}")
                    print(f"   Changed: {original} → {validated} (confidence: {confidence:.2f})")
                elif confidence < 0.7:
                    print(f"⚠️  {table_col}")
                    print(f"   Low confidence: {validated} (confidence: {confidence:.2f})")
                
                print(f"   Reason: {row['reasoning']}")
                print()
            
            if needs_review > 10:
                print(f"... and {needs_review - 10} more items (see full results in CSV)")
        
        print("\n🎉 PII validation completed successfully!")
        print(f"📁 Open '{OUTPUT_FILE}' to review all results")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("\nPlease check:")
        print("1. Your CSV file format matches the expected columns")
        print("2. Your local model is still running")
        print("3. You have write permissions in this directory")
        sys.exit(1)

if __name__ == "__main__":
    main()