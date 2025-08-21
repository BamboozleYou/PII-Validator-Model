#!/usr/bin/env python3
"""
Enhanced PII Validation Runner Script with DeepSeek-R1
Run this script to validate PII classifications in your spreadsheet
"""

import sys
import os
from datetime import datetime
from pii_validator import TargetedPIIValidator

def main():
    print("🔒 Enhanced Local PII Validation System")
    print("=" * 50)
    
    # Configuration - Edit these settings as needed
    MODEL_URL = "http://localhost:11434"  # Ollama default
    MODEL_NAME = "deepseek-r1:1.5b"     # DeepSeek-R1 model
    INPUT_FILE = "dataset1(in).csv"      # Your input file
    
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
        print("\nExpected CSV format:")
        print("Table Name,Column Name,Data Type,Column Width,Guessed PII classification")
        sys.exit(1)
    
    # Initialize enhanced validator
    print("🔧 Initializing enhanced validator...")
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
        print("\nAlternative models you can try:")
        print("  - ollama pull deepseek-r1:1.5b")
        print("  - ollama pull deepseek-r1:7b") 
        print("  - ollama pull deepseek-r1:32b")
        sys.exit(1)
    
    print("✅ DeepSeek-R1 model connection successful!")
    print()
    
    # Process the spreadsheet
    try:
        print("🚀 Starting enhanced PII validation process...")
        print("🧠 Using DeepSeek-R1 with improved prompt engineering...")
        print("This may take a few minutes depending on your data size...")
        print()
        
        results = validator.process_spreadsheet(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE
        )
        
        # Display results summary
        print("\n" + "=" * 60)
        print("📈 ENHANCED VALIDATION SUMMARY")
        print("=" * 60)
        
        total_rows = len(results)
        needs_review = results['Needs_Review'].sum()
        classification_changes = results['changed'].sum()
        avg_confidence = results['confidence'].astype(float).mean()
        high_confidence = (results['confidence'].astype(float) >= 0.8).sum()
        low_confidence = (results['confidence'].astype(float) < 0.5).sum()
        
        print(f"✅ Total rows processed: {total_rows}")
        print(f"⚠️  Rows needing review: {needs_review}")
        print(f"🔄 Classification changes: {classification_changes}")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        print(f"🟢 High confidence (≥0.8): {high_confidence}")
        print(f"🔴 Low confidence (<0.5): {low_confidence}")
        print(f"📄 Results saved to: {OUTPUT_FILE}")
        
        # Show classification distribution
        print(f"\n📊 CLASSIFICATION DISTRIBUTION:")
        print("-" * 40)
        classification_counts = results['validated_classification'].value_counts()
        for classification, count in classification_counts.items():
            percentage = (count / total_rows) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        # Show specific issues that need attention
        if needs_review > 0:
            print(f"\n🔍 ITEMS NEEDING REVIEW ({needs_review} items):")
            print("-" * 60)
            
            review_items = results[results['Needs_Review']].head(10)  # Show first 10
            
            for idx, row in review_items.iterrows():
                original = row['Guessed PII classification']
                validated = row['validated_classification']
                confidence = float(row['confidence'])
                table_col = f"{row['Table Name']}.{row['Column Name']}"
                
                if row['changed']:
                    print(f"🔄 {table_col}")
                    print(f"   Changed: {original} → {validated} (confidence: {confidence:.3f})")
                elif confidence < 0.7:
                    print(f"⚠️  {table_col}")
                    print(f"   Low confidence: {validated} (confidence: {confidence:.3f})")
                elif confidence == 0.0:
                    print(f"❌ {table_col}")
                    print(f"   Processing failed: {validated}")
                
                print(f"   Reason: {row['reasoning'][:100]}...")
                print()
            
            if needs_review > 10:
                print(f"... and {needs_review - 10} more items (see full results in CSV)")
        
        # Show improvement suggestions
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        if classification_changes > 0:
            print(f"✓ DeepSeek-R1 corrected {classification_changes} classifications")
        if high_confidence >= total_rows * 0.8:
            print("✓ High confidence results - model is performing well")
        if low_confidence > total_rows * 0.1:
            print("⚠️  Consider reviewing low confidence items manually")
        if needs_review == 0:
            print("🎉 Perfect! All classifications validated with high confidence")
        
        print("\n🎉 Enhanced PII validation completed successfully!")
        print(f"📂 Open '{OUTPUT_FILE}' to review all results")
        print("\n📋 CSV Columns explained:")
        print("  - validated_classification: DeepSeek-R1's assessment") 
        print("  - confidence: Model's confidence (0.0-1.0)")
        print("  - reasoning: Detailed explanation of the decision")
        print("  - changed: Whether classification was modified")
        print("  - Needs_Review: Flags items requiring manual review")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("\nPlease check:")
        print("1. Your CSV file format matches the expected columns:")
        print("   Table Name,Column Name,Data Type,Column Width,Guessed PII classification")
        print("2. DeepSeek-R1 model is still running and accessible")
        print("3. You have write permissions in this directory")
        print("4. Your CSV file is properly formatted (no special characters)")
        sys.exit(1)

def print_model_info():
    """Print information about DeepSeek-R1 models"""
    print("\n🤖 DEEPSEEK-R1 MODEL INFORMATION:")
    print("-" * 40)
    print("DeepSeek-R1 is a reasoning-focused language model that excels at")
    print("analytical tasks like PII classification. Available variants:")
    print()
    print("  deepseek-r1:1.5b  - Fastest, lowest memory (2GB RAM)")
    print("  deepseek-r1:7b    - Balanced performance (8GB RAM)")  
    print("  deepseek-r1:32b   - Best accuracy (32GB RAM)")
    print("  deepseek-r1:latest - Latest available version")
    print()
    print("To install: ollama pull deepseek-r1:7b")
    print("To run: ollama serve (in separate terminal)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print_model_info()
    else:
        main()
