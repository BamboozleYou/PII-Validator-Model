#!/usr/bin/env python3
"""
Targeted PII Validator for Specific Classification Types
Compares SLM predictions with existing guesses and provides targeted responses
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional

class TargetedPIIValidator:
    """Targeted PII Validator with specific classification types and response format"""
    
    def __init__(self, model_url: str = "http://localhost:11434", model_name: str = "deepseek-r1:8b"):
        self.model_url = model_url.rstrip('/')
        self.model_name = model_name
        self.api_endpoint = f"{self.model_url}/api/generate"
        
        # Specific PII types as defined by user
        self.pii_types = [
            "Address", "Address 2", "Age", "Bank account", "City", "Country",
            "Date of birth", "Drivers license number", "Email address", 
            "First name", "Full name", "Insurance number", "Last name",
            "Medical records", "Medicare ID", "National Identifier", 
            "Phone", "Postal Code", "NOT_PII"
        ]
    
    def get_targeted_prompt(self, table_name: str, column_name: str, data_type: str, 
                          column_length: str, guessed_classification: str) -> str:
        """Create a targeted prompt with comprehensive few-shot learning examples"""
        
        prompt = f"""You are a PII classification expert. Learn from these examples to classify database columns based on column names, then validate against existing guesses.

**VALID PII TYPES:**
- Address, Address 2, Age, Bank account, City, Country, Date of birth, Drivers license number
- Email address, First name, Full name, Insurance number, Last name, Medical records
- Medicare ID, National Identifier, Phone, Postal Code, NOT_PII

**LEARN FROM THESE EXAMPLES:**

🎯 TRUE POSITIVE Examples (Model agrees with guess):
Column: "email_address" | Guess: Email address | Decision: True Positive
Reasoning: Clear email field name matches the guess perfectly

Column: "first_name" | Guess: First name | Decision: True Positive  
Reasoning: Obvious first name field, guess is correct

Column: "phone_number" | Guess: Phone | Decision: True Positive
Reasoning: Unambiguous phone field, matches guess

Column: "date_of_birth" | Guess: Date of birth | Decision: True Positive
Reasoning: Clear birth date field, correct classification

Column: "user_id" | Guess: NOT_PII | Decision: True Positive
Reasoning: Technical identifier, not personal data, guess correct

Column: "ssn" | Guess: National Identifier | Decision: True Positive
Reasoning: SSN is clearly a national identifier, correct guess

Column: "social_security_number" | Guess: National Identifier | Decision: True Positive
Reasoning: Full SSN field name, definitely national identifier

Column: "employee_id" | Guess: NOT_PII | Decision: True Positive
Reasoning: Work identifier, not personal information

Column: "work_phone" | Guess: Phone | Decision: True Positive
Reasoning: Business phone number, still phone type

Column: "license_number" | Guess: Drivers license number | Decision: True Positive
Reasoning: Driver's license field, correct classification

🚫 NEGATIVE Examples (Model disagrees with guess):
Column: "user_email" | Guess: Phone | Decision: Negative: Email address
Reasoning: Contains "email" in name but guessed as phone - clearly an email field

Column: "contact_email" | Guess: Phone | Decision: Negative: Email address
Reasoning: "email" in column name indicates email address, not phone

Column: "home_address" | Guess: City | Decision: Negative: Address
Reasoning: Full address field, not just city information

Column: "customer_phone" | Guess: Email address | Decision: Negative: Phone
Reasoning: Contains "phone" in name, clearly a phone number field

Column: "birth_date" | Guess: Age | Decision: Negative: Date of birth
Reasoning: Birth date field, not age calculation

Column: "zip_code" | Guess: Address | Decision: Negative: Postal Code
Reasoning: ZIP codes are postal codes, not full addresses

❓ UNSURE Examples (Too ambiguous to determine):
Column: "contact_info" | Guess: Email address | Decision: Unsure
Reasoning: Could be email, phone, or address - too generic to determine

Column: "personal_data" | Guess: First name | Decision: Unsure  
Reasoning: Extremely vague, could contain any type of personal information

Column: "emergency_contact" | Guess: Phone | Decision: Unsure
Reasoning: Could be a name, phone number, or email - ambiguous

Column: "contact_person" | Guess: First name | Decision: Unsure
Reasoning: Could be a name, title, or identifier - unclear

Column: "user_info" | Guess: Full name | Decision: Unsure
Reasoning: Too generic, could contain various user data types

Column: "data_field" | Guess: NOT_PII | Decision: Unsure
Reasoning: Completely non-descriptive column name

**YOUR TASK:**
Based on the patterns you learned above, analyze this column:

Table: {table_name}
Column: {column_name}
Data Type: {data_type}
Column Length: {column_length}
Existing Guess: {guessed_classification}

**DECISION PROCESS:**
1. Look at the column name carefully
2. Does it clearly indicate a specific PII type? (like "email_address", "phone_number")
3. Compare your classification with the existing guess
4. If they match → True Positive
5. If they differ → Negative: [YOUR_CLASSIFICATION]  
6. If column name is too vague → Unsure

**RESPOND IN EXACT JSON FORMAT:**
{{
    "response": "True Positive" | "Negative: [PII_TYPE]" | "Unsure",
    "reasoning": "Your analysis of the column name and decision"
}}

Response:"""
        
        return prompt
    
    def test_connection(self) -> bool:
        """Test connection to the local model"""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": "Test",
                    "stream": False
                },
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def query_model(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Query the local model with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.05,  # Very low temperature for consistent results
                            "top_p": 0.8,
                            "repeat_penalty": 1.1
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    print(f"API error (attempt {attempt + 1}): {response.status_code}")
                    
            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        
        return None
    
    def parse_model_response(self, response: str) -> Dict:
        """Parse and validate model response"""
        try:
            # Extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                if 'response' in result and 'reasoning' in result:
                    response_text = result['response'].strip()
                    
                    # Validate response format
                    if response_text == "True Positive":
                        return {
                            "validation_result": "True Positive",
                            "slm_classification": None,
                            "reasoning": result['reasoning']
                        }
                    elif response_text.startswith("Negative:"):
                        # Extract the SLM's classification
                        slm_class = response_text.replace("Negative:", "").strip()
                        return {
                            "validation_result": "Negative",
                            "slm_classification": slm_class,
                            "reasoning": result['reasoning']
                        }
                    elif response_text == "Unsure":
                        return {
                            "validation_result": "Unsure", 
                            "slm_classification": None,
                            "reasoning": result['reasoning']
                        }
            
            # Fallback parsing if JSON is malformed
            response_lower = response.lower()
            if "true positive" in response_lower:
                return {
                    "validation_result": "True Positive",
                    "slm_classification": None,
                    "reasoning": "Parsed from text response"
                }
            elif "negative" in response_lower:
                return {
                    "validation_result": "Negative",
                    "slm_classification": "Unknown",
                    "reasoning": "Could not parse SLM classification"
                }
            elif "unsure" in response_lower:
                return {
                    "validation_result": "Unsure",
                    "slm_classification": None,
                    "reasoning": "Parsed from text response"
                }
            
            # Default fallback
            return {
                "validation_result": "Error",
                "slm_classification": None,
                "reasoning": f"Could not parse response: {response}"
            }
            
        except Exception as e:
            return {
                "validation_result": "Error",
                "slm_classification": None,
                "reasoning": f"Parse error: {str(e)}"
            }
    
    def validate_single_classification(self, table_name: str, column_name: str, 
                                     data_type: str, column_length: str, 
                                     guessed_classification: str) -> Dict:
        """Validate a single PII classification"""
        
        prompt = self.get_targeted_prompt(
            table_name, column_name, data_type, column_length, guessed_classification
        )
        
        response = self.query_model(prompt)
        
        if response is None:
            return {
                "validation_result": "Error",
                "slm_classification": None,
                "reasoning": "Model query failed"
            }
        
        return self.parse_model_response(response)
    
    def process_spreadsheet(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Process entire spreadsheet with targeted validation"""
        
        print(f"📖 Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        required_columns = ['Table Name', 'Column Name', 'Column Datatype', 'Column Length', 'Guessed Classification']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Initialize new columns
        df['Validation_Result'] = ''
        df['SLM_Classification'] = ''
        df['Reasoning'] = ''
        
        total_rows = len(df)
        print(f"🔍 Processing {total_rows} rows...")
        
        for idx, row in df.iterrows():
            print(f"Processing row {idx + 1}/{total_rows}: {row['Table Name']}.{row['Column Name']}")
            
            result = self.validate_single_classification(
                table_name=str(row['Table Name']),
                column_name=str(row['Column Name']),
                data_type=str(row['Column Datatype']),
                column_length=str(row['Column Length']),
                guessed_classification=str(row['Guessed Classification'])
            )
            
            # Update DataFrame
            df.loc[idx, 'Validation_Result'] = result['validation_result']
            df.loc[idx, 'SLM_Classification'] = result['slm_classification'] or ''
            df.loc[idx, 'Reasoning'] = result['reasoning']
            
            # Small delay to avoid overwhelming the model
            time.sleep(0.3)
        
        # Save results
        print(f"💾 Saving results to: {output_file}")
        df.to_csv(output_file, index=False)
        
        return df

# Test function
def test_targeted_validator():
    """Test the targeted validator with specific examples"""
    validator = TargetedPIIValidator()
    
    if not validator.test_connection():
        print("❌ Cannot connect to DeepSeek-R1 model")
        return
    
    # Test cases from the dataset
    test_cases = [
        ("users", "email_address", "varchar", "255", "Email address"),    # Should be True Positive
        ("users", "user_email", "varchar", "255", "Phone"),              # Should be Negative: Email address
        ("users", "personal_data", "text", "500", "First name"),         # Should be Unsure
        ("employees", "ssn", "varchar", "11", "National Identifier"),    # Should be True Positive
        ("users", "contact_info", "varchar", "150", "Email address"),    # Should be Unsure
    ]
    
    print("🧪 Testing targeted PII validator...")
    for table, column, dtype, length, guess in test_cases:
        print(f"\nTesting: {table}.{column} (Guess: {guess})")
        result = validator.validate_single_classification(table, column, dtype, length, guess)
        print(f"Result: {result['validation_result']}")
        if result['slm_classification']:
            print(f"SLM Classification: {result['slm_classification']}")
        print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_targeted_validator()
