import os
import sys
import json
import asyncio
import argparse
from datetime import datetime

# dotenv
from dotenv import load_dotenv
load_dotenv()

# load model proxy
from model_library import model_library_settings
model_library_settings.set()

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now relative imports will work
from custom_model import get_custom_model
from tool import tool_logger
from agent import agent_logger

async def main():
    # Get the custom model
    print(f"Loading model: {model_name}")
    custom_call = await get_custom_model(
        model_name,
        {
            "max_output_tokens": 65536,
            "temperature": 1
        },
        cheat=args.cheat, 
        log_level="INFO" if args.verbose else "WARNING"
    )
    
    # Run the test
    print(f"\nProcessing: {test_question}")
    try:
        result = await custom_call(test_question)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    # Format result
    formatted_result = {
        "question": test_question,
        "success": success,
        "model": model_name
    }
    if success:
        formatted_result["result"] = result
    else:
        formatted_result["error"] = error
    
    # Print status
    status = "✓ Success" if success else "✗ Failed"
    print(f"\n{status}: {test_question}")
    if success:
        print(f"Result: {json.dumps(result['llm_output'], indent=2)}")
    if not success:
        print(f"Error: {error}")
    
    return formatted_result

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test IOI Agent')
    parser.add_argument('--cheat', action='store_true', help='pass solutions to the agent')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--model', type=str, default='openai/gpt-5.1-codex', help='Model to use (default: codex)')
    parser.add_argument('--test', type=str, default='2024/sphinx', help='Test to run (e.g., 2024/sphinx)')
    args = parser.parse_args()
    
    # Get test from command line
    test_question = args.test

    # Use model from command line argument
    model_name = args.model

    # Run the async main function
    asyncio.run(main())
