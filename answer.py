import json
import openai
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

# Azure OpenAI Configuration from .env
AZURE_CHATOPENAI_API_KEY = os.getenv("AZURE_CHATOPENAI_API_KEY")
AZURE_CHATOPENAI_ENDPOINT = os.getenv("AZURE_CHATOPENAI_ENDPOINT")
AZURE_CHATOPENAI_DEPLOYMENT = os.getenv("AZURE_CHATOPENAI_DEPLOYMENT")
AZURE_CHATOPENAI_API_VERSION = "2023-05-15"

client = openai.AzureOpenAI(
    api_key=AZURE_CHATOPENAI_API_KEY,
    azure_endpoint=AZURE_CHATOPENAI_ENDPOINT,
    api_version=AZURE_CHATOPENAI_API_VERSION
)

def read_content_from_json(file_path: str) -> List[str]:
    """
    Read content from JSON file and extract all 'content' fields
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of content strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        contents = []
        # Handle both single object and list of objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'content' in item:
                    contents.append(item['content'])
        elif isinstance(data, dict) and 'content' in data:
            contents.append(data['content'])
        
        return contents
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def gpt_call(prompt: str, system_message: Optional[str] = None) -> str:
    """
    Make a call to GPT with the given prompt
    
    Args:
        prompt: The user prompt
        system_message: Optional system message
        
    Returns:
        GPT response text
    """
    try:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Debug: Print the deployment name and messages
        print(f"Using deployment: {AZURE_CHATOPENAI_DEPLOYMENT}")
        print(f"Messages: {len(messages)} message(s)")
        
        response = client.chat.completions.create(
            model=AZURE_CHATOPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {str(e)}")
        print(f"Deployment: {AZURE_CHATOPENAI_DEPLOYMENT}")
        print(f"Messages structure: {messages}")
        return ""

def summarize_and_process(json_file_path: str, initial_prompt: str) -> str:
    """
    Main function to read content, summarize it, and provide information requested in initial prompt
    
    Args:
        json_file_path: Path to JSON file containing content
        initial_prompt: The initial prompt/request that needs to be answered using the summary
        
    Returns:
        Final GPT response
    """
    # Step 1: Read content from JSON file
    contents = read_content_from_json(json_file_path)
    
    if not contents:
        print("No content found in JSON file")
        return ""
    
    print(f"Found {len(contents)} content items")
    
    # Step 2: Combine all content and summarize
    combined_content = "\n\n".join(contents)
    
    summarize_prompt = f"""
    Please provide a comprehensive summary of the following content. 
    Focus on the key points, main themes, and important details:
    
    {combined_content}
    """
    
    print("Summarizing content...")
    summary = gpt_call(summarize_prompt, "You are a helpful assistant that creates clear, concise summaries.")
    
    if not summary:
        print("Failed to get summary from GPT")
        return ""
    
    print(f"Summary generated ({len(summary)} characters)")
    
    # Step 3: Use the summary to answer the initial prompt
    final_prompt = f"""
    Based on the following summarized information:
    
    {summary}
    
    Please provide the information requested in this initial prompt: {initial_prompt}
    
    Use the summarized information above to give a comprehensive and accurate response.
    """
    
    print("Answering initial prompt using summary...")
    final_response = gpt_call(final_prompt, "You are a helpful assistant that provides detailed, accurate responses using the provided summarized information to answer the specific request.")
    
    return final_response

# Example usage
if __name__ == "__main__":
    # Example usage
    json_file = "relevant_chunks.json"  # or "chunks.json"
    initial_prompt = "what is the latest on fine-tuning techniques for machine learning"
    
    result = summarize_and_process(json_file, initial_prompt)
    
    if result:
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(result)
    else:
        print("Failed to process content")