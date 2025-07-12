import requests
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Replace these with your actual API key and Custom Search Engine (CSE) ID
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

def google_search(query, api_key, cse_id, num_results=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }

    print("API KEY IS ",API_KEY)
    print("CSE ID IS ",CSE_ID)
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    results = []
    data = response.json()
    for item in data.get("items", []):
        result = {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        }
        results.append(result)

    all_results = data.get("items", [])
    print(all_results)

    # Save to JSON file
    with open("ALL_search_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return results

# Example usage
if __name__ == "__main__":
    query = "what is the latest on fine-tuning techniques for machine learning"
    results = google_search(query, API_KEY, CSE_ID)

    # Save to JSON file
    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(results)} results to search_results.json")
