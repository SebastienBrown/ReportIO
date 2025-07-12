from flask import Flask, jsonify, request
from flask_cors import CORS
from search import google_search
import os
from dotenv import load_dotenv
import json
from similarity import rank_websites
from scrapenew import WebScrapingService
from chunkingnew import TextProcessingService
from answer import summarize_and_process
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classifier.testOutput import classifier

load_dotenv(override=True)

# Replace these with your actual API key and Custom Search Engine (CSE) ID
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/response', methods=['GET','POST'])
def response():
     
    if request.method == 'POST':
        data = request.get_json()
        message = data.get('message', 'No message provided')
    else:
        message = request.args.get('message', 'No message provided')
    
    query=message
    print("The message received from the frontend is ",query)

    #classifierOutput=[0,0]
    classifierOutput=classifier(query)
    print(classifierOutput)

    #override to ensure we always complete an internet search while testing the tool
    #in future, will directly use the classifier output to determine nature of response
    classifierOutput[0]=="RAG"

    if classifierOutput[0] =="GPT":
       return

    results = google_search(query, API_KEY, CSE_ID)

    # Save to JSON file
    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to search_results.json")
    
    #with open("search_results.json", "r", encoding="utf-8") as f:
        #results = json.load(f)
    
    sorted_websites=rank_websites(query,results)
    print(sorted_websites)

    # Save to JSON file
    with open("sorted_search_results.json", "w", encoding="utf-8") as f:
        json.dump(sorted_websites, f, indent=2, ensure_ascii=False)
    
    #with open("sorted_search_results.json", "r", encoding="utf-8") as f:
        #sorted_websites = json.load(f)

    service=WebScrapingService()

    scrapedData=[]
    for i in range(min(len(sorted_websites),3)):

        data=service.scrape(sorted_websites[i]["link"])

        scrapedData.append({
            'url': data.url,
            'title': data.title,
            'content': data.content,
            'word_count': data.word_count,
            'success': data.success,
            'scraped_at': data.scraped_at,
            'error_message': data.error_message
        })
    
    print(scrapedData)

    with open("scrapedData.json", "w", encoding="utf-8") as f:
        json.dump(scrapedData, f, indent=2, ensure_ascii=False)
    
    #with open("scrapedData.json", "r", encoding="utf-8") as f:
        #scrapedData = json.load(f)
    
    # Initialize service
    service = TextProcessingService(chunk_size=500, overlap=50, max_workers=3)
    
    # Process documents
    processing_result = service.process_batch(scrapedData)
    print("Processing result:", processing_result)
    
    # Search and save
    if processing_result['success']:
        search_result = service.search_and_save(
            query=query,
            top_k=5,
            output_dir="."
        )
        print("Search result:", search_result)
    
        # Check if it succeeded
    if search_result.get("success"):
        # Get the full results data (what would have gone into best_chunks.json)
        results_data = search_result["results_data"]
    
    # Get stats
    stats = service.get_service_stats()
    print("Service stats:", stats)

    print("DONE")

    #with open("best_chunks.json", "r", encoding="utf-8") as f:
        #bestChunks = json.load(f)

    resultFinal = summarize_and_process(results_data, query)

    print("Result Final is ",resultFinal)

    return jsonify({
        'reply': f'You said: {resultFinal}'
    })

if __name__ == '__main__':
    app.run(debug=True)