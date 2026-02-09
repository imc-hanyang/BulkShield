
import requests
import json
import csv
import datetime

# --- Configuration from js/firebase-config.js ---
API_KEY = "AIzaSyByyzGjMK1okYuSWY4Jvh-prZkavLq0Pkk"
PROJECT_ID = "srt-export-test"
ADMIN_EMAIL = "imc@google.com"
ADMIN_PWD = "1234qwer"

COLLECTION_BOOKING = "booking_verification_results"
COLLECTION_LLM = "llm_analysis_results"

AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
FIRESTORE_BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"

def authenticate():
    """Authenticates with Firebase Auth using email/password."""
    payload = {
        "email": ADMIN_EMAIL,
        "password": ADMIN_PWD,
        "returnSecureToken": True
    }
    try:
        response = requests.post(AUTH_URL, json=payload)
        response.raise_for_status()
        auth_data = response.json()
        return auth_data["idToken"]
    except requests.exceptions.RequestException as e:
        print(f"Authentication Failed: {e}")
        if response is not None:
             print(f"Response: {response.text}")
        return None

def fetch_documents(id_token, collection_name):
    """Fetches all documents from a Firestore collection."""
    url = f"{FIRESTORE_BASE_URL}/{collection_name}"
    headers = {
        "Authorization": f"Bearer {id_token}"
    }
    
    documents = []
    next_page_token = None
    
    while True:
        params = {}
        if next_page_token:
            params["pageToken"] = next_page_token
            
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "documents" in data:
                documents.extend(data["documents"])
            
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching collection {collection_name}: {e}")
            if response is not None:
                print(f"Response: {response.text}")
            break
            
    return documents

def parse_firestore_value(value):
    """Parses a Firestore value object into a Python native type."""
    if "stringValue" in value:
        return value["stringValue"]
    elif "integerValue" in value:
        return int(value["integerValue"])
    elif "doubleValue" in value:
        return float(value["doubleValue"])
    elif "booleanValue" in value:
        return value["booleanValue"]
    elif "timestampValue" in value:
        return value["timestampValue"]
    elif "mapValue" in value:
        return {k: parse_firestore_value(v) for k, v in value["mapValue"]["fields"].items()} if "fields" in value["mapValue"] else {}
    elif "arrayValue" in value:
        return [parse_firestore_value(v) for v in value["arrayValue"]["values"]] if "values" in value["arrayValue"] else []
    elif "nullValue" in value:
        return None
    return value

def flatten_dict(d, parent_key='', sep='_'):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_documents(docs, collection_name):
    """Processes raw Firestore documents into a flat list of dictionaries."""
    processed_data = []
    
    for doc in docs:
        if "fields" not in doc:
            continue
            
        fields = doc["fields"]
        parsed_doc = {k: parse_firestore_value(v) for k, v in fields.items()}
        
        # Add ID just in case
        parsed_doc["firestore_id"] = doc["name"].split("/")[-1]
        
        # Flatten for CSV
        flattened_doc = flatten_dict(parsed_doc)
        processed_data.append(flattened_doc)
        
    return processed_data

def save_to_csv(data, filename):
    """Saves a list of dictionaries to a CSV file."""
    if not data:
        print(f"No data to save for {filename}")
        return

    # Collect all possible keys
    keys = set()
    for item in data:
        keys.update(item.keys())
    
    # Sort keys for consistent output (optional but nice)
    sorted_keys = sorted(list(keys))
    
    # Common keys first if possible
    priority_keys = ["timestamp", "labeler_email", "task_type", "user_info_name"]
    for k in reversed(priority_keys):
        if k in sorted_keys:
            sorted_keys.insert(0, sorted_keys.pop(sorted_keys.index(k)))

    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} records to {filename}")
    except IOError as e:
        print(f"Error writing to CSV {filename}: {e}")

def main():
    print("Authenticating...")
    token = authenticate()
    if not token:
        return

    # 1. Booking Verification Results
    print(f"Fetching {COLLECTION_BOOKING}...")
    booking_docs = fetch_documents(token, COLLECTION_BOOKING)
    print(f"Found {len(booking_docs)} documents.")
    booking_data = process_documents(booking_docs, COLLECTION_BOOKING)
    save_to_csv(booking_data, "firebase_booking_data.csv")

    # 2. LLM Analysis Results
    print(f"Fetching {COLLECTION_LLM}...")
    llm_docs = fetch_documents(token, COLLECTION_LLM)
    print(f"Found {len(llm_docs)} documents.")
    llm_data = process_documents(llm_docs, COLLECTION_LLM)
    save_to_csv(llm_data, "firebase_llm_data.csv")
    
    print("Done.")

if __name__ == "__main__":
    main()
