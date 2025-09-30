MAX_TURNS = 5
import requests

def ask_rag(query, history):
    """
    Function to send a query and history to the backend RAG service and get the response.
    
    Args:
        query (str): The user query.
        history (list): The conversation history.
        
    Returns:
        str: Response from the backend.
    """
    url = "http://localhost:8000/ask"
    # Only include the last MAX_TURNS turns from history
    recent_history = history[-MAX_TURNS:] if history else []
    payload = {
        "query": query,
        "history": recent_history
    }
    print("Payload being sent to backend:", payload)
    response = requests.post(url, json=payload)
    #print("Backend response text:", response.text)
    response.raise_for_status()
    json_response = response.json()
    print("Backend parsed JSON response:", json_response)
    return json_response
