import requests
import json
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_api():
    url = "http://localhost:8000/chat"
    
    # Wait for health check
    print("Waiting for server to be healthy...")
    while True:
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        
    messages = [
        {"role": "user", "content": "I am hiring a Java developer"}
    ]
    
    print("\n--- TURN 1 ---")
    print(f"User: {messages[0]['content']}")
    
    payload = {"messages": messages}
    response = requests.post(url, json=payload).json()
    
    print(f"Agent: {response['reply']}")
    print(f"Recommendations: {len(response['recommendations'])}")
    print(f"End of conversation: {response['end_of_conversation']}")
    
    messages.append({"role": "assistant", "content": response["reply"]})
    messages.append({"role": "user", "content": "Mid-level, around 4 years. They need to be good with stakeholders."})
    
    print("\n--- TURN 2 ---")
    print(f"User: {messages[-1]['content']}")
    
    payload = {"messages": messages}
    response = requests.post(url, json=payload).json()
    
    print(f"Agent: {response['reply']}")
    print(f"Recommendations: {len(response['recommendations'])}")
    for i, rec in enumerate(response['recommendations'], 1):
        print(f"  {i}. {rec['name']} ({rec['url']})")
    print(f"End of conversation: {response['end_of_conversation']}")

if __name__ == "__main__":
    test_api()
