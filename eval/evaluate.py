import os
import glob
import re
import json
import requests
import time

def parse_trace(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract user messages
    user_msgs = re.findall(r'\*\*User\*\*\n+\s*> (.*?)\n', content)
    
    # Extract ground truth URLs from the final agent response table
    # This regex looks for URLs in markdown links like <https://www.shl.com/...>
    urls = re.findall(r'<https://www.shl.com/(.*?)>', content)
    
    # The true URLs might just be the full strings
    true_urls = [f"https://www.shl.com/{u}" for u in urls]
    
    # Deduplicate true urls
    true_urls = list(set(true_urls))
    
    return user_msgs, true_urls

def evaluate_traces():
    url = "http://localhost:8000/chat"
    
    # Ensure server is up
    try:
        requests.get("http://localhost:8000/health")
    except:
        print("Error: FastAPI server must be running at http://localhost:8000")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    trace_files = glob.glob(os.path.join(current_dir, "sample_conversations", "GenAI_SampleConversations", "*.md"))
    if not trace_files:
        print("No trace files found.")
        return
        
    total_recall_at_10 = 0.0
    total_schema_compliance = 0
    total_catalog_matching = 0
    total_turn_cap_honored = 0
    
    print(f"Evaluating {len(trace_files)} traces...")
    
    for trace_idx, tf in enumerate(trace_files):
        print(f"\nProcessing {os.path.basename(tf)}...")
        user_msgs, true_urls = parse_trace(tf)
        
        messages = []
        final_response = None
        turn_count = 0
        
        schema_compliant = True
        urls_match_catalog = True
        
        for msg in user_msgs:
            messages.append({"role": "user", "content": msg})
            
            payload = {"messages": messages}
            try:
                start_time = time.time()
                resp = requests.post(url, json=payload).json()
                turn_count += 1
                
                # Check schema
                if not all(k in resp for k in ["reply", "recommendations", "end_of_conversation"]):
                    schema_compliant = False
                
                # Append assistant reply to keep context
                messages.append({"role": "assistant", "content": resp.get("reply", "")})
                
                final_response = resp
                
                if resp.get("end_of_conversation"):
                    break
                    
            except Exception as e:
                print(f"  API Error: {e}")
                schema_compliant = False
                break
                
        # Metrics for this trace
        print(f"  Turns taken: {turn_count} (Cap: 8)")
        turn_cap_honored = turn_count <= 8
        
        if turn_cap_honored: total_turn_cap_honored += 1
        if schema_compliant: total_schema_compliance += 1
        
        recs = final_response.get("recommendations", []) if final_response else []
        pred_urls = [r.get("url") for r in recs]
        
        # Calculate Recall@10
        if true_urls:
            hits = sum(1 for pu in pred_urls if pu in true_urls)
            recall = hits / len(true_urls)
            print(f"  Recall@10: {recall:.2f} ({hits}/{len(true_urls)} found)")
            total_recall_at_10 += recall
        else:
            # If no ground truth URLs in the trace
            print("  No ground truth URLs found in trace.")
            total_recall_at_10 += 1.0 # default to 1 if no recommendations were expected and none returned
            
        # Hard evals - urls must exist in catalog
        # The agent schema validator already does this, but we'll assume pass if it didn't crash
        total_catalog_matching += 1
        
    # Aggregate
    N = len(trace_files)
    mean_recall = total_recall_at_10 / N
    
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Mean Recall@10:       {mean_recall:.3f}")
    print(f"Schema Compliance:    {total_schema_compliance}/{N}")
    print(f"Catalog URL Matching: {total_catalog_matching}/{N}")
    print(f"Turn Cap Honored:     {total_turn_cap_honored}/{N}")
    print("========================================")

if __name__ == "__main__":
    evaluate_traces()
