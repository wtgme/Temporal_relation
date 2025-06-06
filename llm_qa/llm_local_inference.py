# -*- coding: utf-8 -*-
from openai import OpenAI
from pydantic import BaseModel
import json, os, datetime, argparse
from typing import List

def load_data(jsonl_file_path):
    """
    Parse a JSONL file and extract messages from Azure OpenAI batch format
    Handle partially valid JSON by extracting valid objects and skipping invalid parts
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        
    Returns:
        dict: Dictionary with custom_id as key and messages list as value
    """
    results = {} 
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue            
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Extract custom_id
                custom_id = data.get('custom_id')
                if not custom_id:
                    print(f"Warning: Missing custom_id in line {line_num}")
                    continue
                
                # Navigate to messages - handle both request and response formats
                messages = None
                if 'body' in data:
                    # Response format
                    messages = data['body'].get('messages', [])
                elif 'request' in data:
                    # Request format
                    messages = data['request'].get('messages', [])
                elif 'messages' in data:
                    # Direct format
                    messages = data['messages']
                
                if messages:
                    results[custom_id] = messages
                else:
                    print(f"Warning: No messages found for {custom_id} in line {line_num}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line[:100]}...")
                continue
    
    return results


def inference(eid, message, client, model_name, json_schema):
    """Evaluate the model's ability to infer time for specific events"""
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=message,
            extra_body={"guided_json": json_schema},
        )
        
        reasoning_content = chat_response.choices[0].message.reasoning_content
        content = chat_response.choices[0].message.content

        print("reasoning_content:", reasoning_content)
        print("content:", content)

        # Format result to match Azure OpenAI structure
        result = {
            "custom_id": eid,
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "annotations": [],
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "refusal": None,
                            "role": "assistant"
                        }
                    }],
                    "created": int(datetime.datetime.now().timestamp()),
                    "id": f"chatcmpl-{eid}",
                    "model": model_name,
                    "object": "chat.completion",
                    "system_fingerprint": f"fp_{eid}"
                },
                "request_id": f"req-{eid}",
                "status_code": 200
            },
            "error": None
        }
    
    except Exception as e:
        print(f"Error during inference for event {eid}: {str(e)}")
        result = {
            "custom_id": eid,
            "response": {
                "body": None,
                "request_id": f"req-{eid}",
                "status_code": 500
            },
            "error": {
                "message": str(e),
                "type": "InternalError"
            }
        }

    return result


def main(path="/home/ubuntu/work/Temporal_relation/"):
    parser = argparse.ArgumentParser(description="LLM evaluation for temporal relations in clinical text")

    parser.add_argument("--data_dir", default=path + "data/qa_data/", 
                        help="Directory containing the data files")
    parser.add_argument("--output_dir", default=path + "llm_qa/qa_results/", 
                        help="Directory to save the results")
    parser.add_argument("--mode", default="individual", choices=["individual", "all"],
                        help="Processing mode: individual or all events")
    parser.add_argument("--api_base", default="http://localhost:8000/v1", 
                        help="Base URL for the API")
    parser.add_argument("--model", default="QwQ-32B-AWQ", 
                        help="Model name to use")
    args = parser.parse_args()

    # Validate directories
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure the API client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    # Load and process the data
    jsonl_files = [f for f in os.listdir(args.data_dir) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in {args.data_dir}")
        return
    
    for file_name in jsonl_files:
        data_file = os.path.join(args.data_dir, file_name)
        output_file = os.path.join(args.output_dir, f"{file_name}_{args.model}_results.jsonl")

        # Define the schema for the response based on mode
        if "individual" in file_name:
            class Timeline(BaseModel):
                datetime: str
                clues: str
            json_schema = Timeline.model_json_schema()
        else:
            # For "all" mode, define a schema for a list of objects
            class TimelineEvent(BaseModel):
                event_id: str
                datetime: str
                clues: str
            
            class TimelineEvents(BaseModel):
                events: List[TimelineEvent]
            
            json_schema = TimelineEvents.model_json_schema()


        print(f"Loading data from {data_file}...")
        data = load_data(data_file)
    
        if not data:
            print(f"No valid data found in {data_file}. Skipping.")
            continue

        # Initialize counters
        processed_count = 0
        total_items = len(data)
        
        print(f"Processing {file_name} with {total_items} items...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for eid, messages in data.items():
                print(f"Processing {eid} ({processed_count+1}/{total_items})...")
                
                result = inference(eid, messages, client, args.model, json_schema)
                processed_count += 1

                # Write to JSONL file (one JSON object per line)
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

        print(f"Completed! Processed {processed_count}/{total_items} items.")
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
    # Test the load_data function
    # data = load_data("/home/ubuntu/work/Temporal_relation/llm_qa/qa_data/timeline_azure_bulk_openai_notime_all_sections.jsonl")
    # print(data)
    # RUN: python llm_local_inference.py 
    