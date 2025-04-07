from openai import OpenAI
from pydantic import BaseModel
import json, os, datetime


def data_load(src_folder: str):
    # Get all .xml.label.txt and other files in the source folder
    label_files = [f for f in os.listdir(src_folder) if f.endswith('.xml.notime.label.txt')]
    
    # Sort the files to ensure consistent ordering across runs
    label_files.sort()

    # Dictionary to store the organized data by file ID
    organized_data = {}

    # Process each label file
    for filename in label_files:
        # Extract the file ID (everything before .xml)
        file_id = filename.split('.xml')[0]
        
        # Initialize the dictionary for this ID
        organized_data[file_id] = {}

        # Load label text file
        label_path = os.path.join(src_folder, filename)
        with open(label_path, "r") as f:
            organized_data[file_id]['label'] = f.read()

        # Load corresponding starttime JSON file
        starttime_filename = f"{file_id}.xml.starttime.json"
        starttime_path = os.path.join(src_folder, starttime_filename)
        if os.path.exists(starttime_path):
            with open(starttime_path, "r") as f:
                organized_data[file_id]['starttime'] = json.load(f)
        else:
            organized_data[file_id]['starttime'] = None

        # Load corresponding interval_paths JSON file
        interval_paths_filename = f"{file_id}.xml.interval_paths.json"
        interval_paths_path = os.path.join(src_folder, interval_paths_filename)
        if os.path.exists(interval_paths_path):
            with open(interval_paths_path, "r") as f:
                organized_data[file_id]['interval_paths'] = json.load(f)
        else:
            organized_data[file_id]['interval_paths'] = None

    print(f"Loaded data for {len(organized_data)} IDs.")
    return organized_data


def save_intermediate_results(results, output_dir='/home/jovyan/work/Temporal_relation/data/intermediate_results'):
    """Save intermediate results to avoid data loss if the process stops"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'intermediate_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved intermediate results to {output_file}")


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://host.docker.internal:8000/v1"
model_name = 'Qwen/QwQ-32B-AWQ'

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# llm = VLLMOpenAI(
#     openai_api_key=openai_api_key,
#     openai_api_base=openai_api_base,
#     model_name=model_name,
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


class timeline(BaseModel):
    """
    Model for timeline information with specific datetime format requirements:
    - "ON YYYY-MM-DD" 
    - "YYYY-MM-DD TO YYYY-MM-DD"
    """
    datetime: str  # Format: "BEFORE/AFTER/AT/ON YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS TO YYYY-MM-DD HH:MM:SS"
    clues: str

json_schema = timeline.model_json_schema()


def temporal_inference(text, events):
    """Evaluate the model's ability to infer time for specific events"""
    results = []
    
    # Use the prompt_type parameter to select the appropriate prompt
    
    for event in events:
        eid = event['node_id']
        ground_truth_range = event['formatted_time_range']
        
        # Create a targeted prompt for this specific event, incorporating the selected prompt type
        event_prompt = f"""\n\n
        Given the clinical text, determine the exact date/time when the event marked with <{eid}> and </{eid}> occurred.
        Provide the most precise date information and generate a JSON output containing the datetime and the temporal clues used for your reasoning.
        
        For the datetime field, use one of these exact formats:
        - For point-in-time events: "BEFORE|AFTER|AT|ON YYYY-MM-DD" (e.g., "ON 2023-05-15")
        - For range events: "YYYY-MM-DD TO YYYY-MM-DD" (e.g., "2023-05-15 TO 2023-05-16")
        - If the information is unclear, use "None"
        
        # # Example output:
        {{
          "datetime": "ON 2023-05-15",
          "clues": "The procedure was documented as occurring on May 15"
        }}
        
        Or:
        
        {{
          "datetime": "2023-05-15 TO 2023-05-16",
          "clues": "The medication was started on May 15 and discontinued the next day"
        }}
        """

        # chain = (llm | StrOutputParser())

        # response = chain.invoke(event_prompt + "\n\nCLINICAL TEXT: " + text)
        # print(f"LLM inference: {response}\n")
        # print("*" * 80)

        chat_response = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."},
            {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + text},
                ],
            extra_body={"guided_json": json_schema},
            )
        
        reasoning_content = chat_response.choices[0].message.reasoning_content
        content = chat_response.choices[0].message.content

        print("reasoning_content:", reasoning_content)
        print("content:", content)

        result = {
            "event_id": eid,
            "ground_truth": ground_truth_range,
            "llm_reasoning": reasoning_content,
            "llm_inference": content
        }
        results.append(result)
        
        # print(f"\nEvent ID: {eid}")
        # print(f"Ground truth time range: {ground_truth_range}")
        # print(f"LLM inference: {response}\n")
        print("-" * 80)
    
    return results


if __name__ == "__main__":
    train_data = data_load('/home/jovyan/work/Temporal_relation/data/timeline_training/')
    # test_data = data_load('/home/jovyan/work/Temporal_relation/data/timeline_test/')

    results = {}
    save_interval = 5  # Save after every 5 processed files
    processed_count = 0

    # Get the total number of files for progress tracking and limit to 50
    keys_to_process = list(train_data.keys())[:50]  # Only process first 50 keys
    total_files = len(keys_to_process)
    print(f"Starting evaluation of {total_files} files (limited to first 50)...")

    for key in keys_to_process:
        try:
            print(f"Processing {key} ({processed_count+1}/{total_files})...")
            text = train_data[key]['label']
            start_time = train_data[key]['starttime']
            inference = temporal_inference(text, start_time)
            results[key] = inference
            processed_count += 1
            
            # Save intermediate results at regular intervals
            if processed_count % save_interval == 0 or processed_count == total_files:
                save_intermediate_results(results)
                print(f"Progress: {processed_count}/{total_files} files processed ({processed_count/total_files*100:.1f}%)")
        
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            # Save on error to prevent data loss
            save_intermediate_results(results)

    # Save the final results to a JSON file
    output_file = '/home/jovyan/work/Temporal_relation/data/timeline_training_QwQ-32B-AWQ_results_nontime.json'
    # output_file = '/home/jovyan/work/Temporal_relation/data/timeline_training_QwQ-32B-AWQ_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Completed! Final results saved to {output_file}")
