# A consolidated script for evaluating the performance of LLMs in inferring temporal relations from clinical text.
# Supports different modes: individual/all events, with/without time tags, with/without section context
from openai import OpenAI
from pydantic import BaseModel
import json, os, datetime, re, argparse
from collections import OrderedDict
from typing import List


def data_load(src_folder: str, use_time_tags=False):
    """Load data from files in the source folder"""
    # Get all .xml.label.txt files with appropriate suffix based on using time tags or not
    suffix = '.xml.label.txt' if use_time_tags else '.xml.notime.label.txt'
    label_files = [f for f in os.listdir(src_folder) if f.endswith(suffix)]
    label_files = sorted(label_files)  # Sort files to maintain consistent ordering
    print(label_files)
    
    # Create ordered dictionary to maintain consistent ordering
    organized_data = OrderedDict()

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
    # print(list(organized_data.keys())[:10])
    return organized_data


def save_intermediate_results(results, output_dir):
    """Save intermediate results to avoid data loss if the process stops"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'intermediate_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved intermediate results to {output_file}")


def preprocess_text(text, event_id):
    """
    Removes all event tags <E#> and </E#> from the text, except for the specified event tag.
    
    Args:
        text (str): The input text containing event tags
        event_id (str): The event ID to keep (e.g., "E2", "E5", etc.)
        
    Returns:
        str: The text with all event tags removed except for the specified event
    """
    # Strip the 'E' prefix if it exists to get just the number
    if event_id.startswith('E'):
        event_num = event_id[1:]
    else:
        event_num = event_id
        event_id = 'E' + event_id
    
    # Create the pattern to match all <E#> and </E#> tags except the specified one
    pattern = r'<E(?!{0}>)(\d+)>|</E(?!{0}>)(\d+)>'.format(event_num)
    
    # Remove all matches of the pattern
    processed_text = re.sub(pattern, '', text)
    
    return processed_text


def load_prompts():
    """Load prompts from the prompts file"""
    prompt_file = os.path.join(os.path.dirname(__file__), 'prompts.json')
    try:
        with open(prompt_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file not found at {prompt_file}. Please create this file first.")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing JSON in prompts file at {prompt_file}")


def temporal_inference_individual(text, events, client, model_name, json_schema, prompt_key="individual_notime"):
    """Evaluate the model's ability to infer time for specific events individually"""
    results = []
    prompts = load_prompts()
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts file")
    
    for event in events:
        eid = event['node_id']
        ground_truth_range = event['formatted_time_range']
        
        # Process text to keep only the current event's tags
        processed_text = preprocess_text(text, eid)

        # Get the appropriate prompt from the prompts file
        prompt_template = prompts[prompt_key]
        event_prompt = prompt_template.replace("{eid}", eid)

        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."},
                    {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + processed_text},
                ],
                extra_body={"guided_json": json_schema},
            )
            
            reasoning_content = chat_response.choices[0].message.reasoning_content
            content = chat_response.choices[0].message.content

            print(f"Event ID: {eid}")
            print(event_prompt + "\n\nCLINICAL TEXT: " + processed_text)
            print("reasoning_content:", reasoning_content)
            print("content:", content)

            result = {
                "event_id": eid,
                "ground_truth": ground_truth_range,
                "llm_reasoning": reasoning_content,
                "llm_inference": content
            }
            results.append(result)
            print("-" * 80)
        
        except Exception as e:
            print(f"Error during inference for event {eid}: {str(e)}")
            result = {
                "event_id": eid,
                "ground_truth": ground_truth_range,
                "llm_reasoning": None,
                "llm_inference": None,
                "error": str(e)
            }
            results.append(result)
    
    return results


def temporal_inference_all(text, events, client, model_name, json_schema, prompt_key="all_notime"):
    """Evaluate the model's ability to infer time for all events at once"""
    prompts = load_prompts()
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts file")
    
    event_prompt = prompts[prompt_key]

    print(event_prompt + "\n\nCLINICAL TEXT: " + text)
    
    # No preprocessing needed for "all" mode - use text as is
    try:
        # For "all" mode, use guided_json since we expect a list of objects
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."},
                {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + text},
            ],
            extra_body={"guided_json": json_schema},
        )

        ground_truth = []
        for event in events:
            ground_truth.append({
                "node_id": event['node_id'],
                "formatted_time_range": event['formatted_time_range'],
            })
        

        # Extract reasoning and inference content
        reasoning_content = chat_response.choices[0].message.reasoning_content
        content = chat_response.choices[0].message.content

        print(event_prompt + "\n\nCLINICAL TEXT: " + text)
        print("reasoning_content:", reasoning_content)
        print("content:", content)

        results = {
            "ground_truth": ground_truth,
            "llm_reasoning": reasoning_content,
            "llm_inference": content
        }
    except Exception as e:
        print(f"Error during inference for all events: {str(e)}")
        results = {
            "ground_truth": [{"node_id": event['node_id'], "formatted_time_range": event['formatted_time_range']} for event in events],
            "llm_reasoning": None,
            "llm_inference": None,
            "error": str(e)
        }
    
    return results


def main():
    path = '/home/ubuntu/work/Temporal_relation/'

    parser = argparse.ArgumentParser(description="LLM evaluation for temporal relations in clinical text")

    parser.add_argument("--mode", choices=["individual", "all"], default="individual", 
                        help="Process events individually or all at once")
    parser.add_argument("--time_tags", action="store_true", 
                        help="Whether to include time tags in the text")
    parser.add_argument("--section_context", action="store_true", 
                        help="Whether to include section context information in the prompt")
    parser.add_argument("--data_dir", default=path + "data/timeline_training/", 
                        help="Directory containing the data files")
    parser.add_argument("--output_dir", default=path + "llm_qa/qa_results", 
                        help="Directory to save the results")
    parser.add_argument("--intermediate_dir", default=path + "llm_qa/intermediate_results/", 
                        help="Directory to save intermediate results")
    parser.add_argument("--limit", type=int, default=500000, 
                        help="Limit the number of files to process")
    parser.add_argument("--api_base", default="http://localhost:8000/v1", 
                        help="Base URL for the API")
    parser.add_argument("--model", default="Qwen/QwQ-32B-AWQ", 
                        help="Model name to use")
    args = parser.parse_args()

    # Determine which prompt to use based on the arguments
    prompt_key = f"{'individual' if args.mode == 'individual' else 'all'}_{'time' if args.time_tags else 'notime'}"
    if args.section_context:
        prompt_key += "_section"

    # Configure the API client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    # Define the schema for the response based on mode
    if args.mode == "individual":
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

    # Load the data
    data = data_load(args.data_dir, use_time_tags=args.time_tags)
    
    if not data:
        print("No data found in the specified directory. Exiting.")
        return

    results = {}
    save_interval = 5
    processed_count = 0

    # Get the total number of files for progress tracking and limit if specified
    keys_to_process = list(data.keys())[:args.limit] if args.limit > 0 else list(data.keys())
    total_files = len(keys_to_process)
    print(f"Starting evaluation of {total_files} files...")

    for key in keys_to_process:
        try:
            print(f"Processing {key} ({processed_count+1}/{total_files})...")
            text = data[key]['label']
            start_time = data[key]['starttime']
            
            if start_time is None or not start_time:
                print(f"Warning: No starttime data for {key}, skipping...")
                processed_count += 1
                continue
            
            # Call the appropriate inference function based on mode
            if args.mode == "individual":
                inference = temporal_inference_individual(text, start_time, client, args.model, json_schema, prompt_key)
            else:
                inference = temporal_inference_all(text, start_time, client, args.model, json_schema, prompt_key)
            
            results[key] = inference
            processed_count += 1
            
            # Save intermediate results at regular intervals
            if processed_count % save_interval == 0 or processed_count == total_files:
                save_intermediate_results(results, args.intermediate_dir)
                print(f"Progress: {processed_count}/{total_files} files processed ({processed_count/total_files*100:.1f}%)")
        
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            # Save on error to prevent data loss
            save_intermediate_results(results, args.intermediate_dir)

    # Generate output filename based on parameters
    time_part = "time" if args.time_tags else "notime"
    mode_part = "individual" if args.mode == "individual" else "all"
    section_part = "_sections" if args.section_context else ""
    model_short = args.model.split("/")[-1]
    
    output_file = os.path.join(
        args.output_dir, 
        f"timeline_training_{model_short}_results_{time_part}_{mode_part}{section_part}.json"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Completed! Final results saved to {output_file}")


if __name__ == "__main__":
    main()
