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
    # print(label_files)
    
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

def remove_time_tags(text):
    """
    Removes all time tags <T#> and </T#> from the text.
    
    Args:
        text (str): The input text containing time tags
        
    Returns:
        str: The text with all time tags removed
    """
    # Pattern to match all <T#> and </T#> tags
    pattern = r'<T\d+>|</T\d+>'
    
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


def temporal_data_individual(text, doc_id, events, model="gpt-4o", prompt_key="individual_notime"):
    """Evaluate the model's ability to infer time for specific events individually"""
    results = []
    prompts = load_prompts()
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts file")
    
    for event in events:
        eid = event['node_id']
        
        # Process text to keep only the current event's tags
        processed_text = preprocess_text(text, eid)
        if 'notime' in prompt_key:
            # If the prompt is for "notime", remove all time tags from the text
            processed_text = remove_time_tags(processed_text)

        # Get the appropriate prompt from the prompts file
        prompt_template = prompts[prompt_key]
        event_prompt = prompt_template.replace("{eid}", eid)

        results.append({
            "custom_id": "doc-" + str(doc_id) + "-task-" + eid, 
            "method": "POST", 
            "url": "/chat/completions", 
            "body": {
                "model": model, 
                "messages": [
                    {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."}, 
                    {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + processed_text}
                ]
            }
        })
    
    return results


def temporal_data_all(text, doc_id, model="gpt-4o", prompt_key="all_notime"):
    """Evaluate the model's ability to infer time for all events at once"""
    prompts = load_prompts()
    results = []
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in prompts file")
    
    event_prompt = prompts[prompt_key]
    
    if 'notime' in prompt_key:
        # If the prompt is for "notime", remove all time tags from the text
        text = remove_time_tags(text)
    
    # print(event_prompt + "\n\nCLINICAL TEXT: " + text)

    results.append({
        "custom_id": "doc-" + str(doc_id), 
        "method": "POST", 
        "url": "/chat/completions", 
        "body": {
            "model": model, 
            "messages": [
                {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events."}, 
                {"role": "user", "content": event_prompt + "\n\nCLINICAL TEXT: " + text}
            ]
        }
    })
    
    return results


def process_configuration(mode, time_tags, section_context, data_dir, output_dir, limit, model):
    """Process a single configuration combination"""
    print(f"\n{'='*80}")
    print(f"Processing configuration: mode={mode}, time_tags={time_tags}, section_context={section_context}")
    print(f"{'='*80}")
    
    # Determine which prompt to use based on the arguments
    prompt_key = f"{'individual' if mode == 'individual' else 'all'}_{'time' if time_tags else 'notime'}"
    if section_context:
        prompt_key += "_section"

    # Load the data
    data = data_load(data_dir, use_time_tags=time_tags)
    
    if not data:
        print("No data found in the specified directory. Skipping this configuration.")
        return

    results = []
    processed_count = 0

    # Get the total number of files for progress tracking and limit if specified
    keys_to_process = list(data.keys())[:limit] if limit > 0 else list(data.keys())
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
            if mode == "individual":
                inference = temporal_data_individual(text, key, start_time, model, prompt_key)
            else:
                inference = temporal_data_all(text, key, model, prompt_key)
            
            results.extend(inference)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            processed_count += 1

    # Generate output filename based on parameters
    time_part = "time" if time_tags else "notime"
    mode_part = "individual" if mode == "individual" else "all"
    section_part = "_sections" if section_context else ""
    model_short = model.split("/")[-1]
    
    output_file = os.path.join(
        output_dir, 
        f"timeline_azure_bulk_{model_short}_{time_part}_{mode_part}{section_part}.jsonl"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the final results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"Configuration completed! Results saved to {output_file}")


def main():
    # Define base path and default arguments
    path = '/home/ubuntu/work/Temporal_relation/'
    
    # Define all argument combinations to process
    configurations = [
        # Individual mode combinations
        {"mode": "individual", "time_tags": False, "section_context": False},
        {"mode": "individual", "time_tags": True, "section_context": False},
        {"mode": "individual", "time_tags": False, "section_context": True},
        {"mode": "individual", "time_tags": True, "section_context": True},
        # All mode combinations
        {"mode": "all", "time_tags": False, "section_context": False},
        {"mode": "all", "time_tags": True, "section_context": False},
        {"mode": "all", "time_tags": False, "section_context": True},
        {"mode": "all", "time_tags": True, "section_context": True},
    ]
    
    # Define common arguments
    common_args = {
        "data_dir": path + "data/timeline_training/",
        "output_dir": path + "llm_qa/qa_data",
        "limit": 50000000,
        "model": "gpt-4o-mini"  # Default model, can be overridden in configurations
    }
    
    print(f"Starting processing of {len(configurations)} different configurations...")
    
    # Process each configuration
    for i, config in enumerate(configurations, 1):
        print(f"\n--- Configuration {i}/{len(configurations)} ---")
        try:
            process_configuration(**config, **common_args)
        except Exception as e:
            print(f"Error processing configuration {config}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print("All configurations completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
    # Run: python llm_qa_data_builder.py
