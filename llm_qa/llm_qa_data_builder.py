import os
import json
import re
from collections import OrderedDict

def data_load(src_folder: str, use_time_tags=False):
    """Load data from files in the source folder"""
    # Get all .xml.label.txt files with appropriate suffix based on using time tags or not
    suffix = '.xml.label.txt' if use_time_tags else '.xml.notime.label.txt'
    label_files = [f for f in os.listdir(src_folder) if f.endswith(suffix)]
    label_files = sorted(label_files)  # Sort files to maintain consistent ordering
    
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
                    {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events from clinical text."}, 
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

    results.append({
        "custom_id": "doc-" + str(doc_id), 
        "method": "POST", 
        "url": "/chat/completions", 
        "body": {
            "model": model, 
            "messages": [
                {"role": "system", "content": "You are a medical assistant with expertise in understanding clinical timelines and temporal relationships between medical events from clinical text."}, 
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

    # Generate base output filename based on parameters
    time_part = "time" if time_tags else "notime"
    mode_part = "individual" if mode == "individual" else "all"
    section_part = "_sections" if section_context else ""
    model_short = model.split("/")[-1]
    
    base_filename = f"timeline_azure_bulk_{model_short}_{time_part}_{mode_part}{section_part}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write all results to a single file first, then split if needed
    def save_and_split_if_needed(results, base_filename, output_dir, max_size_mb=200):
        """Save all results to one file, then split if file exceeds size limit"""
        if not results:
            print("No results to save.")
            return
        
        print(f"Writing {len(results):,} results to file...")
        
        # Write complete file first
        complete_filename = os.path.join(output_dir, f"{base_filename}.jsonl")
        
        with open(complete_filename, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Check file size
        file_size_mb = os.path.getsize(complete_filename) / (1024 * 1024)
        print(f"Complete file written: {os.path.basename(complete_filename)} ({file_size_mb:.2f} MB)")
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # If file is within size limit, we're done
        if file_size_mb <= max_size_mb:
            print(f"✅ File size is within {max_size_mb} MB limit. No splitting needed.")
            print(f"Configuration completed! Single file created with {len(results):,} requests")
            return
        
        # File is too large, need to split
        print(f"⚠️  File size ({file_size_mb:.2f} MB) exceeds {max_size_mb} MB limit. Splitting into smaller files...")
        
        # Read the complete file and split it
        current_chunk = []
        current_size = 0
        file_number = 1
        total_files_created = 0
        
        with open(complete_filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_size = len(line.encode('utf-8'))
                
                # If adding this line would exceed the limit and we have content, save current chunk
                if current_size + line_size > max_size_bytes and current_chunk:
                    # Save current chunk
                    chunk_filename = os.path.join(output_dir, f"{base_filename}_part{file_number:03d}.jsonl")
                    
                    with open(chunk_filename, 'w') as chunk_file:
                        chunk_file.writelines(current_chunk)
                    
                    chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
                    print(f"Saved chunk {file_number}: {os.path.basename(chunk_filename)} ({len(current_chunk)} requests, {chunk_size_mb:.2f} MB)")
                    
                    # Reset for next chunk
                    current_chunk = []
                    current_size = 0
                    file_number += 1
                    total_files_created += 1
                
                # Add line to current chunk
                current_chunk.append(line)
                current_size += line_size
                
                # Progress indicator for large files
                if line_num % 1000 == 0:
                    progress_mb = current_size / (1024 * 1024)
                    print(f"   Processing line {line_num:,}... Current chunk: {progress_mb:.2f} MB")
        
        # Save the final chunk if it has content
        if current_chunk:
            chunk_filename = os.path.join(output_dir, f"{base_filename}_part{file_number:03d}.jsonl")
            
            with open(chunk_filename, 'w') as chunk_file:
                chunk_file.writelines(current_chunk)
            
            chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
            print(f"Saved final chunk {file_number}: {os.path.basename(chunk_filename)} ({len(current_chunk)} requests, {chunk_size_mb:.2f} MB)")
            
            total_files_created += 1
        
        # Remove the original large file
        os.remove(complete_filename)
        print(f"🗑️  Removed original large file: {os.path.basename(complete_filename)}")
        
        print(f"✅ Configuration completed! Results split into {total_files_created} files with {len(results):,} total requests")

    # Save the final results with write-first-then-split logic
    max_file_size_mb = 200  # 200 MB limit per file
    save_and_split_if_needed(results, base_filename, output_dir, max_file_size_mb)

def split_existing_large_files():
    """Find and split any existing files that are over the size limit"""
    path = '/home/ubuntu/work/Temporal_relation/'
    
    # Define data directories to check
    data_directories = [
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training",
            "name": "timeline_training"
        },
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training_llm_update",
            "name": "timeline_training_llm_update"
        }
    ]
    
    max_size_mb = 200
    max_size_bytes = max_size_mb * 1024 * 1024
    
    print(f"{'='*100}")
    print(f"CHECKING FOR LARGE FILES TO SPLIT (>{max_size_mb} MB)")
    print(f"{'='*100}")
    
    total_split = 0
    
    for data_config in data_directories:
        output_dir = data_config["output_dir"]
        name = data_config["name"]
        
        print(f"\n{'='*80}")
        print(f"CHECKING DIRECTORY: {name}")
        print(f"{'='*80}")
        
        if not os.path.exists(output_dir):
            print(f"Directory does not exist: {output_dir}")
            continue
        
        # Find all JSONL files
        all_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
        
        for file in all_files:
            if '_part' in file or '_combined' in file:
                continue  # Skip already split or combined files
            
            file_path = os.path.join(output_dir, file)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                print(f"\n📂 Found large file: {file} ({file_size_mb:.2f} MB)")
                
                # Ask for confirmation
                split_file = input(f"Split this file? (y/n): ").lower().strip() == 'y'
                
                if not split_file:
                    print("Skipping...")
                    continue
                
                # Split the file
                base_name = file.replace('.jsonl', '')
                
                current_chunk = []
                current_size = 0
                file_number = 1
                
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line_size = len(line.encode('utf-8'))
                        
                        # If adding this line would exceed the limit and we have content, save current chunk
                        if current_size + line_size > max_size_bytes and current_chunk:
                            # Save current chunk
                            chunk_filename = os.path.join(output_dir, f"{base_name}_part{file_number:03d}.jsonl")
                            
                            with open(chunk_filename, 'w') as chunk_file:
                                chunk_file.writelines(current_chunk)
                            
                            chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
                            print(f"   Saved chunk {file_number}: {os.path.basename(chunk_filename)} ({len(current_chunk)} lines, {chunk_size_mb:.2f} MB)")
                            
                            # Reset for next chunk
                            current_chunk = []
                            current_size = 0
                            file_number += 1
                
                # Save the final chunk
                if current_chunk:
                    chunk_filename = os.path.join(output_dir, f"{base_name}_part{file_number:03d}.jsonl")
                    
                    with open(chunk_filename, 'w') as chunk_file:
                        chunk_file.writelines(current_chunk)
                    
                    chunk_size_mb = os.path.getsize(chunk_filename) / (1024 * 1024)
                    print(f"   Saved final chunk {file_number}: {os.path.basename(chunk_filename)} ({len(current_chunk)} lines, {chunk_size_mb:.2f} MB)")
                
                # Remove the original large file
                os.remove(file_path)
                print(f"   🗑️  Removed original file: {file}")
                print(f"   ✅ Split into {file_number} files")
                
                total_split += 1
            else:
                print(f"✅ {file}: {file_size_mb:.2f} MB (within limit)")
    
    print(f"\n{'='*100}")
    print(f"SPLITTING SUMMARY: {total_split} files were split")
    print(f"{'='*100}")

def count_items_in_file(file_path):
    """Count items and file size for a JSONL file"""
    count = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        # Get file size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        return count, size_mb
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0, 0

def check_and_combine_data():
    """Check split files, combine them, and count dictionary items"""
    path = '/home/ubuntu/work/Temporal_relation/'
    
    # Define data directories to check
    data_directories = [
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training",
            "name": "timeline_training"
        },
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training_llm_update",
            "name": "timeline_training_llm_update"
        }
    ]
    
    print(f"{'='*100}")
    print("DATA CHECKER: Analyzing split files and combining chunks")
    print(f"{'='*100}")
    
    for data_config in data_directories:
        output_dir = data_config["output_dir"]
        name = data_config["name"]
        
        print(f"\n{'='*80}")
        print(f"CHECKING DIRECTORY: {name}")
        print(f"Path: {output_dir}")
        print(f"{'='*80}")
        
        if not os.path.exists(output_dir):
            print(f"Directory does not exist: {output_dir}")
            continue
        
        # Find all JSONL files in the directory
        all_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
        
        if not all_files:
            print(f"No JSONL files found in {output_dir}")
            continue
        
        # Group files by base name (before _part)
        file_groups = {}
        single_files = []
        
        for file in all_files:
            if '_part' in file:
                # This is a split file
                base_name = file.split('_part')[0]
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file)
            else:
                # This is a single file
                single_files.append(file)
        
        # Process single files
        print(f"\n📁 SINGLE FILES ({len(single_files)}):")
        for file in sorted(single_files):
            file_path = os.path.join(output_dir, file)
            count, size_mb = count_items_in_file(file_path)
            print(f"   {file}: {count:,} items, {size_mb:.2f} MB")
        
        # Process grouped files (split chunks)
        print(f"\n📦 SPLIT FILE GROUPS ({len(file_groups)}):")
        for base_name, files in file_groups.items():
            files.sort()  # Sort to ensure correct order
            print(f"\n   {base_name}:")
            
            total_items = 0
            total_size_mb = 0
            combined_data = []
            
            for file in files:
                file_path = os.path.join(output_dir, file)
                count, size_mb = count_items_in_file(file_path)
                total_items += count
                total_size_mb += size_mb
                
                print(f"     {file}: {count:,} items, {size_mb:.2f} MB")
                
                # Read and combine data
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            combined_data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"       ⚠️ Error reading line in {file}: {e}")
            
            print(f"     {'='*50}")
            print(f"     TOTAL: {total_items:,} items, {total_size_mb:.2f} MB")
            
            # Optionally save combined file
            combined_filename = os.path.join(output_dir, f"{base_name}_combined.jsonl")
            save_combined = input(f"     Save combined file for {base_name}? (y/n): ").lower().strip() == 'y'
            
            if save_combined:
                with open(combined_filename, 'w') as f:
                    for item in combined_data:
                        f.write(json.dumps(item) + '\n')
                
                # Get combined file size
                combined_size_mb = os.path.getsize(combined_filename) / (1024 * 1024)
                print(f"     ✅ Saved combined file: {base_name}_combined.jsonl ({combined_size_mb:.2f} MB)")
            
            # Verify combined data integrity
            if len(combined_data) != total_items:
                print(f"     ⚠️ WARNING: Combined data has {len(combined_data)} items but expected {total_items}")

def analyze_all_data():
    """Comprehensive analysis of all data files"""
    path = '/home/ubuntu/work/Temporal_relation/'
    
    # Define data directories to analyze
    data_directories = [
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training",
            "name": "timeline_training"
        },
        {
            "output_dir": path + "llm_qa/qa_data/timeline_training_llm_update",
            "name": "timeline_training_llm_update"
        }
    ]
    
    print(f"{'='*100}")
    print("COMPREHENSIVE DATA ANALYSIS")
    print(f"{'='*100}")
    
    grand_total_items = 0
    grand_total_size = 0
    
    for data_config in data_directories:
        output_dir = data_config["output_dir"]
        name = data_config["name"]
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {name}")
        print(f"{'='*80}")
        
        if not os.path.exists(output_dir):
            print(f"Directory does not exist: {output_dir}")
            continue
        
        # Get all JSONL files
        all_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
        
        if not all_files:
            print(f"No JSONL files found")
            continue
        
        dir_total_items = 0
        dir_total_size = 0
        
        for file in sorted(all_files):
            if '_combined' in file:
                continue  # Skip combined files to avoid double counting
                
            file_path = os.path.join(output_dir, file)
            count, size_mb = count_items_in_file(file_path)
            
            dir_total_items += count
            dir_total_size += size_mb
            
            # Mark files that are close to the 200MB limit
            size_warning = " ⚠️ CLOSE TO 200MB LIMIT" if size_mb > 180 else ""
            print(f"   {file}: {count:,} items, {size_mb:.2f} MB{size_warning}")
        
        print(f"   {'='*60}")
        print(f"   DIRECTORY TOTAL: {dir_total_items:,} items, {dir_total_size:.2f} MB")
        
        grand_total_items += dir_total_items
        grand_total_size += dir_total_size
    
    print(f"\n{'='*100}")
    print(f"GRAND TOTAL ACROSS ALL DIRECTORIES:")
    print(f"Total items: {grand_total_items:,}")
    print(f"Total size: {grand_total_size:.2f} MB")
    print(f"{'='*100}")

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
    
    # Define both data directories to process
    data_directories = [
        {
            "data_dir": path + "data/timeline_training/",
            "output_dir": path + "llm_qa/qa_data/timeline_training",
            "name": "timeline_training"
        },
        # {
        #     "data_dir": path + "data/timeline_training_llm_update/",
        #     "output_dir": path + "llm_qa/qa_data/timeline_training_llm_update",
        #     "name": "timeline_training_llm_update"
        # }
    ]
    
    # Define common arguments
    common_args = {
        "limit": 50000000,
        # "model": "gpt-4o-mini"
        "model": "o3-mini"
        # "model": "o4-mini"
    }
    
    total_combinations = len(configurations) * len(data_directories)
    print(f"Starting processing of {total_combinations} different combinations ({len(configurations)} configurations × {len(data_directories)} data directories)...")
    
    combination_count = 0
    
    # Process each data directory
    for data_config in data_directories:
        print(f"\n{'='*100}")
        print(f"PROCESSING DATA DIRECTORY: {data_config['name']}")
        print(f"{'='*100}")
        
        # Process each configuration for this data directory
        for i, config in enumerate(configurations, 1):
            combination_count += 1
            print(f"\n--- Combination {combination_count}/{total_combinations}: {data_config['name']} - Configuration {i}/{len(configurations)} ---")
            try:
                process_configuration(
                    **config, 
                    data_dir=data_config["data_dir"],
                    output_dir=data_config["output_dir"],
                    **common_args
                )
            except Exception as e:
                print(f"Error processing configuration {config} for {data_config['name']}: {str(e)}")
                continue
    
    print(f"\n{'='*100}")
    print("All combinations completed!")
    print(f"Processed {len(data_directories)} data directories with {len(configurations)} configurations each")
    print(f"{'='*100}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "main":
            main()
        elif command == "check":
            check_and_combine_data()
        elif command == "analyze":
            analyze_all_data()
        elif command == "split":
            split_existing_large_files()
        else:
            print("Usage:")
            print("  python llm_qa_data_builder.py main      # Generate data files")
            print("  python llm_qa_data_builder.py check     # Check and combine split files")
            print("  python llm_qa_data_builder.py analyze   # Analyze all data comprehensively")
            print("  python llm_qa_data_builder.py split     # Split existing large files")
    else:
        print("Usage:")
        print("  python llm_qa_data_builder.py main      # Generate data files")
        print("  python llm_qa_data_builder.py check     # Check and combine split files")
        print("  python llm_qa_data_builder.py analyze   # Analyze all data comprehensively")
        print("  python llm_qa_data_builder.py split     # Split existing large files")