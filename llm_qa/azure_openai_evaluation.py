import llm_qa_data_builder
import json
from datetime import datetime
# import pandas as pd
import os

def get_gold_standard():
    path = '/home/ubuntu/work/Temporal_relation/'
    data_dir = path + "data/timeline_training/"
    results = {}

    data = llm_qa_data_builder.data_load(data_dir, True)

    for key in data:
        ground_truth = {}
        text = data[key]['label']
        event_start_time = data[key]['starttime']
        for event in event_start_time:
            ground_truth[event['node_id']] = {
                "event_id": event['node_id'],
                "formatted_time_range": event['formatted_time_range'],
            }
        # print(ground_truth)
        results[key] = ground_truth
    return results


def parse_jsonl_extract_content(jsonl_file_path):
    """
    Parse a JSONL file and extract message.content from Azure OpenAI responses
    Handle partially valid JSON by extracting valid objects and skipping invalid parts
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        
    Returns:
        dict: Dictionary with custom_id as key and message content as value
    """
    results = {}
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Extract custom_id
                custom_id = data.get('custom_id')
                
                
                # Navigate to message.content
                content = (data.get('response', {})
                          .get('body', {})
                          .get('choices', [{}])[0]
                          .get('message', {})
                          .get('content', None))
                
                if content:
                    if '```json' in content:
                        # Extract JSON from markdown code block
                        json_start = content.find('```json') + 7
                        json_end = content.find('```', json_start)
                        if json_end != -1:
                            json_content = content[json_start:json_end].strip()
                        else:
                            # No closing ```, try to parse from json_start to end
                            json_content = content[json_start:].strip()
                    else:
                        json_content = content.strip()

                    # Parse partially valid JSON
                    parsed_content = parse_partial_json(json_content)
                    
                    if custom_id and parsed_content:
                        results[custom_id] = parsed_content
                    # if custom_id == 'doc-491-task-E9':
                    #     print(f"Custom ID: {custom_id}")
                    #     print(f"Parsed Content: {json_content} \n {parsed_content}")    
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"Error parsing line: {e}")
                print(f"Line content: {line}")
                continue
    
    return results



def parse_partial_json(json_string):
    """
    Parse partially valid JSON by extracting complete JSON objects
    
    Args:
        json_string (str): JSON string that may be partially valid
        
    Returns:
        dict or list: Single dict if one object found, list if multiple objects found, None if no valid objects
    """
    valid_objects = []
    
    # First try to parse as complete JSON
    try:
        result = json.loads(json_string)
        # If it's a list, filter out empty dictionaries
        if isinstance(result, list):
            non_empty_objects = [obj for obj in result if obj and obj != {}]
            if len(non_empty_objects) == 1:
                return non_empty_objects[0]
            elif len(non_empty_objects) > 1:
                return non_empty_objects
            else:
                return None
        return result
    except json.JSONDecodeError:
        pass
    
    # If that fails, try to extract individual objects
    # Look for patterns like { ... } that might be valid JSON objects
    brace_count = 0
    start_pos = -1
    i = 0
    
    while i < len(json_string):
        char = json_string[i]
        
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                # Found a complete object
                potential_object = json_string[start_pos:i+1]
                try:
                    parsed_obj = json.loads(potential_object)
                    # Only add non-empty objects
                    if parsed_obj and parsed_obj != {}:
                        valid_objects.append(parsed_obj)
                except json.JSONDecodeError:
                    # Skip invalid object
                    pass
                start_pos = -1
        
        i += 1
    
    # Return single object if only one found, list if multiple, None if none
    if len(valid_objects) == 1:
        return valid_objects[0]
    elif len(valid_objects) > 1:
        return valid_objects
    else:
        return None

def get_azure_annotation_all(jsonl_file_path):
    """
    Load Azure OpenAI annotations from JSONL file
    """
    
    # Parse the JSONL and extract content
    annotations = parse_jsonl_extract_content(jsonl_file_path)
    
    formatted_annotations = {}
    for cid in annotations.keys():
        tokens = cid.split('-')
        doc_id = tokens[1]
        if doc_id not in formatted_annotations:
            formatted_annotations[doc_id] = {}
            for event in annotations[cid]:
                if isinstance(event, dict) and 'event_id' in event:
                    formatted_annotations[doc_id][event['event_id']] = event
                else:
                    print(f"Missing event_id in {cid}: {event}")

    return formatted_annotations

def get_azure_annotation_individual(jsonl_file_path):
    """
    Load Azure OpenAI annotations from JSONL file for individual task format
    """
    # Update this path to your actual JSONL file location
    jsonl_file_path = '/home/ubuntu/work/Temporal_relation/llm_qa/qa_azure_results/timeline_azure_bulk_openai_notime_individual_gpt4.jsonl'
    # Parse the JSONL and extract content
    annotations = parse_jsonl_extract_content(jsonl_file_path)
    
    formatted_annotations = {}
    for key in annotations.keys():
        tokens = key.split('-')
        doc_id, event_id = tokens[1], tokens[-1]
        if doc_id not in formatted_annotations:
            formatted_annotations[doc_id] = {}
        formatted_annotations[doc_id][event_id] = annotations[key]

    return formatted_annotations

def parse_datetime(date_str):
    # print(date_str)
    """Parse a date string into a datetime object, handling various formats."""
    import re
    
    # Handle various date formats and normalize them
    for old, new in [('DURING ', ''), ('BETWEEN ', ''), ('FROM ', ''), ('MIDWAY THROUGH ', ''), ('-??', '-01'), ('-XX', '-01'), ('-xx', '-01'), ('-00', '-01')]:
        date_str = date_str.replace(old, new)
        
    # Remove time component if present
    if "T" in date_str:
        date_str = date_str.split('T')[0]
    
    # Remove time component with space separator - updated pattern to handle a.m./p.m.
    time_patterns = [
        r'\s+\d{1,2}:\d{2}(:\d{2})?\s*[ap]\.?m\.?',  # Handles "2:30 p.m.", "2:30 pm", "2:30p.m."
        r'\s+\d{1,2}:\d{2}(:\d{2})?'                 # Handles regular "14:30", "2:30"
    ]
    
    for pattern in time_patterns:
        date_str = re.sub(pattern, '', date_str, flags=re.IGNORECASE)

    # Handle "Month DD, YYYY" format first (like "February 18, 1994")
    month_day_year_pattern = r'(\w+)\s+(\d+),?\s+(\d{4})'
    match = re.match(month_day_year_pattern, date_str)
    if match:
        month_name, day, year = match.groups()
        try:
            # Try full month name first
            date = datetime.strptime(f"{month_name} {day} {year}", '%B %d %Y')
            return date
        except ValueError:
            try:
                # Try abbreviated month name
                date = datetime.strptime(f"{month_name} {day} {year}", '%b %d %Y')
                return date
            except ValueError:
                pass
    
    # Handle "Month DD TO Month DD, YYYY" format (like "MARCH 14 TO MARCH 21, 1994")
    month_range_pattern = r'(\w+)\s+(\d+)\s+TO\s+(\w+)\s+(\d+),?\s+(\d{4})'
    match = re.match(month_range_pattern, date_str)
    if match:
        month1, day1, month2, day2, year = match.groups()
        try:
            # Return the start date for range formats
            date = datetime.strptime(f"{month1} {day1} {year}", '%B %d %Y')
            return date
        except ValueError:
            try:
                date = datetime.strptime(f"{month1.title()} {day1} {year}", '%B %d %Y')
                return date
            except ValueError:
                pass
    
    # Handle month name + year format
    if any(month in date_str.upper() for month in ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                                          'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']):
        try:
            # Handle "MONTH YYYY" format
            if any(month in date_str for month in ['January', 'February', 'March', 'April', 'May', 'June',
                                                  'July', 'August', 'September', 'October', 'November', 'December']):
                date = datetime.strptime(date_str, '%B %Y')
                return date
            # Handle "MONTH YYYY" format (uppercase)
            elif any(month in date_str for month in ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                                                    'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']):
                date = datetime.strptime(date_str.title(), '%B %Y')
                return date
        except ValueError:
            # If that fails, try abbreviated month names
            try:
                date = datetime.strptime(date_str, '%b %Y')
                return date
            except ValueError:
                pass
    
    # Handle YYYY-MM-DD TO YYYY-MM-DD format
    if 'TO' in date_str and re.search(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
        date_parts = date_str.split('TO')
        if len(date_parts) == 2:
            start_date_str = date_parts[0].strip()
            # Use the start date
            date_str = start_date_str
    
    # Handle prefixes like "NIGHT BEFORE", "ON", "BEFORE", etc.
    if ' ' in date_str:
        # Look for date patterns in the string
        date_pattern = r'\b\d{4}-\d{1,2}-\d{1,2}\b'
        date_match = re.search(date_pattern, date_str)
        if date_match:
            date_str = date_match.group()
        else:
            # Look for year patterns
            year_pattern = r'\b(19|20)\d{2}\b'
            year_match = re.search(year_pattern, date_str)
            if year_match:
                date_str = year_match.group()
            else:
                # Fall back to last word if no clear date pattern
                words = date_str.split(' ')
                date_str = words[-1]
    
    if ',' in date_str:
        date_str = date_str.split(',')[0]

    # Handle MM-YYYY format (like "12-1992")
    if re.match(r'^\d{1,2}-\d{4}$', date_str):
        month, year = date_str.split('-')
        date_str = f"{year}-{month.zfill(2)}-01"

    # Handle MM/DD/YYYY and M/D/YY formats (like 08/15/1998 or 9/7/93)
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            month, day, year = parts
            # Handle 2-digit years
            if len(year) == 2:
                year_int = int(year)
                # Assume years 00-30 are 2000s, 31-99 are 1900s
                year = f"20{year}" if year_int <= 30 else f"19{year}"
            # Convert to YYYY-MM-DD format with zero padding
            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        elif len(parts) == 2:
            # Handle MM/YY format (like "10/92")
            month, year = parts
            if len(year) == 2:
                year_int = int(year)
                year = f"20{year}" if year_int <= 30 else f"19{year}"
            date_str = f"{year}-{month.zfill(2)}-01"
    
    # Handle dash-separated dates (MM-DD-YYYY, MM-DD-YY, YYYY-MM-DD)
    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts) == 3:
            # Check if it's MM-DD-YYYY format (last part is 4 digits)
            if len(parts[2]) == 4:  # MM-DD-YYYY or M-DD-YYYY or MM-D-YYYY
                month, day, year = parts
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            # Check if it's MM-DD-YY format (last part is 2 digits and first two parts are reasonable dates)
            elif (len(parts[2]) == 2 and 
                  int(parts[0]) <= 12 and int(parts[1]) <= 31):  # M-DD-YY or MM-D-YY or MM-DD-YY
                month, day, year = parts
                year_int = int(year)
                year_prefix = '19' if year_int >= 50 else '20'
                date_str = f"{year_prefix}{year}-{month.zfill(2)}-{day.zfill(2)}"
            # Check if it's already YYYY-MM-DD format (first part is 4 digits)
            elif len(parts[0]) == 4:
                # Already in YYYY-MM-DD format, just ensure zero padding
                year, month, day = parts
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        elif len(parts) == 2:
            # Check if it's MM-DD format (both parts are reasonable dates)
            if (int(parts[0]) <= 12 and int(parts[1]) <= 31):
                # MM-DD format - assume current year
                current_year = datetime.now().year
                date_str = f"{current_year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            else:
                # Format is YYYY-MM (already handled above for MM-YYYY)
                date_str = f"{date_str}-01"
    
    # Handle year-only format
    if '-' not in date_str and re.match(r'^\d{4}$', date_str):
        # Year only
        date_str = f"{date_str}-01-01"
    
    # Final parsing
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        return date
    except ValueError as e:
        # If still can't parse, try to extract just the year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = year_match.group()
            return datetime.strptime(f"{year}-01-01", '%Y-%m-%d')
        raise e


def parse_label(label):
    # print(f"Parsing label: {label}")
    if (label is None) or (label == 'None') or ('None' in label):
        return None
    
    # Handle "UNKNOWN DATE" case
    if 'UNKNOWN DATE' in label.upper():
        return None
    
    # Check if label contains any date-like information
    # Look for years (4 digits), months (1-12), or common date patterns
    import re
    
    # Pattern to match years (1800-2099), months (01-12 or 1-12), days (01-31 or 1-31)
    # Also check for month names and common date separators
    date_patterns = [
        r'\b(18|19|20)\d{2}\b',  # Years 1800-2099
        r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{2}|\d{4})\b',  # MM/DD/YY or MM/DD/YYYY
        r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](\d{2}|\d{4})\b',  # DD/MM/YY or DD/MM/YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Full month names
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',  # Abbreviated month names
        r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',  # MM/DD or MM-DD
    ]
    
    has_date_info = any(re.search(pattern, label, re.IGNORECASE) for pattern in date_patterns)
    
    if not has_date_info:
        # print(f"No date information found in label: {label}")
        return None
    
    if label.startswith('AFTER ON'):
        label = label.replace('AFTER ON', 'AFTER')
    elif label.startswith('AFTER OR ON'):
        label = label.replace('AFTER OR ON', 'AFTER')
    elif label.startswith('BEFORE ON'):
        label = label.replace('BEFORE ON', 'BEFORE')
    elif label.startswith('BEFORE OR ON'):
        label = label.replace('BEFORE OR ON', 'BEFORE')
    elif label.startswith('AT'):
        label = label.replace('AT', 'ON')

    try:
        if label.startswith('ON'):
            # Handle cases like "ON NIGHT BEFORE 2019-06-14" or "ON February 18, 1994"
            remaining = ' '.join(label.split()[1:])  # Everything after "ON"
            date = parse_datetime(remaining)
            return 'ON', date, date
        elif label.startswith('BEFORE'):
            # Handle cases like "BEFORE February 18, 1994"
            remaining = ' '.join(label.split()[1:])  # Everything after "BEFORE"
            date = parse_datetime(remaining)
            return 'BEFORE', datetime.min, date
        elif label.startswith('AFTER'):
            # Handle cases like "AFTER February 18, 1994"
            remaining = ' '.join(label.split()[1:])  # Everything after "AFTER"
            date = parse_datetime(remaining)
            return 'AFTER', date, datetime.max
        elif 'TO' in label:
            # Split more carefully for TO cases
            if 'TO UNKNOWN DATE' in label.upper():
                # Handle cases like "10/92 TO UNKNOWN DATE"
                start_part = label.split('TO')[0].strip()
                d1 = parse_datetime(start_part)
                return 'TO', d1, datetime.max
            else:
                # Handle cases like "MARCH 14 TO MARCH 21, 1994"
                parts = label.split('TO')
                if len(parts) == 2:
                    start_part = parts[0].strip()
                    end_part = parts[1].strip()
                    d1 = parse_datetime(start_part)
                    d2 = parse_datetime(end_part)
                    return 'TO', d1, d2
    except (ValueError, IndexError) as e:
        print(f"Error parsing date from label '{label}': {e}")
        return None
    
    return None

# Function to check interval overlap
def intervals_overlap(gt_interval, pred_interval):
    if not gt_interval or not pred_interval:
        return False
    label_gt, start_gt, end_gt = gt_interval
    label_pred, start_pred, end_pred = pred_interval
    return (label_gt==label_pred) & (max(start_gt, start_pred) <= min(end_gt, end_pred))


def evaluate_azure_annotations(annotations, gold_standard):
    """
    Evaluate Azure OpenAI annotations against the gold standard
    """
    # Evaluate
    # Initialize counters for each category
    categories = ["ON", "BEFORE", "AFTER", "TO"]
    category_totals = {cat: 0 for cat in categories}
    category_strict_matches = {cat: 0 for cat in categories}
    category_relaxed_matches = {cat: 0 for cat in categories}
    # Overall counters
    total = 0
    strict_match = 0
    relaxed_match = 0

    for record_id, entries in gold_standard.items():
        total += len(entries)
        for event_id in entries.keys():
            gt = parse_label(entries[event_id]['formatted_time_range'])
            pred = parse_label(annotations.get(record_id, {}).get(event_id, {}).get('datetime', None))
            
            if gt is None:
                continue
                
            category = gt[0]
            category_totals[category] += 1

            is_strict_match = gt == pred
            is_relaxed_match = is_strict_match or (intervals_overlap(gt, pred))
            if is_strict_match:
                strict_match += 1
                category_strict_matches[category] += 1
            if is_relaxed_match:
                relaxed_match += 1
                category_relaxed_matches[category] += 1

    # Calculate overall accuracy
    strict_accuracy = strict_match / total if total > 0 else 0
    relaxed_accuracy = relaxed_match / total if total > 0 else 0

    # Print results without pandas
    print("Overall Results:")
    print(f"Total Samples: {total}")
    print(f"Strict Matches: {strict_match}")
    print(f"Strict Accuracy: {strict_accuracy:.4f}")
    print(f"Relaxed Matches: {relaxed_match}")
    print(f"Relaxed Accuracy: {relaxed_accuracy:.4f}")
    
    print("\nCategory Breakdown:")
    print(f"{'Category':<10} {'Total':<8} {'Strict Matches':<15} {'Strict Acc':<12} {'Relaxed Matches':<16} {'Relaxed Acc':<12}")
    print("-" * 80)
    
    # for cat in categories:
    #     cat_total = category_totals[cat]
    #     strict_acc = category_strict_matches[cat] / cat_total if cat_total > 0 else 0
    #     relaxed_acc = category_relaxed_matches[cat] / cat_total if cat_total > 0 else 0
    #     print(f"{cat:<10} {cat_total:<8} {category_strict_matches[cat]:<15} {strict_acc:<12.4f} {category_relaxed_matches[cat]:<16} {relaxed_acc:<12.4f}")


if __name__ == "__main__":
    gold_standard = get_gold_standard()
    # print(label_data)
    # jsonl_file_path_all = '/home/ubuntu/work/Temporal_relation/llm_qa/qa_azure_results/timeline_azure_bulk_openai_time_all_gpt4.jsonl'
    # jsonl_file_path_ind = '/home/ubuntu/work/Temporal_relation/llm_qa/qa_azure_results/timeline_azure_bulk_openai_notime_individual_gpt4.jsonl'
    # # results = parse_jsonl_extract_content(jsonl_file_path_ind)

    # annotations_all = get_azure_annotation_all(jsonl_file_path_all)
    # annotations_ind = get_azure_annotation_individual(jsonl_file_path_ind)
    # # print(len(annotations))
    # # for key in annotations:
    #     # print(f"{key}: {annotations[key]}")
    # # print(set(annotations_ind.keys()) -  (set(annotations_all.keys())))
    # # print(results['doc-541'])
    # # print(list(label_data.items())[:1])
    # # print(list(annotations_ind.items())[:1])
    # # print(list(annotations_all.items())[:1])
    # # print(results['doc-491-task-E9'])
    # # print(annotations_ind['422'])
    # evaluate_azure_annotations(annotations_all, gold_standard)
    # evaluate_azure_annotations(annotations_ind, gold_standard)


    for i in range(1, 9):
        path = "/home/ubuntu/work/Temporal_relation/llm_qa/GPT4/" + 'file' + str(i) + '/'
        directory = os.fsencode(path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if 'results' in filename:
                if "all" in filename: 
                    # print(os.path.join(directory, filename))
                    annotations = get_azure_annotation_all(path+filename)
                else:
                    annotations = get_azure_annotation_individual(path+filename)
                print(filename)
                evaluate_azure_annotations(annotations, gold_standard)


    


