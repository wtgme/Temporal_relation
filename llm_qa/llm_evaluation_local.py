import llm_qa_data_builder
import json
from datetime import datetime
from collections import Counter, defaultdict
# import pandas as pd
import os
import re

def get_gold_standard(folder='timeline_training'):
    path = '/home/ubuntu/work/Temporal_relation/'
    data_dir = path + "data/" + folder + "/"
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
                if content is None:
                    content = (data.get('response', {})
                              .get('body', {})
                              .get('choices', [{}])[0]
                              .get('message', {})
                              .get('reasoning_content', None))
              

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
            events = annotations[cid]

            if 'events' in events:
                events = events['events']
            for event in events:
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
    """Parse a date string into a datetime object, handling various formats."""
    import re
    
    # Handle various date formats and normalize them
    temporal_modifiers = [
        ('APPROXIMATELY ', ''), ('DURING ', ''), ('BETWEEN ', ''), ('FROM ', ''), 
        ('MIDWAY THROUGH ', ''), ('ROUGHLY ', ''), ('ABOUT ', ''), ('AROUND ', ''),
        ('-??', '-01'), ('-XX', '-01'), ('-xx', '-01'), ('-00', '-01')
    ]
    
    for old, new in temporal_modifiers:
        date_str = date_str.replace(old, new)
    
    # Handle "BETWEEN ... AND ..." pattern
    between_and_pattern = r'BETWEEN\s+(.+?)\s+AND\s+(.+)'
    match = re.search(between_and_pattern, date_str, re.IGNORECASE)
    if match:
        start_date, end_date = match.groups()
        date_str = f"{start_date.strip()} TO {end_date.strip()}"
        
    # Remove time components
    if "T" in date_str:
        date_str = date_str.split('T')[0]
    
    # Remove time patterns like "AT 21:16", "14:30", etc.
    time_patterns = [
        r'\s+AT\s+\d{1,2}:\d{2}(:\d{2})?',
        r'\s+\d{1,2}:\d{2}(:\d{2})?\s*[ap]\.?m\.?',
        r'\s+\d{1,2}:\d{2}(:\d{2})?',
        r'\s+at\s+\d{1,2}:\d{2}(:\d{2})?\s*[ap]\.?m\.?',
    ]
    
    for pattern in time_patterns:
        date_str = re.sub(pattern, '', date_str, flags=re.IGNORECASE)
    
    date_str = date_str.strip()

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
        # Handle complex temporal expressions like "2 TO 3 DAYS BEFORE 2011-06-10"
        relative_patterns = [
            r'^\d+\s+TO\s+\d+\s+(DAYS?|WEEKS?|MONTHS?|YEARS?)\s+BEFORE\s+(.+)$',
            r'^(TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\s+TO\s+(TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\s+(DAYS?|WEEKS?|MONTHS?|YEARS?)\s+BEFORE\s+(.+)$',
            r'^(TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\s+(DAYS?|WEEKS?|MONTHS?|YEARS?)\s+(PRIOR\s+TO|BEFORE)\s+(.+)$',
            r'^\d+\s+(DAYS?|WEEKS?|MONTHS?|YEARS?)\s+(PRIOR\s+TO|BEFORE)\s+(.+)$'
        ]
        
        for pattern in relative_patterns:
            match = re.match(pattern, date_str, re.IGNORECASE)
            if match:
                # Extract the base date from the end of the expression
                base_date_str = match.groups()[-1]  # Last group is always the base date
                date_str = base_date_str
                break
        
        if not any(re.match(pattern, date_str, re.IGNORECASE) for pattern in relative_patterns):
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
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = year_match.group()
            return datetime.strptime(f"{year}-01-01", '%Y-%m-%d')
        raise e

def label_starts_with_date_pattern(label):
    """
    Check if a label starts with any of the defined date patterns
    
    Args:
        label (str): The label to check
        
    Returns:
        bool: True if label starts with a date pattern, False otherwise
    """
    if not label:
        return False
        
    date_patterns = [
        r'^(18|19|20)\d{2}\b',  # Years 1800-2099 at start
        r'^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{2}|\d{4})\b',  # MM/DD/YY or MM/DD/YYYY at start
        r'^(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](\d{2}|\d{4})\b',  # DD/MM/YY or DD/MM/YYYY at start
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Full month names at start
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',  # Abbreviated month names at start
        r'^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b',  # MM/DD or MM-DD at start
    ]
    
    return any(re.search(pattern, label, re.IGNORECASE) for pattern in date_patterns)


def parse_label(label):
    # print(f"Parsing label: {label}")
    if (label is None) or (label == 'None'):
        return None
    
    # Handle case where label is a dictionary (extract the value we need)
    if isinstance(label, dict):
        # Try to get 'datetime' field from the dictionary
        if 'datetime' in label:
            label = label['datetime']
        else:
            # If no 'datetime' key, return None
            return None
    
    # Ensure label is a string
    if not isinstance(label, str):
        return None
    
    # Check for 'None' string after conversion
    if 'None' in label:
        return None
    
    # Handle "UNKNOWN DATE" case
    if 'UNKNOWN DATE' in label.upper():
        return None
    
    # Check if label contains any date-like information
    # Look for years (4 digits), months (1-12), or common date patterns
    
    
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
    
    # Normalize temporal modifiers first
    temporal_modifiers = ['APPROXIMATELY ', 'ROUGHLY ', 'ABOUT ', 'AROUND ']
    for modifier in temporal_modifiers:
        if label.startswith(modifier):
            label = label[len(modifier):].strip()
            break
    
    # Handle "BETWEEN ... AND ..." pattern early - convert to "... TO ..." format
    between_and_pattern = r'^BETWEEN\s+(.+?)\s+AND\s+(.+)$'
    match = re.match(between_and_pattern, label, re.IGNORECASE)
    if match:
        start_date, end_date = match.groups()
        label = f"{start_date.strip()} TO {end_date.strip()}"
    
    # Normalize temporal preposition variants
    preposition_replacements = {
        'AFTER ON': 'AFTER',
        'AFTER AT': 'AFTER',
        'POST': 'AFTER',
        'SINCE': 'AFTER',
        'AFTER OR ON': 'AFTER',
        'AFTER OR AT': 'AFTER',
        'AFTER OR IN': 'AFTER',
        'BEFORE ON': 'BEFORE',
        'BEFORE AT': 'BEFORE',
        'BEFORE OR ON': 'BEFORE',
        'BEFORE OR AT': 'BEFORE',
        'BEFORE OR IN': 'BEFORE',
        'BACK IN': 'ON',
        'BY': 'BEFORE',
        'AT': 'ON',
        'IN': 'ON'
    }
    
    # Apply replacements
    for old_prefix, new_prefix in preposition_replacements.items():
        if label.startswith(old_prefix):
            label = label.replace(old_prefix, new_prefix, 1)  # Replace only first occurrence
            break
    
    # Handle labels that start with date patterns
    if label_starts_with_date_pattern(label) and 'TO' not in label.upper():
        label = 'ON ' + label  # Prepend "ON" if it starts with a date pattern
 
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
def extract_datetime_from_response(annotation_data, event_id):
    """
    Extract datetime value from various response formats
    
    Args:
        annotation_data: The response data (can be dict, list, or string)
        event_id: The event ID we're looking for
        
    Returns:
        tuple: (datetime_value, clues) or (None, None) if not found
    """
    # Handle direct string responses
    if isinstance(annotation_data, str):
        return annotation_data, ''
    
    # Handle None or empty responses
    if not annotation_data:
        return None, ''
    
    # Handle list of responses (take first one)
    if isinstance(annotation_data, list):
        if len(annotation_data) > 0:
            annotation_data = annotation_data[0]
        else:
            return None, ''
    
    # Handle dictionary responses
    if isinstance(annotation_data, dict):
        # Format 1: Simple format {'datetime': '...', 'clues': '...'}
        if 'datetime' in annotation_data:
            datetime_value = annotation_data['datetime']
            clues = annotation_data.get('clues', '')
            return datetime_value, clues
        
        # Format 2: Nested event format {'event': {'datetime': '...', 'clues': '...'}}
        elif 'event' in annotation_data and isinstance(annotation_data['event'], dict):
            event_data = annotation_data['event']
            datetime_value = event_data.get('datetime')
            clues = event_data.get('clues', '')
            return datetime_value, clues
        
        # Format 3: Multiple events format {'events': [...]}
        elif 'events' in annotation_data and isinstance(annotation_data['events'], list):
            events = annotation_data['events']
            
            # Try to find the event by matching event_id with event name/type
            for event in events:
                if isinstance(event, dict):
                    # Check if event matches the event_id (remove 'E' prefix for comparison)
                    event_name = event.get('event', '')
                    if (event_name.upper() == event_id.upper() or 
                        event_name.upper() == event_id.replace('E', '').upper() or
                        event_id.replace('E', '').upper() in event_name.upper()):
                        datetime_value = event.get('datetime')
                        clues = event.get('clues', '')
                        return datetime_value, clues
            
            # If no specific match found, take the first event
            if events and isinstance(events[0], dict):
                datetime_value = events[0].get('datetime')
                clues = events[0].get('clues', '')
                return datetime_value, clues
    
    return None, ''

def intervals_overlap(gt_interval, pred_interval):
    if not gt_interval or not pred_interval:
        return False
    label_gt, start_gt, end_gt = gt_interval
    label_pred, start_pred, end_pred = pred_interval
    # label should match, as before 1990 and after 1989 overlap but do not align. 
    return (label_gt==label_pred) & (max(start_gt, start_pred) <= min(end_gt, end_pred))


def evaluate_azure_annotations(annotations, gold_standard, show_detailed_analysis=False):
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
    
    # Error tracking
    parsing_errors = []
    mismatches = []
    gt_parsing_failed = 0
    pred_parsing_failed = 0

    for record_id, entries in gold_standard.items():
        total += len(entries)
        for event_id in entries.keys():
            gt_label = entries[event_id]['formatted_time_range']
            
            # Parse ground truth
            try:
                gt = parse_label(gt_label)
                if gt is None:
                    gt_parsing_failed += 1
                    continue
            except Exception as e:
                gt_parsing_failed += 1
                parsing_errors.append({
                    'type': 'ground_truth',
                    'record_id': record_id,
                    'event_id': event_id,
                    'label': gt_label,
                    'error': str(e)
                })
                continue
            
            # Get annotation data and extract datetime/clues using the new function
            annotation_data = annotations.get(record_id, {}).get(event_id, {})
            datetime_value, clues = extract_datetime_from_response(annotation_data, event_id)
            
            # Parse prediction
            try:
                pred = parse_label(datetime_value)
                if pred is None and datetime_value is not None:
                    print(f"Error parsing prediction for record {record_id}, event {event_id}: {datetime_value}")
                    pred_parsing_failed += 1
            except Exception as e:
                pred_parsing_failed += 1
                parsing_errors.append({
                    'type': 'prediction',
                    'record_id': record_id,
                    'event_id': event_id,
                    'label': datetime_value,
                    'error': str(e)
                })
                print(f"Error parsing prediction for record {record_id}, event {event_id}: {datetime_value}")
                pred = None
            
            if gt is None:
                continue
                
            category = gt[0]
            category_totals[category] += 1

            is_strict_match = gt == pred
            is_relaxed_match = is_strict_match or (intervals_overlap(gt, pred))
            
            # Track mismatches for detailed analysis
            if not is_strict_match:
                mismatches.append({
                    'record_id': record_id,
                    'event_id': event_id,
                    'category': category,
                    'gt_label': gt_label,
                    'gt_parsed': gt,
                    'pred_label': datetime_value,
                    'pred_parsed': pred,
                    'pred_clues': clues,
                    'is_relaxed_match': is_relaxed_match
                })
            
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
    
    for cat in categories:
        cat_total = category_totals[cat]
        strict_acc = category_strict_matches[cat] / cat_total if cat_total > 0 else 0
        relaxed_acc = category_relaxed_matches[cat] / cat_total if cat_total > 0 else 0
        print(f"{cat:<10} {cat_total:<8} {category_strict_matches[cat]:<15} {strict_acc:<12.4f} {category_relaxed_matches[cat]:<16} {relaxed_acc:<12.4f}")
    
    # Show detailed analysis if requested
    if show_detailed_analysis:
        print(f"\n{'='*80}")
        print("DETAILED ERROR ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\nPARSING STATISTICS:")
        print(f"Ground truth parsing failed: {gt_parsing_failed}")
        print(f"Prediction parsing failed: {pred_parsing_failed}")
        print(f"Total mismatches: {len(mismatches)}")
        
        # Show parsing errors
        if parsing_errors:
            print(f"\nPARSING ERRORS ({len(parsing_errors)} total):")
            error_types = Counter([err['type'] for err in parsing_errors])
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count} errors")
            
            print(f"\nSample parsing errors:")
            for i, error in enumerate(parsing_errors[:5]):
                print(f"  {i+1}. [{error['type']}] {error['record_id']}/{error['event_id']}")
                print(f"     Label: '{error['label']}'")
                print(f"     Error: {error['error']}")
                print()
        
        # Analyze mismatches by category
        if mismatches:
            print(f"\nMISMATCH ANALYSIS BY CATEGORY:")
            category_mismatches = {}
            for mismatch in mismatches:
                cat = mismatch['category']
                if cat not in category_mismatches:
                    category_mismatches[cat] = []
                category_mismatches[cat].append(mismatch)
            
            for category, cat_mismatches in category_mismatches.items():
                print(f"\n{category} CATEGORY ({len(cat_mismatches)} mismatches):")
                
                # Show prediction patterns
                pred_patterns = Counter([str(m['pred_parsed']) for m in cat_mismatches])
                print(f"  Most common predictions:")
                for pattern, count in pred_patterns.most_common(5):
                    print(f"    {pattern}: {count} times")
                
                # Show specific examples
                print(f"  Sample mismatches:")
                for i, mismatch in enumerate(cat_mismatches[:3]):
                    print(f"    {i+1}. {mismatch['record_id']}/{mismatch['event_id']}")
                    print(f"       GT: '{mismatch['gt_label']}' -> {mismatch['gt_parsed']}")
                    print(f"       Pred: '{mismatch['pred_label']}' -> {mismatch['pred_parsed']}")
                    print(f"       Pred clues: '{mismatch['pred_clues']}'")
                    print(f"       Relaxed match: {mismatch['is_relaxed_match']}")
                    print()
            
            # Show most common mismatch patterns
            print(f"\nMOST COMMON MISMATCH PATTERNS:")
            mismatch_patterns = Counter([
                f"GT:{m['gt_parsed'][0] if m['gt_parsed'] else 'None'} -> Pred:{m['pred_parsed'][0] if m['pred_parsed'] else 'None'}"
                for m in mismatches
            ])
            for pattern, count in mismatch_patterns.most_common(10):
                print(f"  {pattern}: {count} times")
        
        print(f"\n{'='*80}")
        print("END OF DETAILED ANALYSIS")
        print(f"{'='*80}")
        
    return strict_accuracy, relaxed_accuracy, category_totals, category_strict_matches, category_relaxed_matches


if __name__ == "__main__":
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # folder = 'timeline_training_llm_update'
    folder = 'timeline_training'

    gold_standard = get_gold_standard(folder)
    print(f"Gold standard loaded with {len(gold_standard)} records.")
    
    # OpenAI Azure annotations
    # path = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_results/timeline_training/" 
    # Local LLM annotations
    path = "/home/ubuntu/work/Temporal_relation/llm_qa/qa_results/local/" 
    
    
    # path = path + folder + '/'
    print(f"Processing files in directory: {path}")
    directory = os.fsencode(path)
    results = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if 'results' in filename: 
            # and 'timeline_azure_bulk_DeepSeek-R1-Distill-Llama-8B_notime_individual_results.jsonl' in filename:
            tokens = filename.split('_')
            model = tokens[3]
            time = 'no_time' if 'notime' in filename else 'time'
            section = 'sections' if 'sections' in filename else 'no_sections'
            if "all" in filename: 
                annotations = get_azure_annotation_all(path+filename)
                mode = 'all'
            else:
                annotations = get_azure_annotation_individual(path+filename)
                mode = 'individual'
            print(filename)
            # print(annotations)
            strict_accuracy, relaxed_accuracy, category_totals, category_strict_matches, category_relaxed_matches = evaluate_azure_annotations(annotations, gold_standard, show_detailed_analysis=False)
            results.append([model, mode, time, section, filename, strict_accuracy, relaxed_accuracy])
    
    df = pd.DataFrame(results, columns=['Model', 'Mode', 'Time', 'Section', 'Filename', 'Strict Accuracy', 'Relaxed Accuracy'])
    
    # Create combined setting identifier
    df['Setting'] = df['Model'] + '_' + df['Mode'] + '_' + df['Time'] + '_' + df['Section']
    
    # Reshape data for combined plotting
    df_melted = df.melt(
        id_vars=['Setting'],
        value_vars=['Strict Accuracy', 'Relaxed Accuracy'],
        var_name='Accuracy Type',
        value_name='Accuracy'
    )
    print(df_melted.head())
    df_melted = df_melted.sort_values(by='Accuracy')
    # Single plot with both accuracy types per setting
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x='Setting', y='Accuracy', hue='Accuracy Type')

    plt.title('Model Performance: Strict vs Relaxed Accuracy by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.legend(title='Accuracy Type')
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/work/Temporal_relation/llm_qa/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to accuracy_comparison.png")

    # Save results
    df.to_csv(f'/home/ubuntu/work/Temporal_relation/llm_qa/{folder}.csv', index=False)
    print(f"Results saved to {folder}.csv")



# change folder = 'timeline_training'
# python llm_evaluation.py > evaluation.log
# change folder = 'timeline_training_llm_update'
# python llm_evaluation.py > evaluation_llm_update.log

