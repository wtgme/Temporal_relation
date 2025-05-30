#!/usr/bin/env python3
import json
from datetime import datetime, timedelta
import pandas as pd
import llm_qa_data_builder
# import ace_tools as tools

def load_results(filepath):
    """Load the JSON results file containing LLM datetime annotations. 
    this was for vllm results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_gold_standard():
    path = '/home/ubuntu/work/Temporal_relation/'
    data_dir = path + "data/timeline_training/"
    results = {}

    data = llm_qa_data_builder.data_load(data_dir, True)

    for key in data:
        ground_truth = []
        text = data[key]['label']
        event_start_time = data[key]['starttime']
        for event in event_start_time:
            ground_truth.append({
                "node_id": event['node_id'],
                "formatted_time_range": event['formatted_time_range'],
            })
        # print(ground_truth)
        results[key] = {'ground_truth': ground_truth}
    return results


def extract_datetime_annotations(results):
    """Extract all datetime annotations from the results."""
    datetime_annotations = {}
    
    for record_id, entries in results.items():            
        annotations = []
        for entry in entries:
            inference = json.loads(entry["llm_inference"])
            annotations.append({
                'event_id': entry['event_id'],
                'ground_truth': entry['ground_truth'],
                "inference": inference["datetime"],
                # "clues": inference.get("clues", "")
            })
        datetime_annotations[record_id] = annotations
    # print(list(datetime_annotations.keys()))  
    return datetime_annotations


def parse_datetime(date_str):
    # print(date_str)
    """Parse a date string into a datetime object, handling various formats."""
    # Handle various date formats and normalize them
    
    # Replace placeholder characters
    if "-??" in date_str:
        date_str = date_str.replace("-??", "-01")
        
    # Handle '00' day or month
    if "-00" in date_str:
        date_str = date_str.replace("-00", "-01")
        
    # Remove time component if present
    if "T" in date_str:
        date_str = date_str.split('T')[0]
        
    # Handle prefixes like "ON", "BEFORE", etc.
    if ' ' in date_str:
        date_str = date_str.split(' ')[0]
    if ',' in date_str:
        date_str = date_str.split(',')[0]

    # Handle MM/DD/YYYY format (like 08/15/1998)
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            # Convert MM/DD/YYYY to YYYY-MM-DD
            month, day, year = parts
            date_str = f"{year}-{month}-{day}"
    
    # Handle MM-DD-YY format (like 09-07-93)
    if len(date_str.split('-')) == 3:
        parts = date_str.split('-')
        # Only transform if first part is not a 4-digit year (YYYY-MM-DD)
        if len(parts[0]) != 4 and len(parts[2]) == 2:  # MM-DD-YY format
            # Convert to YYYY-MM-DD format (assuming 19xx for years before 50, 20xx for years after)
            year = int(parts[2])
            year_prefix = '19' if year >= 50 else '20'
            date_str = f"{year_prefix}{parts[2]}-{parts[0]}-{parts[1]}"
    
    # Now parse based on dash count
    if '-' not in date_str:
        # Year only
        date_str = f"{date_str}-01-01"
        date = datetime.strptime(date_str, '%Y-%m-%d')
    elif len(date_str.split('-')) == 2:  
        # Format is YYYY-MM
        date_str = f"{date_str}-01"
        date = datetime.strptime(date_str, '%Y-%m-%d')
    elif len(date_str.split('-')) == 3:  
        # Format is YYYY-MM-DD
        date = datetime.strptime(date_str, '%Y-%m-%d')
    # print('-->', date)
    return date



# Helper function to convert labels to intervals
def parse_label(label):
    if label is None or label == 'None':
        return None
    if label.startswith('AFTER ON'):
        label = label.replace('AFTER ON', 'AFTER')
    elif label.startswith('BEFORE ON'):
        label = label.replace('BEFORE ON', 'BEFORE')
    elif label.startswith('AT'):
        label = label.replace('AT', 'ON')

    if label.startswith('ON'):
        date = parse_datetime(label.split()[1])
        return 'ON', date, date
    elif label.startswith('BEFORE'):
        date = parse_datetime(label.split()[1])
        return 'BEFORE', datetime.min, date
    elif label.startswith('AFTER'):
        date = parse_datetime(label.split()[1])
        return 'AFTER', date, datetime.max
    elif 'TO' in label:
        d1, d2 = [parse_datetime(d.strip()) for d in label.split('TO')]
        return 'TO', d1, d2
    return None

# Function to check interval overlap
def intervals_overlap(gt_interval, pred_interval):
    if not gt_interval or not pred_interval:
        return False
    label_gt, start_gt, end_gt = gt_interval
    label_pred, start_pred, end_pred = pred_interval
    return (label_gt==label_pred) & (max(start_gt, start_pred) <= min(end_gt, end_pred))


def main(file):
    # Load and process results
    print(f"Loading results from {file}...")
    results = load_results(file)
    
    print("Extracting and analyzing datetime annotations...")
    annotations = extract_datetime_annotations(results)
    # print(annotations)
    
    print("Calculating accuracy...")

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

    for record_id, entries in annotations.items():
        # print(f"Record ID: {record_id}, Entries: {len(entries)}")
        total += len(entries)
        for entry in entries:
            # print(entry)
            gt = parse_label(entry['ground_truth'])
            pred = parse_label(entry['inference'])
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

    # Calculate category-specific accuracy
    category_strict_acc = {}
    category_relaxed_acc = {}
    
    for cat in categories:
        cat_total = category_totals[cat]
        if cat_total > 0:
            category_strict_acc[cat] = category_strict_matches[cat] / cat_total
            category_relaxed_acc[cat] = category_relaxed_matches[cat] / cat_total
        else:
            category_strict_acc[cat] = 0
            category_relaxed_acc[cat] = 0

    # Create overall results dataframe
    results_df = pd.DataFrame({
        'Total Samples': [total],
        'Strict Matches': [strict_match],
        'Strict Accuracy': [strict_accuracy],
        'Relaxed Matches': [relaxed_match],
        'Relaxed Accuracy': [relaxed_accuracy]
    })
    
    # Create category breakdown dataframe
    category_df = pd.DataFrame({
        'Category': categories,
        'Total': [category_totals[cat] for cat in categories],
        'Strict Matches': [category_strict_matches[cat] for cat in categories],
        'Strict Accuracy': [category_strict_acc[cat] for cat in categories],
        'Relaxed Matches': [category_relaxed_matches[cat] for cat in categories],
        'Relaxed Accuracy': [category_relaxed_acc[cat] for cat in categories]
    })
    print("Overall Results:")
    print(results_df)
    print("\nCategory Breakdown:")
    print(category_df)

if __name__ == "__main__":
    print(datetime.min, datetime.max)
    path = '/home/ubuntu/work/Temporal_relation/llm_qa/qa_results/'
    files = [
            #  'timeline_training_QwQ-32B-AWQ_results_notime_all.json',
            'timeline_training_QwQ-32B-AWQ_results_notime_individual.json',
             'timeline_training_QwQ-32B-AWQ_results_notime_individual_sections.json',
             'timeline_training_QwQ-32B-AWQ_results_time_individual.json', 
             'timeline_training_QwQ-32B-AWQ_results_time_individual_sections.json'
             ]
    for file in files:
        print(f"\n\nProcessing file: {path + file}")
        main(path + file)
# Errors 
# {'event_id': 'E34', 'ground_truth': '1998-05-09 TO 1998-05-25', 'inference': 'AFTER ON 1998-05-10'}
# {'event_id': 'E44', 'ground_truth': 'AT 2011-09-24T14:00', 'inference': 'ON 2011-09-24'}
# {'event_id': 'E15', 'ground_truth': '2011-09-26 TO 2011-09-27T10:55', 'inference': 'ON 2011-09-26'}
# {'event_id': 'E3', 'ground_truth': 'BEFORE 1989-12', 'inference': 'None'}
# {'event_id': 'E4', 'ground_truth': 'BEFORE 1989', 'inference': 'None'}
# {'event_id': 'E21', 'ground_truth': 'BEFORE 1997-04-07', 'inference': 'ON 1997-01-??'}
# ValueError: time data '1996-08-00' does not match format '%Y-%m-%d'
# Total Samples are not the same