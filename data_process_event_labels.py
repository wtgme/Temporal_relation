import xmltodict, json, re
import os
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
 

def data_loader(data_path):
    """data: {
        filename: {
            "ClinicalNarrativeTemporalAnnotation":{
                "TEXT": text,
                "TAGS": {
                    'EVENT': list({
                        'id':id, 
                        'start':start position, 
                        'end': end position, 
                        'text': text, 
                        'modality': whether an EVENT actually occurred or not ('FACTUAL','CONDITIONAL','POSSIBLE','PROPOSED') 
                        'polarity': positive or negative ('POS'/'NEG'), 
                        'type': event type ('TEST','PROBLEM','TREATMENT','CLINICAL_DEPT','EVIDENTIAL','OCCURENCE')
                        }),
                    'TIMEX3':list({
                        'id':id, 
                        'start':start position, 
                        'end': end position, 
                        'text': text, 
                        'type': event type ('DATE','TIME','DURATION','FREQUENCY'),
                        'val': regularised time expression,
                        'mod': modifier for regularised time expression,
                        }),
                    'TLINK':list({
                        'id':id, 
                        'fromID': head event/timex3 id, 
                        'fromText': head event/timex3 text, 
                        'toID': tail event/timex3 id, 
                        'toText': tail event/timex3 text,
                        'type': temporal relation type ('BEFORE', 'AFTER', 'OVERLAP'),
                        }),
                    'SECTIME': list({
                        'id': id, 
                        'start':start position, 
                        'end': end position, 
                        'text': text, 
                        'type': 'ADMISSION'/'DISCHARGE',
                        'dvalue': regularised date time,
                        }),
                }
            }
        }
    }
    """
    data = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".xml"): 
            f = (os.path.join(data_path, filename))
#             print(f)
            fb = open(f, "rb").read().decode(encoding="utf-8")
#     invalid character '&' https://github.com/martinblech/xmltodict/issues/277
            fb = fb.replace('&', '&amp;')
            dic = xmltodict.parse(fb, attr_prefix='')
#     restore orginal character "&"
            dic['ClinicalNarrativeTemporalAnnotation']['TEXT'] = dic['ClinicalNarrativeTemporalAnnotation']['TEXT'].replace('&amp;', '&')
            data[filename] = (dic)
    return data

def process_text_labels(text, events, include_label=False):
    """
    给文本中的指定位置插入标签，标记事件。考虑事件overlap的可能性。

    参数:
    - text (str): 原始文本
    - events (List[Dict]): 每个事件包含 'label', 'start', 'end'

    返回:
    - str: 处理后的文本，包含标签
    """
    insertions = []

    for idx, event in enumerate(events): # start, end统一排序，以适应overlap的情况
        label = event['label']
        eid = event['id']
        start = event['start']-1
        end = event['end']-1

        start_tag = f"<{eid}:{label}>" if include_label else f"<{eid}>"
        end_tag = f"</{eid}>"

        insertions.append((start, start_tag))  # 插入起始标签
        insertions.append((end, end_tag))      # 插入结束标签

    # 按照插入位置排序，位置相同的先插后面的（所以 reverse）
    insertions.sort(key=lambda x: x[0], reverse=True)

    modified_text = text
    for pos, tag in insertions:
        modified_text = modified_text[:pos] + tag + modified_text[pos:]


    return modified_text


def extract_events_from_labeled_text(text, include_label=False):
    """
    从插入了标签的文本中抽取出所有事件。用于检查。
    支持嵌套和overlap的事件提取器。
 
    返回:
    - List[Dict]，每个包含：eid, text, label (if include_label)
    """
 
    stack = []
    events = []
    idx = 0
    while idx < len(text):
        # 匹配起始标签
        if include_label:
            start_match = re.match(r"<E(\d+):([a-zA-Z_]+)>", text[idx:])
        else:
            start_match = re.match(r"<E(\d+)>", text[idx:])
        if start_match:
            event_id = 'E'+start_match.group(1)
            label = start_match.group(2) if include_label else None
            tag_len = start_match.end()

            stack.append({
                "id": event_id,
                "label": label,
                "start_idx": idx,
                "text_start": idx + tag_len
            })
            idx += tag_len
            continue

        # 匹配结束标签
        end_match = re.match(r"</E(\d+)>", text[idx:])
        if end_match:
            event_id = 'E' + end_match.group(1)
            tag_len = end_match.end()

            # 从栈中找到对应的起始标签
            for i in reversed(range(len(stack))):
                if stack[i]['id'] == event_id:
                    start_info = stack.pop(i)
                    event_text = text[start_info['text_start']:idx]
                    event_text = re.sub(r"</?E\d+(?::[a-zA-Z_]+)?>", "", event_text)
                    events.append({
                        "id": event_id,
                        "label": start_info['label'],
                        "text": event_text,
                        "start": start_info['text_start'],
                        "end": idx
                    })
                    break
            idx += tag_len
            continue

        idx += 1

    return events

def check_results(src_folder, valid_types = ['PROBLEM', 'TEST', 'TREATMENT']):
    """
    从修改过的文本中提取事件，与原始事件进行对比，检查是否正确。
    """
    data = data_loader(src_folder)
    for filename in data:
        print(filename)
        text = data[filename]['ClinicalNarrativeTemporalAnnotation']['TEXT']
        events = data[filename]['ClinicalNarrativeTemporalAnnotation']['TAGS']['EVENT']
        events = [event for event in events if event['type'].upper() in valid_types]
        orig_events = {}
        for event in events:
            orig_events[event['id']] = {'id': event['id'], 'label': event['type'], 'start': int(event['start']), 'end': int(event['end']), 'text':event['text']}
        
        with open(f"{src_folder}/{filename}.label.txt", "r") as f:
            modified_text = f.read()

        extracted_events = extract_events_from_labeled_text(modified_text)
        for event in extracted_events:
            assert(orig_events[event['id']]['text'] == event['text'])

        


def insert_event_labels(src_folder, valid_types = ['PROBLEM', 'TEST', 'TREATMENT', 'TIME', 'DATE'], file_suffix = 'label.txt'):
    """遍历xml文档, 为每个文档插入事件标签，保存到文件*.xml.label.txt中"""
    data = data_loader(src_folder)
    for filename in data:
        print(filename)
        text = data[filename]['ClinicalNarrativeTemporalAnnotation']['TEXT']
        events = data[filename]['ClinicalNarrativeTemporalAnnotation']['TAGS']['EVENT']
        events += data[filename]['ClinicalNarrativeTemporalAnnotation']['TAGS']['TIMEX3']
        events = [event for event in events if event['type'].upper() in valid_types]
        events = [{'id': event['id'], 'label': event['type'], 'start': int(event['start']), 'end': int(event['end']), 'text':event['text']} for event in events]
        modified_text = process_text_labels(text, events)
        with open(f"{src_folder}/{filename}.{file_suffix}", "w") as f:
            f.write(modified_text)
       


if __name__ == '__main__':
    # insert_event_labels("./data/i2b2/timeline_test/")
    # /home/jovyan/work/Temporal_relation/data/
    insert_event_labels("/home/jovyan/work/Temporal_relation/data/timeline_test/", ['PROBLEM', 'TEST', 'TREATMENT'], 'notime.label.txt')
    insert_event_labels("/home/jovyan/work/Temporal_relation/data/timeline_training/", ['PROBLEM', 'TEST', 'TREATMENT'], 'notime.label.txt')
    # check_results("./data/i2b2/original_training/")