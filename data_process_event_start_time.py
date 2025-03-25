import xmltodict, json, re
import os
import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
from datetime import datetime 
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from collections import defaultdict

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
            f = (os.path.join(data_path, filename)) #             print(f)
            fb = open(f, "rb").read().decode(encoding="utf-8") #     invalid character '&' https://github.com/martinblech/xmltodict/issues/277
            fb = fb.replace('&', '&amp;')
            dic = xmltodict.parse(fb, attr_prefix='') #     restore orginal character "&"
            dic['ClinicalNarrativeTemporalAnnotation']['TEXT'] = dic['ClinicalNarrativeTemporalAnnotation']['TEXT'].replace('&amp;', '&')
            data[filename] = (dic)
    return data

def find_first_regex(text, substrings):
    pattern = '|'.join(map(re.escape, substrings))  # Escape special characters
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.start()
    else:
        raise ValueError("None of the substrings found in the text.")

def build_section_graph(doc_id, data):
    """构建完整的图，不做任何过滤，并补全边的对称关系。
    注：根据时序关系对某些边进行了反向处理，同时对于SIMULTANEOUS，OVERLAP进行了双向处理；
    具体操作的时候，先对所有边构建了反向边；然后删掉了不想要的逆时序边；
    """
    
    text = data[doc_id]['ClinicalNarrativeTemporalAnnotation']['TEXT']
    substrings = ['HOSPITAL COURSE']
    try:
        course_start = find_first_regex(text, substrings)
    except:
        course_start = len(text)

    # 读取事件信息
    events = pd.DataFrame(data[doc_id]['ClinicalNarrativeTemporalAnnotation']['TAGS']['EVENT'])
    events['start'] = events['start'].astype(int)
    events['end'] = events['end'].astype(int)
    events['hospital_course'] = events['start'] > course_start


    # 读取时间信息
    times = pd.DataFrame(data[doc_id]['ClinicalNarrativeTemporalAnnotation']['TAGS']['TIMEX3'])
    times['start'] = times['start'].astype(int)
    times['end'] = times['end'].astype(int)

    # 获取所有需要保留的节点
    nodes_keep = list(events['id']) + list(times['id'])

    # 读取时间关系（TLINK）
    all_links = pd.DataFrame(data[doc_id]['ClinicalNarrativeTemporalAnnotation']['TAGS']['TLINK'])
    all_links = all_links.loc[all_links['type'] != '']  # 只保留有意义的边

    # 构建有向图，添加原始变
    G = nx.from_pandas_edgelist(all_links[['fromID', 'toID', 'type']], source='fromID', target='toID', edge_attr=True, create_using=nx.DiGraph())
    
    # 添加事件和时间节点
    events = events.assign(event_or_time="event")
    times = times.assign(event_or_time="time")

    # 设置节点属性
    nx.set_node_attributes(G, events.set_index("id").to_dict(orient="index"))
    nx.set_node_attributes(G, times.set_index("id").to_dict(orient="index"))

    # 定义对称关系映射
    reciprocal_relations = {
        'BEFORE': 'AFTER',
        'AFTER': 'BEFORE',
        'SIMULTANEOUS': 'SIMULTANEOUS',
        'OVERLAP': 'OVERLAP',
        'BEGUN_BY': 'BEGINS',
        'BEGINS': 'BEGUN_BY',
        'ENDED_BY': 'ENDS',
        'ENDS': 'ENDED_BY',
        'DURING': 'COVER',
        'COVER': 'DURING',
        'BEFORE_OVERLAP': 'AFTER_OVERLAP',
        'AFTER_OVERLAP': 'BEFORE_OVERLAP'
    }

    # 遍历所有边，补全对称关系
    edges_to_add = []
    for u, v, data in G.edges(data=True):
        type_u_v = data.get('type', '').upper()  # 规范化大小写
        if u not in G[v]:  # 检查是否已经存在 v → u
            reciprocal_type = reciprocal_relations.get(type_u_v, None)
            if reciprocal_type:
                edges_to_add.append((v, u, reciprocal_type))
    # 添加补全的边
    for v, u, relation in edges_to_add:
        G.add_edge(v, u, type=relation)

    # 过滤逆时序边
    edge_types_to_remove = {'AFTER', 'BEGUN_BY', 'ENDS', 'DURING', 'AFTER_OVERLAP'}
    
    # 过滤出要删除的边
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') in edge_types_to_remove]
    # 删除边
    G.remove_edges_from(edges_to_remove)


    # 只保留 nodes_keep 里的节点
    G = G.subgraph(nodes_keep).copy()

    return G, text

def preprocess_time_overlap_edges(graph):
    tmp_graph = graph.copy()
    # Step 1: 查找类型为 DURATION, TIME, DATE 的节点
    temporal_types = {'DURATION', 'TIME', 'DATE'}
    temporal_nodes = {n for n, attr in graph.nodes(data=True) if attr.get('type').upper() in temporal_types}

    # Step 2: 替换与 temporal_nodes 相连且类型为 OVERLAP 的边为 SIMULTANEOUS
    for node in temporal_nodes:
        for successor in graph.successors(node):
            if graph.has_edge(node, successor):
                edge_data = graph.get_edge_data(node, successor)
                if edge_data.get('type').upper() == 'OVERLAP':
                    tmp_graph[node][successor]['type'] = 'SIMULTANEOUS'
        for predecessor in graph.predecessors(node):
            if graph.has_edge(predecessor, node):
                edge_data = graph.get_edge_data(predecessor, node)
                if edge_data.get('type').upper() == 'OVERLAP':
                    tmp_graph[predecessor][node]['type'] = 'SIMULTANEOUS'
    return tmp_graph

def get_paths_to_time_nodes(G, node, time_node_types=['time', 'date'], filter_overlap=False, filter_mult_time_paths=False, filter_consecutive_durations=False):
    """
    Get paths from `node` to all `time` nodes and from `time` nodes to `node`.
    Optimized for performance by filtering the graph and limiting path length.
    """
    if node not in G:
        return []
    
    # Identify time nodes
    time_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type", "").lower() in time_node_types]

    

    # Pre-filter the graph to remove unnecessary edges
    def filter_graph(graph):
        filtered_graph = graph.copy()
        if filter_overlap:
            
            # Step 3: 删除类型为 OVERLAP 或 UNKNOWN 的边
            edges_to_remove = [
                (u, v) for u, v, attr in filtered_graph.edges(data=True)
                if attr.get('type', 'UNKNOWN') in {'OVERLAP', 'UNKNOWN'}
            ]
            filtered_graph.remove_edges_from(edges_to_remove)
        return filtered_graph

    def path_contains_multiple_time_nodes(path):
        time_node_count = sum(1 for n in path if n in time_nodes)
        return time_node_count > 1
    
    def path_contains_consecutive_durations(path):
        for i in range(len(path) - 1):
            if G.nodes[path[i]].get("type", "").lower() == "duration" and \
               G.nodes[path[i + 1]].get("type", "").lower() == "duration":
                return True
        return False


    def find_out_paths(source, target):
        filtered_graph = filter_graph(G)
        paths = []
        for path in nx.all_simple_paths(filtered_graph, source=source, target=target):
            if filter_mult_time_paths and path_contains_multiple_time_nodes(path):
                continue
            if filter_consecutive_durations and path_contains_consecutive_durations(path):
                continue
            paths.append(path)
        return paths
    
    def find_in_paths(source, target):
        # 反转图的边方向，以便从源节点到前置时间节点
        G_reversed = G.reverse(copy=True)
        filtered_graph = filter_graph(G_reversed)
        paths = []
        for path in nx.all_simple_paths(filtered_graph, source=source, target=target):
            if filter_mult_time_paths and path_contains_multiple_time_nodes(path):
                continue
            if filter_consecutive_durations and path_contains_consecutive_durations(path):
                continue
            # 反转的路径
            paths.append(path[::-1])
        return paths

    # Collect paths
    in_paths = []
    out_paths = []
    out_paths += find_out_paths(source=node, target=time_nodes)
    in_paths += find_in_paths(source=node, target=time_nodes)
    
    return in_paths, out_paths

def parse_duration(duration_str):
    """
    Parse duration strings in the format P[n][Y/M/W/D] or PT[n][H/M/S] into relativedelta or timedelta.
    :param duration_str: str, e.g., 'P2Y', 'P3M', 'P1W', 'P5D', 'PT15M', 'PT2H'
    :return: relativedelta or timedelta object
    """
    duration_str = duration_str.upper()  # Normalize to uppercase
    clean_val = re.sub(r'\.([YMWDHS])', r'\1', duration_str)  # Handle decimal points before units
    if clean_val.startswith("PT"):  # Time components
        pattern = re.compile(r'PT([\d]+(?:\.[\d]+)?)([HMS])', re.IGNORECASE)
    elif clean_val.startswith("P"):  # Date components
        pattern = re.compile(r'P([\d]+(?:\.[\d]+)?)([YMWD])', re.IGNORECASE)
    else:
        print(f"Invalid duration format: {duration_str}. Must start with 'P' or 'PT'.")
        return None

    match = pattern.match(clean_val)
    if match:
        # Extract the numeric value and unit letter
        number = match.group(1)
        unit = match.group(2)
        str_splits = str(number).split('.')
        if len(str_splits) == 2:
            v_i = int(str_splits[0])
            v_d = float(f"0.{str_splits[1]}")
        else:
            v_i = int(str_splits[0])
            v_d = 0

        if unit == "Y":
            return relativedelta(years=v_i, months=round(12 * v_d))
        # Put this before months to avoid conflicts
        elif unit == "M" and clean_val.startswith("PT"):
            return relativedelta(minutes=v_i, seconds=round(60 * v_d))
        elif unit == "M" and clean_val.startswith("P"):
            return relativedelta(months=v_i, days=round(30 * v_d))
        elif unit == "W":
            return relativedelta(weeks=v_i, days=round(7 * v_d))
        elif unit == "D":
            return relativedelta(days=v_i, hours=round(24 * v_d))
        elif unit == "H":
            return relativedelta(hours=v_i, minutes=round(60 * v_d))
        elif unit == "S":
            return relativedelta(seconds=round(v_i + v_d))
    else:
        print(f"Invalid duration format: {duration_str}.")
        return None

def intersect_time_intervals(intervals):
    """
    计算一系列时间区间的交集。
    :param intervals: List of  [['at/before/after', time_val, time_str], ...]
    :return: (start_time, end_time) 交集区间 或 None（若无交集）
    """
    if not intervals:
        return (datetime.min, datetime.max, "NO INFO", [])
    
    starts = [parse(x[1]) if x[0] in ['at','after'] else datetime.min for x in intervals]
    ends = [parse(x[1]) if x[0] in ['at','before'] else datetime.max for x in intervals] 
    for i in range(len(ends)):
        if intervals[i][0] in ['at','before'] and 'T' not in intervals[i][1]:
            ends[i] = ends[i].replace(hour=23, minute=59, second=59) # 如果终止日期是某天，转化成当天23:59:59
 
    start_index, start_value = max(enumerate(starts), key=lambda x: x[1])
    start_text = intervals[start_index][2]
    start_idxes = [i for i in range(len(starts)) if starts[i]==start_value]
    
    end_index, end_value = min(enumerate(ends), key=lambda x: x[1])
    end_text = intervals[end_index][2]
    end_idxes = [i for i in range(len(ends)) if ends[i]==end_value]
    paths = [intervals[i][3] for i in list(set(start_idxes) | set(end_idxes))]


    if start_value <= end_value:
        if start_value == end_value:
            text = 'AT %s'%start_text
            if intervals[start_index][0] == 'after' and intervals[end_index][0] == 'before': # before 和 after 同一个时间点，可能是标注错误
                return None
        elif start_value == datetime.min:
            text = 'BEFORE %s'%end_text
        elif end_value == datetime.max:
            text = 'AFTER %s'%start_text
        else:
            text = "%s TO %s"%(start_text, end_text)
        return (start_value, end_value, text, paths)
    else:
        return None  

def infer_head_start_time(G, out_paths):
    """
    currentnode -> timenode
    根据 out_paths 逻辑推断 head 的开始时间，返回一个一致且最精确的起始时间区间。
    """
    inferred_times = []
   
    def process_duration_node(node, inferred_time, link_type):
        """特殊逻辑处理 duration 节点的开始时间"""
        duration_text = G.nodes[node].get("text")
        duration = parse_duration(G.nodes[node].get("val"))
        
        if link_type == "BEFORE" or link_type == "BEFORE_OVERLAP":
            if inferred_time[0] == 'after': inferred_time = None
            if inferred_time:
                inferred_time[0] == 'before'
                new_time = parse(inferred_time[1]) - duration
                inferred_time[1] = new_time.strftime("%Y-%m-%dT%H:%M:%S")
                inferred_time[2] = f"{inferred_time[2]} - {duration_text}"
        elif link_type == "SIMULTANEOUS":
            pass # unchanged
        elif link_type == "BEGINS":
            pass # unchanged
        elif link_type == "ENDED_BY":
            if inferred_time[0] == 'after': # before和at的情况不变
                inferred_time = None
            else:
                new_time = parse(inferred_time[1]) - duration
                inferred_time[1] = new_time.strftime("%Y-%m-%dT%H:%M:%S")
                inferred_time[2] = f"{inferred_time[2]} - {duration_text}"
        elif link_type == "COVER":
            if inferred_time[0] == 'after': inferred_time = None
            else:
                inferred_time[0] == 'before' 
        elif link_type == "OVERLAP":
            pass # 对duration的overlap事件放宽为和simultaneous一样处理
        else:
            inferred_time = None
            print("Warning: unknown relation type %s"%link_type)

        # if duration and inferred_time:
        #     new_time = parse(inferred_time[1]) - duration
        #     inferred_time[1] = new_time.strftime("%Y-%m-%dT%H:%M:%S")
        #     inferred_time[2] = f"{inferred_time[2]} - {duration_text}"
        # else:
        #     return None

        return inferred_time

    for path in out_paths:
        time_node = path[-1]
        time_attrs = G.nodes[time_node]
        inferred_time = ['at', time_attrs['val'], time_attrs['text'], path]
        
        for i in range(len(path) - 2, -1, -1):  # 逆向遍历路径
            node = path[i]
            node_type = G.nodes[node].get("type", "").upper()
            link_type = G.get_edge_data(path[i], path[i + 1], {}).get("type", "").upper()
            if not node_type or not link_type:
                inferred_time = None
                break
            if node_type == 'DURATION':
                inferred_time = process_duration_node(node, inferred_time, link_type)
            else:
                if link_type == "BEFORE":
                    if inferred_time[0] != 'after': 
                        inferred_time[0] = 'before' 
                    else:
                        inferred_time = None
                elif link_type == "BEFORE_OVERLAP":
                    if inferred_time[0] != 'after':
                        inferred_time[0] = 'before' 
                    else:
                        inferred_time = None
                elif link_type == "SIMULTANEOUS":
                    pass # unchanged
                elif link_type == "BEGINS":
                    pass # unchanged
                elif link_type == "ENDED_BY":
                    if inferred_time[0] != 'after':
                        inferred_time[0] = 'before'
                    else:
                        inferred_time = None
                elif link_type == "COVER":
                    if inferred_time[0] != 'after':
                        inferred_time[0] = 'before' 
                    else:
                        inferred_time = None
                elif link_type == "OVERLAP":
                    inferred_time = None
                    # if inferred_time[0] != 'at': # 如果和一个具体的日期相连，overlap视为simultaneous
                    #     inferred_time = None
                else:
                    inferred_time = None
                    print("Warning: unknown relation type %s"%link_type)
            if not inferred_time:
                break  
        
        if inferred_time:
            inferred_times.append(inferred_time)

    # print(inferred_times)
    # start_time = intersect_time_intervals(inferred_times)
    return inferred_times

def infer_tail_start_time(G, out_paths):
    """
    timenode -> currentnode
    根据 out_paths 逻辑推断 tail 的开始时间，返回一个一致且最精确的起始时间区间。
    """
    inferred_times = []
   
    def process_duration_node(node, inferred_time, link_type):
        """特殊逻辑处理 duration 节点的开始时间"""
        duration_text = G.nodes[node].get("text")
        duration = parse_duration(G.nodes[node].get("val"))
        
        if link_type == "BEFORE" or link_type == "BEFORE_OVERLAP":
            if inferred_time[0] != 'before':
                inferred_time[0] = 'after' 
            else:
                inferred_time = None
        elif link_type == "SIMULTANEOUS":
            pass # unchanged
        elif link_type == "BEGINS":
            pass # unchanged
        elif link_type == "ENDED_BY":
            if inferred_time[0] != 'before':
                inferred_time[0] = 'after' 
            else:
                inferred_time = None
        elif link_type == "COVER":
            if inferred_time[0] != 'before':
                inferred_time[0] = 'after' 
            else:
                inferred_time = None
        elif link_type == "OVERLAP":
            pass # 对duration的overlap事件放宽为和simultaneous一样处理
        else:
            inferred_time = None
            print("Warning: unknown relation type %s"%link_type)

        if duration and inferred_time:
            inferred_time # 在向后路径上，duration时长不能推断开始时间
        else:
            return None

        

    for path in out_paths:
        time_node = path[0]
        time_attrs = G.nodes[time_node]
        inferred_time = ['at', time_attrs['val'], time_attrs['text'], path]
        
        for i in range(1, len(path), 1):  
            node = path[i]
            node_type = G.nodes[node].get("type", "").upper()
            link_type = G.get_edge_data(path[i-1], path[i], {}).get("type", "").upper()
            if not node_type or not link_type:
                inferred_time = None
                break
            if node_type == 'DURATION':
                inferred_time = process_duration_node(node, inferred_time, link_type)
            else:
                if link_type == "BEFORE":
                    if inferred_time[0] != 'before':
                        inferred_time[0] = 'after' 
                    else:
                        inferred_time = None
                elif link_type == "BEFORE_OVERLAP":
                    if inferred_time[0] != 'before':
                        inferred_time[0] = 'after' 
                    else:
                        inferred_time = None
                elif link_type == "SIMULTANEOUS":
                    pass # unchanged
                elif link_type == "BEGINS":
                    pass # unchanged
                elif link_type == "ENDED_BY":
                    if inferred_time[0] != 'before':
                        inferred_time[0] = 'after' 
                    else:
                        inferred_time = None
                elif link_type == "COVER":
                    if inferred_time[0] != 'before':
                        inferred_time[0] = 'after' 
                    else:
                        inferred_time = None
                elif link_type == "OVERLAP":
                    inferred_time = None
                    # if inferred_time[0] != 'at': # 如果和一个具体的日期相连，overlap视为simultaneous
                    #     inferred_time = None
                else:
                    inferred_time = None
                    print("Warning: unknown relation type %s"%link_type)
            if not inferred_time:
                break  
        
        if inferred_time:
            inferred_times.append(inferred_time)

    # print(inferred_times)
    # start_time = intersect_time_intervals(inferred_times)
    return inferred_times  

def get_all_event_start_times(G):
    """
    处理所有类型为 ['PROBLEM', 'TEST', 'TREATMENT'] 的节点，
    计算它们的 start_time 并按 start_time 排序返回。
    """
    medical_types = {"PROBLEM", "TEST", "TREATMENT"}
    nodes = [n for n, attr in G.nodes(data=True) if attr.get("type").upper() in medical_types]
    results = []
    incomplete = False

    # 先获取入院时间
    admission_node = None
    for node, data in G.nodes(data=True):
        if data.get("text").lower() == "admission":
            admission_node = node
            break  # 找到后立即退出循环
    if admission_node:
        in_paths, out_paths = get_paths_to_time_nodes(G, admission_node,
                                                       time_node_types=['date'], 
                                                       filter_overlap=True, 
                                                       filter_mult_time_paths=True, 
                                                       filter_consecutive_durations=True)
        inferred_start_times_before = infer_head_start_time(G, out_paths)
        inferred_start_times_after = infer_tail_start_time(G, in_paths)
        admission_start_time = intersect_time_intervals(inferred_start_times_before + inferred_start_times_after)
    else:
        admission_start_time = None
    # print("Admission", admission_start_time)
    
    for node in nodes:
        in_paths, out_paths = get_paths_to_time_nodes(G, node, 
                                                       filter_overlap=True, 
                                                       filter_mult_time_paths=True, 
                                                       filter_consecutive_durations=True)
        
        inferred_start_times_before = infer_head_start_time(G, out_paths)
        inferred_start_times_after = infer_tail_start_time(G, in_paths)

        start_time = intersect_time_intervals(inferred_start_times_before + inferred_start_times_after)

        if start_time:
            in_hospital_course = G.nodes[node].get("hospital_course", None)
            if in_hospital_course and admission_start_time:
                if start_time[0] == datetime.min and start_time[1] > admission_start_time[0]:
                    start_time = (admission_start_time[0], start_time[1], admission_start_time[0].strftime("%Y-%m-%d") +" "+ start_time[2].replace('BEFORE', 'TO'), start_time[3])

            results.append((node, start_time))
        else:
            incomplete = True
            print("Can't infer the starttime, check the annotation please: %s, %s, %s"%(node, G.nodes[node].get("type","UNKNOWN"), G.nodes[node].get("text","UNKNOWN")))
    
    # 按 start_time 排序
    # print(results)
    results.sort(key=lambda x: (x[1][1] if x[1] else None, x[1][0] if x[1] else None))
    
    return results, incomplete

def get_print_paths(G, paths, abbr_path=True):
    """
    获取路径字符串
    """
    output_paths = []
    for path in paths:
        formatted_path = []
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            n1_info = G.nodes[n1]
            n2_info = G.nodes[n2]
            relation = G.get_edge_data(n1, n2).get("type", "UNKNOWN")
            
            if abbr_path:
                formatted_path.append(f"{n1}")
            else:
                formatted_path.append(f"({n1}: {n1_info.get('text', '')})")
            
            formatted_path.append(f" -> [{relation}] -> ")
        
        last_node = G.nodes[path[-1]]
        if abbr_path:
            formatted_path.append(f"{path[-1]}")
        else:
            formatted_path.append(f"({path[-1]}: {last_node.get('text', '')})")
        
        output_paths.append("".join(formatted_path))
    return output_paths

def format_time_range(start, end):
    def format_time_stamp(dt):
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            return dt.strftime('%Y-%m-%d')
        elif dt.second == 0:
            return dt.strftime('%Y-%m-%d %H:%M')
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')

    if end.hour == 23 and end.minute == 59 and end.second == 59:
        end = datetime(end.year, end.month, end.day)
    
    if start == datetime.min:
        return "BEFORE %s" % format_time_stamp(end)
    elif end == datetime.max:
        return "AFTER %s" % format_time_stamp(start)
    else:
        return "%s TO %s" % (format_time_stamp(start), format_time_stamp(end))

def merge_overlapping_nodes(G):
    
    # 先找出所有 "overlap" 关系的连通分量
    overlap_graph = nx.Graph()
    for u, v, attrs in G.edges(data=True):
        if attrs.get("type").lower() in ["overlap", "simultaneous"]:  # 只保留 overlap 关系
            overlap_graph.add_edge(u, v)

    # 计算所有需要合并的节点组
    merged_nodes = {}
    merged_attrs = {}
    for comp in nx.connected_components(overlap_graph):
        merged_id = ",".join(map(str, sorted(comp)))  # 生成合并后的新 id
        merged_type = "+".join(sorted(set(G.nodes[n].get("type", "") for n in comp if "type" in G.nodes[n])))
        merged_text = "+".join(sorted(set(G.nodes[n].get("text", "") for n in comp if "text" in G.nodes[n])))
        
        merged_attrs[merged_id] = {"type": merged_type, "text": merged_text}
        for node in comp:
            merged_nodes[node] = merged_id  # 记录合并映射

            
    # 构建新的有向图
    newG = nx.DiGraph()
    for node in G.nodes():
        new_node = merged_nodes.get(node, node)  # 如果在 merge 中，就替换成新 id
        if new_node not in newG:
            newG.add_node(new_node, **merged_attrs.get(new_node, G.nodes[node]))

    # 重新添加边，保持所有原有的边属性
    for u, v, attrs in G.edges(data=True):
        merged_u = merged_nodes.get(u, u)  # 如果 u 被合并，就用新 id
        merged_v = merged_nodes.get(v, v)  # 如果 v 被合并，就用新 id

        if merged_u != merged_v or attrs.get("type").lower() not in ["overlap", "simultaneous"]:  # 避免 self-loop
            newG.add_edge(merged_u, merged_v, **attrs)

    return newG 

def get_subgraph_of_interval(G, nodes):
    subgraph = G.subgraph(nodes).copy() 
    
    # 合并overlap
    subgraph = merge_overlapping_nodes(subgraph)
    # 找出孤立节点
    isolated_nodes = list(nx.isolates(subgraph))
    # 移除孤立节点
    subgraph.remove_nodes_from(isolated_nodes)
    return subgraph

def find_distinct_paths(G):
    all_paths = []
    for start in G.nodes:
        for end in G.nodes:
            if start != end:
                for path in nx.all_simple_paths(G, source=start, target=end):
                    all_paths.append(path)
    
    # 去除被完全包含的路径
    filtered_paths = []
    for path in all_paths:
        if not any(set(path) < set(other_path) for other_path in all_paths if path != other_path):
            filtered_paths.append(path)
    
    return filtered_paths
 
def process_start_time():
    """
    手动修改:
    Train:
    121.xml,T36的val错误:1993-0907T14:30 -> 1993-09-07T14:30
    526.xml, T25的val错误: "" -> 2001-09-27
    751.xml, T7: "" -> 1996-12-31
    596.xml, T4: P5S -> PT5S
    541.xml, T2, T12: DATE -> DURATION
    602.xml, T8: DATE -> DURATION
    757.xml, T21: DATE -> DURATION
    218.xml, T18: DATE -> DURATION
    472.xml, T20: 2017-07-012 -> 2017-07-12
    107.xml, T3, T16: "" -> 2000-02-01; T11: DATE -> DURATION
    191.xml, T7, "" -> 1997-12-31
    471.xml, T2, type="DATE" val="1993-05-31" mod="APPROX"
    267.xml, T0, 2013-03-25T 24:00 -> 2013-03-26T00:00
    517.xml, T30, T16 P8H -> PT8H; T37 PT24H
    756.xml, T13, DURATION -> FREQUENCY
    577.xml, T10, rpt1h -> pt1h
    301.xml, T3, T4, p24h -> pt24h
    23.xml, T2, T5, p8h -> pt8h, p12h -> pt12h
    68.xml, T17, DATE -> FREQUENCY
    256.xml, T5, DATE -> DURATION
    408.xml, T25, P12H -> PT12H
    Ground truth: None
    """
    src_foler = "data/i2b2/timeline_training"
    data = data_loader(src_foler)
    incomplete_files = []
    for fid, filename in enumerate(data):
        print(str(fid), ":", filename)
        # if os.path.exists(os.path.join(src_foler, "%s.starttime.json"%filename)):
        #     continue
        G, text = build_section_graph(filename, data)
        G = preprocess_time_overlap_edges(G)
        results, incomplete = get_all_event_start_times(G)
        if incomplete:
            incomplete_files.append(filename)
        output = []
        for node, (start, end, time_in_text, paths) in results:
            node_data = {}
            node_type = G.nodes[node].get("type", "UNKNOWN")
            node_text = G.nodes[node].get("text", "UNKNOWN")
            formatted_time_range = format_time_range(start, end)
            formatted_paths = get_print_paths(G, paths, abbr_path=False)
            node_data["node_id"] = node
            node_data["type"] = node_type
            node_data["text"] = node_text
            node_data["start_range"] = [str(start),str(end)]
            node_data["formatted_time_range"] = formatted_time_range
            node_data["context_time_range"] = time_in_text
            node_data["paths"] = formatted_paths
            output.append(node_data)
        with open(os.path.join(src_foler, "%s.starttime.json"%filename), "w") as f:
            json.dump(output, f, indent=4)

    with open("data/i2b2/incomplete_start_time_train.txt", "w") as f:
        f.write("\n".join(incomplete_files))

def process_interval_paths():
    src_foler = "data/i2b2/timeline_test"
    data = data_loader(src_foler)
    for fid, filename in enumerate(data):
        print(str(fid), ":", filename)
 
        G, text = build_section_graph(filename, data)
        G = preprocess_time_overlap_edges(G)
        results, incomplete = get_all_event_start_times(G)
        interval_to_nodes = defaultdict(list)
        output = []
        for node_id, (start, end, time_in_text, paths) in results:
            time_range = format_time_range(start, end)
            interval_to_nodes[time_range].append(node_id)
        
        for time_range, nodes in interval_to_nodes.items():
            subgraph = get_subgraph_of_interval(G, nodes)
            paths = find_distinct_paths(subgraph)
            formatted_paths = get_print_paths(subgraph, paths, abbr_path=False)
            output.append({"time_range": time_range, "nodes": nodes, "subgraph_paths": formatted_paths})

        with open(os.path.join(src_foler, "%s.interval_paths.json"%filename), "w") as f:
            json.dump(output, f, indent=4)

if __name__ == '__main__':
    # process_start_time()
    process_interval_paths()