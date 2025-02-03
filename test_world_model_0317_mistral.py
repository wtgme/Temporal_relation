import os

import json
import pandas as pd
import time
from tqdm import tqdm
import re
import traceback

from mistral import Mistral

llm = Mistral()


system_prompt_single = '''I will provide you with a quantity-related question, and you need to help me extract relevant evidence from the given information to assist me in answering the question. Specifically, you need to help me list the evidence related to the question and store it in Python classes. 

Remember: Your task is to extract evidence from the information; you don't need to answer questions.

Tips:
- You need to store all specific evidence related to the question in the EvidenceSingle class. EvidenceSingle should be initialized with the name, quantity, and time of the evidence.
- The single_time is represented using the Time class.
- If there are multiple related evidence, create multiple EvidenceSingle instances.

class EvidenceSingle(object):
    def __init__(self, name, number, single_time):
    # name: The name of this evidence
    # number: The quantity corresponding to the current evidence, not the total number (even if you can infer the total).
    # single_time: The corresponding time of the current evidence.

class Time(object):
    def __init__(self, year, month, day):

EXAMPLE :
Information:
Season 4 of Prison Break primarily aired on September 1, 2008, following a similar format to the previous seasons. It consisted of 24 episodes and concluded the original run of the series until its revival with Season 5 several years later.

"Prison Break: Resurrection" or season 5 aired on April 4, 2017, and received mixed reviews from critics and audiences alike. However, it was praised for its nostalgic appeal and the return of beloved characters.

Question: How many seasons does Prison Break have?

Extract Evidence:
```python
evidence1 = EvidenceSingle('Season 4', 1, Time(year=2008, month=9, day=1)) # Season 4 of Prison Break primarily aired on September 1, 2008. 
evidence2 = EvidenceSingle('Season 5', 1, Time(year=2017, month=4, day=4)) # "Prison Break: Resurrection" or season 5 aired on April 4, 2017.
```'''

system_prompt_summary = '''I will provide you with a quantity-related question, and you need to help me extract relevant evidence from the given information to assist me in answering the question. Specifically, you need to help me list the evidence related to the question and store it in Python classes. 

Remember: Your task is to extract evidence from the information; you don't need to answer questions.

Tips:
- You need to summarize the quantity of evidence related to the question, it should be stored in the EvidenceSummary class. The EvidenceSummary needs to be initialized with the quantity and the time point of the summary.
- If there is no explicit summary time in the information, use the time of the latest evidence as the summary time.
- The summary_time is represented using the Time class.
- Only extract one EvidenceSummary.


class EvidenceSummary(object):
    def __init__(self, number, sumarry_time):

class Time(object):
    def __init__(self, year, month, day):

EXAMPLE:
Information:
As of 2021 September, the Lego Movie, a 2014 animated adventure comedy film written and directed by Phil Lord and Christopher Miller (both pictured), won 39 awards from 76 nominations.

Question: How many awards did "The Lego Movie win?

Extract Evidence:
```python
summary = EvidenceSummary(39, Time(year=2021, month=9, day=None)) # The LEGO Movie won a total of 39 awards as of September 2021.
```'''
user_prompt = 'Information:\n#INFORMATION#\n\nQuestion:\n#QUESTION#\n\nExtract Evidence:\nshort answer'


def forward(system_message, user_message, name):
    global llm_output
    if name in llm_output.keys():
        return llm_output[name].replace('</s>', '')
    response = llm.forward([{'role': 'user', 'content': system_message + '\n' + user_message}])
    llm_output[name] = response
    return response.replace('</s>', '')


def replace_prompt(prompt, old, new):
    for o, n in zip(old, new):
        prompt = prompt.replace(o, n)
    return prompt

class EvidenceSummary(object):
    def __init__(self, number, valid_time):
        self.number = int(number) if number is not None and valid_time is not None else 0
        self.time = valid_time if valid_time is not None else Time(2021, 9, None)
        if valid_time.year == 0 and valid_time.month == 12 and valid_time.day == 31:
            self.time = Time(2021, 9, None)

class EvidenceSingle(object):
    def __init__(self, name, number, single_time):
        self.name = name
        self.number = int(number) if number is not None and single_time is not None else 0
        self.time = single_time if single_time is not None else Time(None, None, None)

class Time(object):
    def __init__(self, year=None, month=None, day=None):
        self.year = year if year is not None else 0
        self.month = month if month is not None else 12
        self.day = day if day is not None else 31

    def compare(self, time2):
        if self.year > time2.year:
            return 1
        elif self.year == time2.year:
            if self.month > time2.month:
                return 1
            elif self.month == time2.month:
                if self.day > time2.day:
                    return 1
                elif self.day == time2.day:
                    return 0
                else:
                    return -1
            else:
                return -1
        else:
            return -1
        
    def compare_coarse(self, time2):
        if self.year > time2.year:
            return 1
        elif self.year == time2.year:
            if self.month > time2.month:
                return 1
            elif self.month == time2.month:
                if self.day > time2.day:
                    return 1
                elif self.day == time2.day:
                    return 0
                else:
                    return -1
            else:
                return -1
        else:
            return -1

class Wolrd(object):
    def __init__(self, quantity_list):
        self.quantity_list = quantity_list
    
    def query(self, query_time):
        candidate_list = []
        for quantity in self.quantity_list:
           if query_time.compare(quantity.time) == 1:
               candidate_list.append(quantity)
        
        final_quantity = 0

        final_list = [1] * len(candidate_list)
        for i, candidate in enumerate(candidate_list):
            for j, check in enumerate(candidate_list):
                if i != j:
                    if isinstance(check, EvidenceSummary) and candidate.time.compare(check.time) <= 0:
                        final_list[i] = 0
                        break
                    if final_list[i] and isinstance(check, EvidenceSingle) and isinstance(candidate, EvidenceSingle) and candidate.time.compare(check.time) == 0 and check.name == candidate.name:
                        final_list[j] = 0
                        break
            
            if final_list[i]:
                final_quantity += candidate.number
        return final_quantity


def split_code(response, name):
    try:
        result = re.search('```python(.*?)```', response, re.DOTALL)
        code = result.group(1)
        return code.strip()
    except:
        try:
            response += '```'
            result = re.search('```python(.*?)```', response, re.DOTALL)
            code = result.group(1)
            return code.strip()
        except:
            print('===============')
            print(name)
            print('cannot split code {}'.format(response[:-3]))
            print('===============')
            return '\n'.join([x for x in response[:-3].strip().split('\n') if '=' in x])

def extract_quantity(question, information_dict, top_name):
    quantity_list = []
    for key, information in information_dict.items():
        if key == 'internal evidence':
            name = 'internal evidence'
            system_prompt = system_prompt_summary
        else:   
            name = top_name
            system_prompt = system_prompt_single
        current_user = replace_prompt(user_prompt, ['#INFORMATION#', '#QUESTION#'], [information, question])
        response = forward(system_prompt, current_user, name)
        code = split_code(response, name)
        # print(code)
        try:
            g = {
                'EvidenceSummary': EvidenceSummary,
                'EvidenceSingle': EvidenceSingle,
                'Time': Time
            }
            l = {}
            code_compile = compile(code, '', 'exec')
            exec(code_compile, g, l)
            for quantity in l.values():
                if isinstance(quantity, EvidenceSummary) or isinstance(quantity, EvidenceSingle):
                    quantity_list.append(quantity) 
        except:
            print(cur_id)
            print(top_name)
            print(traceback.format_exc())

    return quantity_list     
    # return [Quantity(8, Time(None, None, None), Time(2021, None, None)), Quantity(1, Time(2022, None, None), Time(2022, None, None))]


def main(question, information_list, query_time, top_name):
    quantity_list = extract_quantity(question, information_list, top_name)
    world_model = Wolrd(quantity_list)
    final_quantity = world_model.query(query_time)
    # try:
    #     final_quantity = world_model.query(query_time)
    # except:
    #     global llm_output
    #     print(llm_output)

    return final_quantity

model_str = 'mistral'

data = pd.read_csv('dataset/testset_ori_v5_{}.csv'.format(model_str))

print(data.columns)
length = len(data)
question_str = 'question_date'
version = 'world_model_0317_{}'.format(model_str)

correct = [0, 0, 0]
acc = [0, 0, 0]
data_iter = tqdm(data.iterrows())
count = [0, 0, 0]
for index, row in data_iter:
    # if row['id'] != 38:
    #     continue
    cur_id = row['id']
    target = row['question']
    llm_output_path = 'result/{}/{}.json'.format(version, row['id'])
    if os.path.exists(llm_output_path):
        with open(llm_output_path, 'r') as f:
            llm_output = json.load(f)
    else:
        llm_output = {}

    i1 = row['information1'].strip()
    i2 = row['information2'].strip()
    top_name = '{}_{}_{}'.format(row['type'], row['old_length'], row['new_length'])
    # if top_name == 'type2_2_2':
    #     pass
    prediction = main(target, {'internal evidence': i1, 'external evidence': i2}, Time(row['last_year'], 12, 31), top_name)

    answer = row['answer'] if row['type'] != 'type3' else row['old_answer']
    llm_output['answer_{}'.format(top_name)] = answer
    llm_output['prediction_{}'.format(top_name)] = prediction

    if prediction == answer:
        check = 1
    else:
        check = 0

    if row['type'] == 'type1':
        # print(row['answer'], prediction)
        count[0] += 1
        correct[0] += check
        acc[0] = correct[0] / count[0]
    elif row['type'] == 'type2':
        count[1] += 1
        correct[1] += check  
        acc[1] = correct[1] / count[1]
    else:
        count[2] += 1
        correct[2] += check  
        acc[2] = correct[2] / count[2]

    data_iter.set_description('with accuracy: type1: {:.2f}({}/{})%, type2:{:.2f}({}/{})%, type3:{:.2f}({}/{})%'.format(acc[0]*100, correct[0], count[0], acc[1]*100, correct[1], count[1], acc[2]*100, correct[2], count[2]))
    
    if llm_output:
        with open(llm_output_path, 'w') as  f:
            json.dump(llm_output, f)
    

