
from asyncore import read
import re
import json
import os
from typing import Dict, List
import itertools
import csv

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    MBart50TokenizerFast,
)
import jieba

import sys
sys.path.append("/remote-home/xtzhang/playground/tmp/codedetect/")


from utils.io import read_csv


def wash_n(data):
    import re
    return [re.sub("\n", "", i) for i in data ]

def wash_id(data):
    
    return [ " ".join(i.split()[1:]) for i in data ]


def load_json(data_path):
    
    file = open(data_path, 'r', encoding='utf-8')

    data = []

    for line in file.readlines():
        tmp = json.loads(line)
        data.extend( tmp["original_string"].split("\n") )   

    return data

def read_code(head="/remote-home/xtzhang/playground/tmp/codedetect/data/eng/code"):
    #dirs = os.listdir( nature_dir )
    types = [ "train", "dev", "test" ]
    program_names = [ "python", "go", "php", "java", "javascript", ]
    
    code_data = { i:[] for i in types }
    trunc_table = {"train":12000, "dev":1500, "test":1500}

    for program_name in program_names:
        for i in types:
            if i == "dev":
                name = "valid"
            else:
                name = i
            path = "_".join([program_name, name, "0"]) + '.jsonl'
            tmp = load_json( head + "/" +path )
            #print(len(tmp))
            
            code_data[i].extend(tmp[:trunc_table[i]])
    
    dirs = os.listdir(head + "/c")
    all_c = []
    for d in dirs:
        tmp = read_csv(head + "/c/" + d)
        if tmp:
            all_c.extend(tmp)

    import random
    random.shuffle(all_c)
    
    code_data["train"].extend(all_c[:12000])
    code_data["dev"].extend(all_c[12000:13500]) 
    code_data["test"].extend(all_c[13500:])
    
    return code_data

def read_nature(head="/remote-home/xtzhang/playground/tmp/codedetect/data/eng/nature"):
    #dirs = os.listdir( nature_dir )
    types = [ "train", "dev", "test" ]
    dataset_names = [ "QNLI", "QQP", "STS-B", "WNLI", ]
    
    def get_CoLA_data(i):
        data=[]
        with open(head + "/CoLA/" + i + ".tsv") as fd:
            rd=csv.reader(fd, delimiter="\t", quotechar='"')
            for line in rd:
                data.append(line[-1])         
        return data
    
    def get_pairs_data_template(name, i):
        data=[]
        with open(head + "/" + name + "/" + i + ".tsv") as fd:
            rd=csv.reader(fd, delimiter="\t", quotechar='"')
            for i, line in enumerate(rd):
                if i == 0:
                    continue
                if i != "test":
                    data.append(line[-2])
                    data.append(line[-3])
                else:
                    data.append(line[-1])
                    data.append(line[-2])      
        return data
    
    nature_data = { i:[] for i in types }
    trunc_table = {"train":12000, "dev":1500, "test":1500}

    for name in dataset_names:
        for i in types:
            tmp = get_pairs_data_template(name, i)#read_csv( head + "/" +path )
            nature_data[i].extend(tmp[:trunc_table[i]])
    
    for i in types:
        tmp = get_CoLA_data(i)
        nature_data[i].extend(tmp[:trunc_table[i]])
    
    return nature_data
    
from fastNLP import cache_results
@cache_results(_cache_fp='cache/codedetect', _refresh=True)
def load_codedetect():
    code_data = read_code()
    nature_data = read_nature()

    tokenizer_model_name_path="bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name_path)

    train_source = wash_n(code_data["train"] + nature_data["train"] )
    dev_source = wash_n(code_data["dev"] + nature_data["dev"] )
    test_source = wash_n(code_data["test"] + nature_data["test"] )

    train_source_tok = tokenizer.batch_encode_plus(train_source, return_token_type_ids=False)
    dev_source_tok = tokenizer.batch_encode_plus(dev_source, return_token_type_ids=False) 
    test_source_tok = tokenizer.batch_encode_plus(test_source, return_token_type_ids=False)

    train_source_tok["labels"] = [[1]] * len(code_data["train"]) + [[0]] * len(nature_data["train"])
    dev_source_tok["labels"] = [[1]] * len(code_data["dev"]) + [[0]] * len(nature_data["dev"]) 
    test_source_tok["labels"] =  [[1]] * len(code_data["test"]) + [[0]] * len(nature_data["test"]) 

    def transpose(inputs):
        features = []
        for i in tqdm(range(len(inputs["input_ids"]))):
            features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
        return features 

    return transpose(train_source_tok), transpose(dev_source_tok), transpose(test_source_tok)

class myBatchEncoding():
    """
    Only Using when Debuging FaskNLP Dataset -> HuggingFace Transformers BatchEncodings
    """
    def __init__(self, data:Dict, encodings:List[Dict]):
        self.data = data
        self.encodings = encodings

    def __getitem__(self, key):
        #
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, int):
            return self.encodings[key]
        else:
            raise KeyError("Wrong Key !")

    def __len__(self):
        return len(self.encodings)


if __name__ == "__main__":
    #load_sighan()
    #a, b, c = load_lattice_sighan()
    #a,b,c = load_abs_pos_sighan_plus(path_head=".")
    """
    Check length for csc task
    """
    #a,b,c = load_sighan_enchanted()
    a, b, c = load_codedetect()
    
    for index, i in enumerate(a):
        if ( len(i["input_ids"]) != len(i["attention_mask"]) ):
            print(index)
            print(len(i['input_ids']))
            print(len(i['labels']))
            print(len(i['attention_mask']))
            print(i['input_ids'])
            print(i["labels"])
            print("[Data] something goes wrong!")
            exit()
    else:
        print("[Data] Seems working well !")
