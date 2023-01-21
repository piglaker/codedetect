
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
)

class CNN(nn.Module):
    def __init__(self, input_size=21168, embedding_dim=768, kernel_wins=[3,4,5], num_class=2, embedding=None):
        super(CNN, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(21168, embedding_dim, max_norm=True, padding_idx=-100)
        else:
            self.embedding = embedding
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, embedding_dim, size) for size in kernel_wins])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(kernel_wins)*embedding_dim, num_class)

    def forward(self, input_ids, attention_mask, labels=None):
        embeded_x = self.embedding(input_ids)

        embeded_x = embeded_x.transpose(1, 2)
        con_x = [conv(embeded_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)

        fc_x = self.dropout(fc_x)

        fc_x = fc_x.squeeze(-1)

        logits = self.fc(fc_x)
        
        loss_function = torch.nn.CrossEntropyLoss()
        #print(logits.shape, labels.shape) 
        if labels:
            loss = loss_function( logits.float(), labels.long().squeeze(1) )
        else:
            loss = None 
        
        return loss, logits

class FoolDataCollatorForSeq2Seq:
    label_pad_token_id = -100
    def __call__(self, features):
        """
        """
        from copy import deepcopy

        f_copy = deepcopy(features)

        shared_max_length = max([ len(i['input_ids']) for i in f_copy])

        def simple_pad(f_copy, key):
            max_length = shared_max_length
            f_key = [ f[key] for f in f_copy ]
            if f_key is not None:
                padding_side = "right"
                if key == "attention_mask":
                    label_pad_token_id = 0
                elif key == "input_ids":
                    label_pad_token_id = 0
                elif key.find("labels") != -1:
                    label_pad_token_id= -100
                    
                else:
                    label_pad_token_id = self.label_pad_token_id 

                for f in f_copy: 
                    remainder = [label_pad_token_id] * (max_length - len(f[key]))
                    
                    f[key] = (f[key] + remainder if padding_side == "right" else remainder + f[key])
            
            return f_copy

        for key in f_copy[0].keys():
            if key != "labels":
                f_copy = simple_pad(f_copy, key)

        new = {}

        for key in f_copy[0].keys():  
            new[key] = []
        
        for feature in f_copy:
            for key in feature.keys():
                new[key].append(feature[key])

        for key in new.keys():
            try:
                new[key] = torch.tensor(new[key]) 
            except:
                print("[Lib] DataCollatorForSeq2Seq Error : Mismatch length ! ")
                print("[Lib] key : ", key)
                print([ len(i) for i in new[key] ])
                print("[Lib] WHY WE CATCH HERE: https://github.com/pytorch/pytorch/issues/67538")
                exit()

        return new

class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def transpose(inputs):
    features = []
    for i in range(len(inputs["input_ids"])):
        features.append({key:inputs[key][i][:128] for key in inputs.keys()}) #we fix here (truncation 
    return features 

def code_detect(test_case):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", force_download=True)

    
    embeddings = AutoModel.from_pretrained("bert-base-uncased").embeddings
    
    model = CNN(input_size=21168, embedding_dim=768, kernel_wins=[3,4,5], num_class=2, embedding=embeddings)
    model.load_state_dict(torch.load("/remote-home/xtzhang/playground/tmp/codedetect/tmp/bert/textcnn_eval_epoch10_bs512_seed3471/pytorch_model.bin"))
    model.eval() 
    
    sentences = test_case.split("\n")

    preprocess = FoolDataCollatorForSeq2Seq()

    tmp = tokenizer.batch_encode_plus(sentences, return_token_type_ids=False) 
 
    _, logits = model(**preprocess(mydataset(transpose(tmp))))
    
    preds = torch.argmax(torch.nn.functional.softmax(logits, 1), -1)
    
    return preds

test_case = """specify the language using the iso 639 - 1 2 - letter language code. for
A woman is cutting an onion.
print("Happy New Year !")
User
Show me the Python code
OpenChat
Sure thing! Here's some sample code written in Python:
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[o]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        sorted left = quick sort(left)
        sorted_right = quick_sort(right)
        return sorted_left + [pivot] + sorted_right
8.73s

"""


res = code_detect(test_case)

print(res)

