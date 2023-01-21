import os
import re
import time
from dataclasses import dataclass, field
from timeit import repeat
from typing import Optional,Dict,Union,Any,Tuple,List

import fitlog
import nltk
import numpy as np
import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers.trainer_pt_utils import nested_detach
from transformers import BertForMaskedLM
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from data.DatasetLoadingHelper import (
    load_codedetect
)

def ddp_exec(command):
    """
    """
    if os.environ["LOCAL_RANK"] != '0':
        return
    else:
        exec(command) 

def ddp_print(*something):
    """
    out of time
    """
    if os.environ["LOCAL_RANK"] != '0':
        return
    else:
        for thing in something:
            print(thing)

        return 

def fitlogging(training_args):
    for attr in dir(training_args):
        if not re.match("__.*__", attr) and isinstance(getattr(training_args, attr), (int, str, bool, float)):
            fitlog.add_hyper(value=getattr(training_args, attr), name=attr)
    return

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    # hack for bug
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

    # extra args
    model_name: str=field(default="textcnn", metadata={"help":"which bert model "})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})
    num_gpus:int = field(default=4, metadata={"help":"num_gpus"})
    pretrained_name:str = field(default="roberta", metadata={"help":"pretrained_name"})
    log_path:str = field(default="Recent_train.log", metadata={"help":"log path or name"})

class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def argument_init(trainingarguments=Seq2SeqTrainingArguments):
    """ 
    """
    parser = HfArgumentParser(trainingarguments)

    training_args = parser.parse_args_into_dataclasses()[0]

    return training_args


def get_model(model_name="textcnn", pretrained_model_name_or_path="bert-base-uncased", training_args=None):
    """
    Just get model
    """
    model = None

    print("[Core] Hint: Loading Model " + "*"*5 + model_name + "*"*5)

    if model_name == "textcnn":
        from models.textcnn import CNN as ProtoModel
    
    embeddings = AutoModel.from_pretrained(pretrained_model_name_or_path).embeddings
    
    model = ProtoModel(input_size=21168, embedding_dim=768, kernel_wins=[3,4,5], num_class=2, embedding=embeddings)
    
    if not model:
        print("[Core]  Warning: wrong model name ! Check the core.py  ")
        exit()
    return model


def get_dataset():
    """
    """

    print("[Core] Loading Dataset !")
    exec("os.system('date')")

    
    train_data, eval_data, test_data = load_codedetect()
    
    train_dataset, eval_dataset, test_dataset = mydataset(train_data), mydataset(eval_data), mydataset(test_data)

    print("[Core] Loading Succeed !")
    exec("os.system('date')")

    return train_dataset, eval_dataset, test_dataset

def get_metrics():
    """
    """
    
    import numpy as np
    from datasets import load_metric

    def compute_metrics(eval_preds):
        """
        """
        Achilles = time.time()

        sources, preds, labels = eval_preds# (num, length) np.array
        #print(preds, labels)
        #exit()
        hit = 0

        for i in range(len(sources)):
            
            if preds[i] == labels[i]:
                hit +=1 

        acc = hit / len(sources)

        Turtle = time.time() - Achilles

        return {"Acc": float(acc),"Metric_time":Turtle}

    return compute_metrics


if __name__ == "__main__":
    print("[Core] Lets test !")

    training_args = argument_init(TrainingArguments)

    train_dataset, eval_dataset, test_dataset = get_dataset(training_args.dataset) 

    compute_metrics = get_metrics(None)

    print("[Core] Done")

