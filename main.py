# Copyright 2021 piglake
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import sys

class DDP_std_IO(io.StringIO):
    def write(self, txt):
        if os.environ["LOCAL_RANK"] != '0':
            pass
        else:
            sys.__stdout__.write(txt)

        #sys.__stdout__.write(txt)

class DDP_err_IO(io.StringIO):
    def write(self, txt):
        if os.environ["LOCAL_RANK"] != '0':
            pass
        else:
            sys.__stderr__.write(txt)

sys.stdout = DDP_std_IO()
sys.stderr = DDP_err_IO()

from dataclasses import dataclass, field

# import fitlog
import numpy as np

from transformers import (
    AutoTokenizer,
)

from transformers.trainer_utils import set_seed

#from transformers import trainer_utils, training_args

from core import (
    get_model,
    get_metrics, 
    argument_init, 
    get_dataset,
    MySeq2SeqTrainingArguments, 
)
from lib import MyTrainer, FoolDataCollatorForSeq2Seq, subTrainer 
#from models.bart.modeling_bart_v2 import BartForConditionalGeneration

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

#fitlog.set_log_dir("./fitlogs/")
#fitlog.add_hyper_in_file(__file__)

#sys.stdout = sys.__stdout__

import logging

logger = logging.getLogger(__name__)

def adapt_learning_rate(training_args):
    training_args.learning_rate = (training_args.num_gpus * training_args.per_device_train_batch_size / 128 )* 7e-5
    print("[Main] Adapted Learning_rate:", training_args.learning_rate)
    return training_args

class DDP_std_saver(io.StringIO):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.__stdout__
        dirs = "/".join(filename.split("/")[:-1])
        if os.environ["LOCAL_RANK"] == '0':
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            self.log = open(filename, "w+")
 
    def write(self, txt):
        if os.environ["LOCAL_RANK"] != '0':
            pass
        else:
            self.terminal.write(txt)
            self.log.write(txt)

class DDP_err_saver(io.StringIO):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.__stderr__
        #dirs = "/".join(filename.split("/")[:-1])
        #if not os.path.exists(dirs):
        #    os.makedirs(dirs)
        if os.environ["LOCAL_RANK"] != '0':
            pass
        else: 
            self.log = open(filename, "w+")
 
    def write(self, txt):
        if os.environ["LOCAL_RANK"] != '0':
            pass
        else:
            self.terminal.write(txt)
            self.log.write(txt)

def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)
    
    sys.stdout = DDP_std_saver(training_args.log_path)
    sys.stderr = DDP_err_saver("Recent_Error.log")
    
    #fitlogging(training_args)

    set_seed(training_args.seed)

    training_args = adapt_learning_rate(training_args)

    name_dict = { 
        "bert":"bert-base-uncased", \
    }

    name = name_dict[training_args.pretrained_name]

    print("[Main] Possible Backbone Pretrained Model Name_or_Path:" + name)

    # Tokenizer    
    tokenizer_model_name_path=name#"bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path,
        #force_download=True
    )

    # Dataset
    train_dataset, eval_dataset, test_dataset = get_dataset()

    # Model
    model = get_model(
        model_name= "textcnn" if training_args.model_name is None else training_args.model_name, 
        pretrained_model_name_or_path=tokenizer_model_name_path,
        training_args=training_args,
    ) #base

    # Metrics
    compute_metrics = get_metrics()

    # Data Collator

    data_collator = FoolDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
    )#my data collator  fix the length for bert.
    
    # Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,         
        train_dataset=train_dataset,    
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,      
        compute_metrics=compute_metrics,#hint:num_beams and max_length effect heavily on metric["F1_score"], so I modify train_seq2seq.py to value default prediction_step function
    )

    # fitlog.finish()

    # Train
    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation 
    # reference:https://github1s.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict",
        )
        print(predict_results)
        
        output_prediction_file = os.path.join("./", "predictions.txt")#training_args.output_dir, "predictions.txt")

        final_results = []
        acc = 0
        for i in range(len(test_dataset)):
            s = tokenizer.decode(test_dataset[i]["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            tmp = predict_results.predictions[i]#s + " " + str(test_dataset[i]["labels"]) + " " +predict_results[i]
            label = predict_results.label_ids[i]
            if tmp == label[0]:
                acc += 1
            #else:
            final_results.append( "\t".join([s, str(tmp), str(label[0])]) )
        
        print("Test Acc:", acc / len(test_dataset), )

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(final_results))

        #metrics here


    logger.info("*"*10 + "Curtain" + "*"*10)

    print("*"*10 + "over" + "*"*10)

    return

if __name__ == "__main__":
    run()
