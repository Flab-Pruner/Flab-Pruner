import math
import os
import sys

import pandas as pd
import torch

from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from pre_train.custom_datasets import Dataset_Pretrain
from badam import BlockOptimizer
from optims.Adam_mini import Adam_mini
from optims.lomo import Lomo
from optims.adalomo import AdaLomo

import torch.nn as nn

class LLAMA():

    def __init__(self, model, tokenizer, source_len=512, optim=''):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.source_len = source_len
        # 初始化LLM模型

        tokenizer.pad_token = tokenizer.eos_token
        self.model, self.tokenizer = model, tokenizer

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        self.model.to(self.device)
        self.optim = optim

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs,
            do_eval, eval_filename, eval_batch_size, output_dir):

        train_data = Dataset_Pretrain(train_filename, tokenizer=self.tokenizer, source_len=self.source_len)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                    batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        if self.optim == 'badam':
            optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
            optimizer = BlockOptimizer(
                base_optimizer=optimizer, # can be any torch.Optimizer
                named_parameters_list=list(self.model.named_parameters()), 
                switch_block_every=100, # switch to the new block every 50 updates, the $K$ Adam steps in paper. It can be set adaptively by $K = n/(BD)$, where $n$ is the number of training data points, $B$ is the batch size, and $D$ is the number of blocks in BAdam; see "Hyperparameter Suggestion" section for a detailed explaination about setting this hyperparameter. 
                switch_mode="random", # update order of blocks, one can choose "random" (random reshuffling update order), "ascending" (update from input layer to output layer), or "descending" (update from output layer to input layer). The default is "random".
                verbose=2, # information level, will print trainable parameters when setting to 2
                # include_embedding=True,
                # include_lm_head=True,
            )
        elif self.optim == 'adam_mini':
            optimizer = Adam_mini(self.model, 
                                lr=learning_rate, 
                                n_feature=4096,
                                n_head=32,
                                n_kv_head=4)
        elif self.optim == 'lomo':
            optimizer = Lomo(self.model, 
                            lr=learning_rate)
        elif self.optim == 'adalomo':
            optimizer = AdaLomo(self.model, 
                            lr=learning_rate)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_steps, tr_loss = 0, 0
            self.model.train()
            for step, (input_ids, token_labels) in enumerate(bar):
                input_ids = input_ids.to(self.device)
                labels = token_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                tr_loss += loss.item()
                nb_tr_steps += 1

                if self.optim == 'badam' or self.optim == 'adam_mini':
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.fused_backward(loss=loss, lr=scheduler.get_last_lr()[0])
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if do_eval:
                # Eval model with dev dataset
                eval_data = Dataset_Pretrain(eval_filename, tokenizer=self.tokenizer, source_len=self.source_len)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("***** Running evaluation  *****")
                print("  Num examples = %d", eval_data.__len__())
                print("  Batch size = %d", eval_batch_size)
                print("  Num epoch = %d", cur_epoch)
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for (input_ids, token_labels) in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    input_ids = input_ids.to(self.device)
                    labels = token_labels.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                    eval_loss += loss.mean().item()
                    batch_num += 1
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_loss': round(eval_loss, 5),
                        'global_step': global_step + 1,
                        'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.model.save_pretrained(output_dir)
                print('best eval loss: ', str(eval_loss))
                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
    def get_model(self):
        return self.model