from model import Transformer,ModelArgs
from gpt_core import tokenizer_merge
from tokenizer_merge import sp_config
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import json
from tqdm import tqdm
import os
import torch
import logging
import warnings
from grokadamw import GrokAdamW
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class dataset_config:
    max_seq_len : int = 512
    sp_config : sp_config = sp_config()
    train_dir : str = 'train_data/txt_data/非正式语料.txt'
    # train_dir = 'instruct.jsonl'

@dataclass
class train_config:
    train_dataset_config : dataset_config = dataset_config(
        max_seq_len = 512,
        sp_config = sp_config(
            vocab_size = 8192,
            max_seq_len = 512,
            model_prefix = 'spm_path/spm',
            sp_model_path = 'spm_path/spm.model',
            sp_train_dir ='train_data/txt_data/非正式语料.txt',
            sp_model_type = 'bpe',
            special_tokens = ['<|im_start|>', '<|im_end|>', '<|pad|>', '<|unk|>']
        ),
        train_dir ='train_data/txt_data/非正式语料.txt'
    )
    valid_dataset_config : dataset_config = dataset_config(
        max_seq_len = 512,
        sp_config = sp_config(
            vocab_size = 8192,
            max_seq_len = 512,
            model_prefix = 'spm_path/spm',
            sp_model_path = 'spm_path/spm.model',
            sp_train_dir ='train_data/txt_data/非正式语料.txt',
            sp_model_type = 'bpe',
            special_tokens = ['<|im_start|>', '<|im_end|>', '<|pad|>', '<|unk|>']
        ),
        train_dir ='train_data/instruct.jsonl'
    )
    model_config : ModelArgs = ModelArgs(
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=16384,
        multiple_of = 128,
        norm_eps = 1e-05,
        rope_theta = 10000.0,
        max_batch_size= 32,
        max_seq_len = 512
    )
    device : str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size : int = 4
    lr : float = 1e-3
    epoch : int = 32
    is_continue_training : bool = False
    train_dir_savesuffix : str = 'train_records'
    loss_update_interval : int = 100
    model_save_interval : int = 1000
    gradient_accumulation_steps : int = 1
    weight_decay : float = 0.01
    lr_scheduler : str = 'linear'
    model_path : str = 'model.pth'
    grokking_signal = 0.0


class process_dataset(Dataset):
    def __init__(self,dataset_config):
        self.encoded_text = []
        self.block_size = dataset_config.max_seq_len
        self.tokenizer = tokenizer_merge.sp_tokenizer(dataset_config.sp_config)
        # 检查文件是否存在
        if dataset_config.train_dir.endswith('.jsonl'):
            with open(dataset_config.train_dir, 'r', encoding='utf-8') as file:
                dialogues = [json.loads(line) for line in file]
                for dialogue in dialogues[:10]:
                    instruction = "<|im_start|>system: 你是一个助手<|im_end|>"
                    input = "<|im_start|>user: " + dialogue['instruction'] + "<|im_end|>"
                    output = "<|im_start|>ai: " + dialogue['output']
                    if len(input + output + instruction+ "<|im_end|>") > self.block_size - 1:
                        output = output[:self.block_size - 1 - len("<im_end>")]
                    output += "<|im_end|>"
                    raw_text = instruction+input + output
                    #print(raw_text)
                    encoded_input = self.tokenizer.encode(raw_text, padding='max_length', max_length=self.block_size,truncation='longest_first')
                    #print(len(encoded_input))

                    self.encoded_text.append(encoded_input)
        elif dataset_config.train_dir.endswith('.txt'):
            with open(dataset_config.train_dir, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                instruction_prompt = "<|im_start|>system: 补齐下面的文本内容<|im_end|>"
                for i in range(0, len(raw_text), self.block_size-len(instruction_prompt)):
                    input_text = raw_text[i:i+self.block_size-len(instruction_prompt)]
                    process_text = instruction_prompt + input_text
                    # print(process_text)
                    # print(len(process_text))
                    encoded_input = self.tokenizer.encode(instruction_prompt + input_text, padding='max_length', max_length=self.block_size,truncation='longest_first')
                    self.encoded_text.append(encoded_input)
        else:
            raise ValueError("Invalid file format or file does not exist")

    def __len__(self):
        return len(self.encoded_text)
    def __getitem__(self, idx):
        tokens = self.encoded_text[idx]
        # Masking: 1 for real tokens and 0 for padding tokens
        targets = tokens[1:] + [self.tokenizer.pad_token_id]
        return torch.tensor(tokens), torch.tensor(targets)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.01, path='model.pth'):
        """
        Args:
            patience (int): 允许模型在验证集上性能没有提升的 epoch 数量，默认值：7
            verbose (bool): 是否打印早停信息，默认值：False
            delta (float): 允许模型性能波动的最小值，默认值：0.0
            path (str): 保存最佳模型参数的文件路径，默认值：'model.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存模型参数"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def eval_model_params(model):
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
    #返回精度.3f
    return f"{param_sum/1e6:.3f}"

def dump_train_record(save_pth,train_loss,valid_loss):
    jsonl_file = f"{save_pth}/train_record.jsonl"
    #先检查文件是否存在，不存在则创建
    if not os.path.exists(jsonl_file):
        with open(jsonl_file, 'w', encoding='utf-8') as file:
            file.write('')
    with open(jsonl_file, 'a', encoding='utf-8') as file:
        record = {
            "train_loss": train_loss,
            "valid_loss": valid_loss
        }
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def train(config=train_config):
    if config.is_continue_training:
        model = Transformer(config.model_config).to(config.device)
        model.load_state_dict(torch.load(config.model_path))
    else:
        model = Transformer(config.model_config).to(config.device)
    def grokking_signal_fn():
         return train_config.grokking_signal
    train_dataset = process_dataset(config.train_dataset_config)
    valid_dataset = process_dataset(config.valid_dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = GrokAdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, grokking_signal_fns=[grokking_signal_fn])
    #optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #训练记录文件夹名定义： 模型大小_损失记录，检查并创建文件夹
    loss_dir = f"{eval_model_params(model)}M_model_{config.train_dir_savesuffix}"
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    #训练记录在每次保存模型时，保存一次损失记录，用于绘制训练曲线，记录内容有：训练损失，验证损失
    #训练可视化
    start = time.time()
    early_stopping = EarlyStopping(patience=7, verbose=True, path=f'{loss_dir}/model_best.pth')
    progress_bar = tqdm(total=len(train_loader)*config.epoch, desc="Training Progress", unit="iter")
    #模型训练过程：预热，梯度累计，梯度更新，记录损失，早停
    iter = 0
    # 初始化损失值为 None
    train_loss_value = -1
    valid_loss_value = -1
    optimizer.zero_grad()
    for epoch in range(config.epoch):
        model.train()
        train_loss_sum = 0
        for iteration, (inputs, targets) in enumerate(train_loader):
            iter += 1
            model(inputs.to(config.device), targets.to(config.device))
            loss = model.last_loss
            #print(loss)
            loss.backward()
            train_config.grokking_signal = loss.item()
            # 梯度累计
            if iter % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss_sum += loss.item()
            # 更新验证损失
            if iter % config.loss_update_interval == 0:
                model.eval()
                valid_loss_sum = 0
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        model(inputs.to(config.device), targets.to(config.device))
                        valid_loss_sum += model.last_loss.item()
                valid_loss_value = valid_loss_sum / len(valid_loader)
                train_loss_value = train_loss_sum / config.loss_update_interval
                model.train()
                train_loss_sum = 0

                # 早停
                early_stopping(valid_loss_value, model)
                dump_train_record(loss_dir,train_loss_value,valid_loss_value)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # 更新进度条
            end = time.time()
            avg_time_per_iter = (end - start) / (iteration + 1)
            progress_bar.set_postfix(loss=f'{loss.item():.4f}',time_per_iter=f'{avg_time_per_iter:.4f}s')
            progress_bar.update(1)


        if early_stopping.early_stop:
            break  # 退出整个 epoch 循环

    progress_bar.close()


















  