import trainer_merge
from trainer_merge import train_config
from trainer_merge import dataset_config
from tokenizer_merge import sp_config
from model import ModelArgs
from model import Transformer
from tokenizer_merge import sp_tokenizer
from pth2safetensors import convert_pth_to_safetensors
import torch
import os
train_config = train_config(
    train_dataset_config = dataset_config(
        max_seq_len = 512,
        sp_config = sp_config(
            vocab_size = 8192,
            max_seq_len = 512,
            model_prefix = 'spm_path/spm',
            sp_model_path = 'spm_path/spm.model',
            sp_model_type = 'bpe',
            special_tokens = ['<|im_start|>', '<|im_end|>', '\n']
        ),
        train_dir ='train_data/txt_data/非正式语料.txt'
    ),
    valid_dataset_config = dataset_config(
        max_seq_len = 512,
        sp_config = sp_config(
            vocab_size = 8192,
            max_seq_len = 512,
            model_prefix = 'spm_path/spm',
            sp_model_path = 'spm_path/spm.model',
            sp_model_type = 'bpe',
            special_tokens = ['<|im_start|>', '<|im_end|>', '\n']
        ),
        train_dir ='train_data/txt_data/凡人修仙传.txt'
    ),
    model_config = ModelArgs(
        dim=128,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=8192,
        multiple_of = 128,
        norm_eps = 1e-05,
        rope_theta = 10000.0,
        max_batch_size= 32,
        max_seq_len = 512
    ),
    device = 'mps' if torch.backends.mps.is_available() else 'cpu',
    batch_size = 4,
    lr = 1e-4,
    epoch = 32,
    is_continue_training = False,
    train_dir_savesuffix = 'train_records',
    loss_update_interval = 100,
    model_save_interval = 1000,
    gradient_accumulation_steps = 1,
    weight_decay = 0.01,
    lr_scheduler = 'linear',
    model_path = 'model.pth'
)

def train_and_save():
    trainer_merge.train(train_config)

train_record_base_path = '2.884736M_model_train_records'
def load_and_generate():
    text = '十年生死两茫茫'
    system_prompt_text = "<|im_start|>system: 补齐下面的文本内容<|im_end|>"
    prompt_text = "<|im_start|>" + text
    model = Transformer(train_config.model_config).to('cpu')
    model_path = os.path.join(train_record_base_path, 'model_best.pth')
    #检查模型是否存在
    if not os.path.exists(model_path):
        print('model not found')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokenizer = sp_tokenizer(train_config.train_dataset_config.sp_config)
    tokens = tokenizer.encode(system_prompt_text + prompt_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to('cpu')
    generator = model.generate(tokens, max_new_tokens=train_config.model_config.max_seq_len, temperature=1, top_k=20,eos=tokenizer.get_command("<|im_end|>"))
    for token in generator:
        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token)),end='')

def load_and_train():
    model_path =os.path.join(train_record_base_path, 'model_best.pth')
    train_config.is_continue_training = True
    train_config.model_path = model_path
    trainer_merge.train(train_config)

def convert_model_from_pth_to_safetensor():
    pth_file = os.path.join(train_record_base_path, 'model_best.pth')
    output_file = os.path.join(train_record_base_path, 'model_best.safetensors')
    convert_pth_to_safetensors(pth_file, output_file)

if __name__ == '__main__':
    #train_and_save()
    #load_and_generate()
    #convert_model_from_pth_to_safetensor()
    load_and_train()