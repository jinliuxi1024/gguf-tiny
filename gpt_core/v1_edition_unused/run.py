from gpt_core.v1_edition_unused.train import train
import torch
from model import ModelArgs, Transformer
from gpt_core.v1_edition_unused.tokenizer import GPTTokenizer
config ={
    "device": "cpu" if torch.backends.mps.is_available() else "cpu",
    "dim": 256,
    "n_layers": 8,
    "n_heads": 8,
    "n_kv_heads": 4,
    "multiple_of": 1024,
    "vocab_size": 8192,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 50000.0,
    "max_seq_len": ModelArgs().max_seq_len,
    "train_dir": './train_dataset/pretraindata.jsonl',
    "instruct_dir": "instruct.jsonl",
    "valid_dir": "instruct.jsonl",
    "model_prefix": 'sp_path/sp',
    "sp_model_path": 'sp_path/sp.model',
    "sp_train_dir": '主指导文件_试训练.jsonl',
    "sp_model_type": 'bpe',
    "speaicl_tokens": ['\n','蒋','鹤'],
    "batch_size": 2,
    "lr": 1e-4,
    "epoch": 1,
    "model": None
}
def get_args(config):
    args = ModelArgs(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        ffn_dim_multiplier=config['ffn_dim_multiplier'],
        norm_eps=config['norm_eps'],
        max_seq_len=config['max_seq_len'],
        multiple_of=config['multiple_of'],
        vocab_size=config['vocab_size'],
    )
    return args
def generate(text,max_length,config):
    tokenizer = GPTTokenizer(config)
    #input = "华：{"+text +"}"
    input = text
    tokens = tokenizer.encode(input)
    tokens = torch.tensor(tokens).unsqueeze(0).to(config['device'])
    #print(tokens.shape)
    #计算一下每个token的平均用时
    model = config['model']
    generator = model.generate(tokens, max_new_tokens=max_length, temperature=1, top_k=20, eos=tokenizer.pad_token_id)

    for token in generator:
        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token)),end='')


def train_save(config):
    config['model'] = Transformer(get_args(config)).to(config['device'])
    train(config)



def load_train(config):
    config['model'] = Transformer(get_args(config)).to(config['device'])
    config['model'].load_state_dict(torch.load('model.pth'))
    train(config)

def load_test(config):

    dtype = torch.float16 if config['device'] == 'mps' else torch.float32
    config['model'] = Transformer(get_args(config)).to(device=config['device']).eval()
    #半精度载入
    config['model'].load_state_dict(torch.load('model.pth', map_location=config['device'],weights_only=True))
    text = '给出一个小故事。'
    print('-----------模型载入成功-----------')
    print(text)
    generate(text,512,config)
    print('\n------------推理完成-------------')

def covert_to_cpumodel():
    model = Transformer(get_args(config)).to('cpu').eval()
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    torch.save(model.state_dict(), 'model.pth')
if __name__ == '__main__':
    #train_save(config)
    #load_train(config)
    load_test(config)
    #covert_to_cpumodel()