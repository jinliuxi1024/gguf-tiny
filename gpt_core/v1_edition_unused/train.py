#训练模块
from model import ModelArgs,print_model_parameters
from gpt_core.v1_edition_unused import tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import json

from tqdm import tqdm
import torch
config ={
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "dim": 512,
    "n_layers": 4,
    "n_heads": 8,
    "n_kv_heads": 4,
    "multiple_of": 1024,
    "vocab_size": 8192,
    "ffn_dim_multiplier": 1.3,
    "norm_eps": 1e-05,
    "rope_theta": 50000.0,
    "max_seq_len": 256,
    "train_dir": './train_dataset/pretraindata.jsonl',
    "instruct_dir": "instruct.jsonl",
    "model_prefix": 'sp_path/sp',
    "sp_model_path": 'sp_path/sp.model',
    "sp_train_dir": '主指导文件_试训练.jsonl',
    "sp_model_type": 'bpe',
    "speaicl_tokens": ['\n','蒋','鹤'],
    "batch_size": 4,
    "lr": 1e-4,
    "epoch": 5,
}
class gptDataset(Dataset):
    def __init__(self,config,dir):
        self.encoded_dialogues = []
        self.block_size = config['max_seq_len']
        self.tokenizer = tokenizer.GPTTokenizer(config)
        # 检查文件是否存在
        with open(dir, 'r', encoding='utf-8') as file:
            dialogues = [json.loads(line) for line in file]
        for dialogue in dialogues:
            #print(dialogue)
            input = "华：{"+dialogue['instruction'] +"}"
            output = "蒋云鹤：{"+dialogue['output']
            if  len(input) + len(output) > self.block_size-1:
                output = output[:self.block_size-1-len(input)]
            output += "}"
            encoded_input = self.tokenizer.encode(input, output, padding='max_length', max_length=self.block_size-1,
                                                  truncation='longest_first')
            self.encoded_dialogues.append(encoded_input)
    def __len__(self):
        return len(self.encoded_dialogues)

    def __getitem__(self, idx):
        tokens = self.encoded_dialogues[idx]
        # Masking: 1 for real tokens and 0 for padding tokens
        targets = tokens[1:] + [self.tokenizer.pad_token_id]
        return torch.tensor(tokens), torch.tensor(targets)


class GrokAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 alpha_init=0.98, lamb=2.0, gamma=0.1, grokking_signal_fns=None,
                 grokking_signal_decay_rate=0.1, gradient_clipping=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_init=alpha_init, lamb=lamb, gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(GrokAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                if group['gradient_clipping'] > 0:
                    grad = torch.clamp(grad, -group['gradient_clipping'], group['gradient_clipping'])

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grok_ema'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, grok_ema = state['exp_avg'], state['exp_avg_sq'], state['grok_ema']
                beta1, beta2 = group['betas']

                state['step'] += 1

                layer_beta1 = beta1 * (1 - group['gamma']) ** i

                alpha = group['alpha_init'] * torch.exp(
                    torch.tensor(-group['grokking_signal_decay_rate'] * grokking_signal))
                grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
                grok_grad = grad + group['lamb'] * grok_ema

                exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def train(config):
    train_dataset = gptDataset(config,config['train_dir'])
    instruct_dataset = gptDataset(config,config['instruct_dir'])
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
    model = config['model']
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    instruct_loader = DataLoader(instruct_dataset, batch_size=config['batch_size'], shuffle=True)
    print_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    #使用tqdm显示进度条
    start = time.time()
    progress_bar = tqdm(total=len(train_loader), desc="Training Progress", unit="iter")
    instruct_iter = iter(instruct_loader)
    num_batches_per_epoch = len(train_loader)
    num_instruct_batches = int(0.05 * num_batches_per_epoch)
    for epoch in range(config['epoch']):
        for iteration, (inputs, targets) in enumerate(train_loader):
            model(inputs.to(config['device']), targets.to(config['device']))
            loss = model.last_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 进度条更新
            end = time.time()
            avg_time_per_iter = (end - start) / (iteration + 1)
            progress_bar.set_postfix(loss=loss.item(), avg_time_per_iter=avg_time_per_iter)
            progress_bar.update(1)

            # 使用指导训练集进行微调
            if iteration % (num_batches_per_epoch // num_instruct_batches) == 0:
                try:
                    instruct_inputs, instruct_targets = next(instruct_iter)
                except StopIteration:
                    instruct_iter = iter(instruct_loader)
                    instruct_inputs, instruct_targets = next(instruct_iter)

                model(instruct_inputs.to(config['device']), instruct_targets.to(config['device']))
                instruct_loss = model.last_loss
                optimizer.zero_grad()
                instruct_loss.backward()
                optimizer.step()

            # 保存模型
            if iteration % 1000 == 0:
                model.cpu()
                torch.save(model.state_dict(), 'model.pth')
                model.to(config['device'])
    torch.save(model.state_dict(), 'model.pth')
    progress_bar.close()

if __name__ == '__main__':
    train(config)
