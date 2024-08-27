import os
import sentencepiece as spm
from transformers import PreTrainedTokenizer
from dataclasses import dataclass, field

#使用类来设置config
@dataclass
class sp_config:
    vocab_size :int = 16384
    max_seq_len : int = 512
    model_prefix : str = 'spk_path/spm'
    sp_model_path : str = 'spk_path/spm.model'
    sp_train_dir :str = 'train_data/200M标准训练语料.jsonl'
    sp_model_type :str = 'bpe'
    special_tokens : list = field(default_factory=lambda: ['<|im_start|>', '<|im_end|>', '\n'])



class sp_tokenizer(PreTrainedTokenizer):
    def __init__(self,config):
        self.model_prefix = config.model_prefix
        self.model_type = config.sp_model_type
        self.train_dir = config.sp_train_dir
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.get_model_path())
        self.special_tokens = {
            "<|im_start|>": self.tokenizer.piece_to_id("<|im_start|>"),
            "<|im_end|>": self.tokenizer.piece_to_id("<|im_end|>"),
            "\n": self.tokenizer.piece_to_id("\n")
        }
        super(sp_tokenizer, self).__init__(clean_up_tokenization_spaces=True,bos_token='<|im_start|>', eos_token='<|im_end|>', pad_token='<|im_end|>')
    def get_model_path(self):
        return self.model_prefix + '.model'
    def get_vocab_path(self):
        return self.model_prefix + '.vocab'
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    def _tokenize(self, text, **kwargs):
        return self.tokenizer.EncodeAsPieces(text)
    def _convert_token_to_id(self, token):
        return self.tokenizer.piece_to_id(token)
    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_piece(index)
    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]
    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("<|im_start|>")]
        return prefix_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<|im_end|>")]
        return token_ids_0




def train_tokenizer(config):
    spm.SentencePieceTrainer.train(
        input=config.sp_train_dir,
        model_prefix=config.model_prefix,
        vocab_size=config.vocab_size,
        model_type=config.sp_model_type,
        user_defined_symbols=config.special_tokens,
        character_coverage=1.0,
    )
    print('训练完成')
    print('模型路径:',config.model_prefix + '.model')
    print('词表路径:',config.model_prefix + '.vocab')

if __name__ == '__main__':
    config = sp_config()
    train_tokenizer(config)
    sp = sp_tokenizer(config)
    text = '给出一个小故事。'
    tokens = sp.tokenize(text)
    print(tokens)
    print('-'*20)
    ids = sp.convert_tokens_to_ids(tokens)
    print(ids)
    print('-'*20)
    special_tokens = sp.build_inputs_with_special_tokens(ids)
    print(special_tokens)
    print('-'*20)
    print(sp.convert_ids_to_tokens(special_tokens))
    print('-'*20)
    print(sp.convert_tokens_to_string(tokens))