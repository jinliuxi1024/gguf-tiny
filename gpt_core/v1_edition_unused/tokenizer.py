import os
import sentencepiece as spm

from transformers import PreTrainedTokenizer
from typing import List, Optional, Union
config ={
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_layers": 6,
    "vocab_size": 8192,
    "max_seq_len": 512,
    "model_prefix": '../spk_path/sp',
    "sp_model_path": '../spk_path/sp.model',
    "sp_train_dir": '../train_dataset/非正式语料.txt',
    "sp_model_type": 'bpe',
    "special_tokens": ['\n','蒋','鹤']
}

class sp_train:
    def __init__(self,config):
        self.model_prefix = config['model_prefix']
        self.vocab_size = config['vocab_size']
        self.model_type = config['sp_model_type']
        self.train_dir = config['sp_train_dir']
    def train(self):
        spm.SentencePieceTrainer.train(input=self.train_dir, model_prefix=self.model_prefix, vocab_size=self.vocab_size,
                                       model_type=self.model_type, user_defined_symbols=config['special_tokens'])
    def get_model_path(self):
        return self.model_prefix + '.model'
    def get_vocab_path(self):
        return self.model_prefix + '.vocab'



class SPTokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.unk_id: int = self.sp_model.unk_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        special_tokens = ["sop", "eop", "pad"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def encode(self, s: str, bos: bool = False, eos: bool = False):
        assert isinstance(s, str), s
        t = self.sp_model.encode(s)
        if bos:
            s = "<bos> " + s
        if eos:
            s = s + " <eos>"
        return t

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def decode(self, t: List[int]) -> str:
        assert isinstance(t, list), t
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        return self.sp_model.decode_pieces(tokens)

    def convert_token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        if index in self.index_special_tokens or index in [self.bos_id, self.eos_id, self.pad_id] or index < 0 :
            return ""
        #[pad] skip
        return self.sp_model.IdToPiece(index)

class GPTTokenizer(PreTrainedTokenizer):
    def __init__(self, config):
        self.tokenizer = SPTokenizer(config['sp_model_path'])
        self.special_tokens = {
            "bos": self.tokenizer.bos_id,
            "eos": self.tokenizer.eos_id,
            "pad": self.tokenizer.pad_id,
        }

        super().__init__(clean_up_tokenization_spaces=True,pad_token='[pad]')

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.decode_tokens(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("bos")]
        return prefix_tokens

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[
        int]:
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("eos")]
        return token_ids_0



if __name__ == "__main__":
    sp = sp_train(config)
    sp.train()
    words_a = '最是人间留不住，朱颜辞镜花辞树。'
    words_b = "十年生死两茫茫，不思量，自难忘。蒋云鹤"
    tokenizer = GPTTokenizer(config)
    ids = tokenizer.encode(words_b, padding='max_length', max_length=32)
    print(ids)
    print(tokenizer.decode(ids,skip_special_tokens=False))
    print(tokenizer.pad_token_id)
    print(tokenizer.tokenizer.bos_id)


