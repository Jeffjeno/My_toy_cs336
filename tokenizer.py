from typing import Iterable, Iterator,Dict,List,Tuple
import regex as re 
def _to_byte_syms(s: str) -> tuple[bytes, ...]:
    b = s.encode("utf-8")
    return tuple(bytes([x]) for x in b) 
class Tokenizer:
    def __init__(self, vocab: Dict[int,bytes], merges :List[Tuple[bytes,bytes]], special_tokens:List[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.merges_rank =  {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens =[]
        self._pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        #=====check sp_tokens
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in self.vocab.values():
                    new_id = max(self.vocab.keys())+1
                    self.vocab[new_id] = b
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        with open(vocab_filepath,"rb") as f:
            for line in f:
                token_id,token = line.strip().split()
                vocab[int(token_id)] = token
        
        merges = []
        with open(merges_filepath,"rb") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(b"#"):   # ← 跳过 "#version: ..."
                    continue
                a,b = line.split()
                merges.append((a,b))
        
        return cls(vocab,merges,special_tokens)
    
    def _bpe_merges(self, input):
        parts = list(input)
        if len(parts) <=1:
            return parts
        while True:
            best_i = -1
            best_rank= None
            for i in range(len(parts)-1):
                pair = (parts[i],parts[i+1])
                r = self.merges_rank.get(pair)
                if (r is not None) and (best_rank == None or r < best_rank):
                    best_rank = r
                    best_i = i
            
            if best_i == -1:
                break
            else:
                mer = parts[best_i]+parts[best_i+1]
                parts[best_i:best_i+2] = [mer]
            
        return parts
                    
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        #step1:Pre-tokenize
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            pattens = re.split(pattern,text)
        else:
            pattens = [text]
        
        pre_toks = []
        for pt in pattens:
            if pt  in self.special_tokens:
                pre_toks.append(pt)
                continue
            if not pt:
                continue
            for match in re.finditer(self._pat,pt):
                w = _to_byte_syms(match.group())
                pre_toks.append(w)
        #Step 2: Apply the merges
        seq = []
        for pre_tok in pre_toks:
            #here pre_tok is a tuple[bytes]
            if isinstance(pre_tok,str) and pre_tok in self.special_tokens:
                encode_sp = pre_tok.encode(encoding="utf-8")
                seq.append(self.rev_vocab[encode_sp])
                continue
            
            #=====
            else:
                for b in self._bpe_merges(pre_tok):
                    seq.append(self.rev_vocab[b])
            #======
        return seq
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for stream in iterable:
            #step1:Pre-tokenize
            if self.special_tokens:
                pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
                pattens = re.split(pattern,stream)
            else:
                pattens = [stream]
            
            
            pre_toks = []
            for pt in pattens:
                if pt  in self.special_tokens:
                    pre_toks.append(pt)
                    continue
                if not pt:
                    continue
                for match in re.finditer(self._pat,pt):
                    w = _to_byte_syms(match.group())
                    pre_toks.append(w)
            #Step 2: Apply the merges
            for pre_tok in pre_toks:
                #here pre_tok is a tuple[bytes]
                if isinstance(pre_tok,str) and pre_tok in self.special_tokens:
                    encode_sp = pre_tok.encode(encoding="utf-8")
                    yield self.rev_vocab[encode_sp]
                    continue
                
                else:
                    for b in self._bpe_merges(pre_tok):
                        yield self.rev_vocab[b]
    
    def decode(self, ids: list[int]) -> str:
        out = []
        for id in ids:
            out.append(self.vocab[id])
        _out = b"".join(out)
        return _out.decode(encoding="utf-8", errors ="replace" )