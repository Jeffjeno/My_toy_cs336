import regex as re
from typing import Dict,List
from multiprocessing import Pool, cpu_count
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter, defaultdict
from functools import partial
import heapq
def _to_byte_syms(s: str) -> tuple[bytes, ...]:
    b = s.encode("utf-8")
    return tuple(bytes([x]) for x in b) 
def pre_tokenization(chunk_bytes,special_tokens: List[str]):
    #========remove sp_tokens=======#
    chunk = chunk_bytes.decode("utf-8", errors="replace")  
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    post_chunk = re.split(pattern,chunk)
    
    #preprocessing

    chunk_vocab = {}
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    for chunk in post_chunk:
        if not chunk:
            continue
        for match in PAT.finditer(chunk):
            w = _to_byte_syms(match.group())
            if w not in chunk_vocab.keys():
                chunk_vocab[w] = 1
            else:
                chunk_vocab[w]+=1
    return chunk_vocab

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
  
    
        
    
    #========init vocab=======#
    vocab: Dict[int,bytes] = {}
    merges = []
    #add special_tokens
    for i,sp_token in enumerate(special_tokens):
        vocab[i] = sp_token.encode("utf-8")
    offset = len(special_tokens)
    #add base 256 
    for i in range(256):
        vocab[i+offset]= bytes([i])
    next_id = offset + 256
    chunks: List[str] = []
    #=====preprocess=======
    with open(input_path, "rb") as f:   # here use rb to get the bytes level data
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)    #utf level data
            chunks.append(chunk_bytes)    #chunk str
    

    
        
    with Pool(processes=cpu_count()) as pool:
        fn = partial(pre_tokenization,special_tokens=special_tokens) 
        results = pool.map(fn,chunks)
    
    result = Counter()
    
    #here we gather all the results
    for item in results:
        result.update(item)
 
    token_table = dict(result)
    
    
    #=====transfer the structure
    def convert(table):
        pair_freq = Counter()
        for tok,freq in table.items():
            if freq == 0 or len(tok)<2:
                continue
            for i in range(len(tok)-1):
                pair = (tok[i],tok[i+1])
                pair_freq[pair] += freq
        return pair_freq


    #=====tokenizer========#

    
    while(len(vocab) < vocab_size):
        pair_table = convert(token_table)
        (p0, p1), _ = max(pair_table.items(), key=lambda kv: (kv[1], kv[0]))
        merged = p0+p1
        vocab[next_id] = merged
        merges.append((p0, p1))
        next_id += 1
        #update our token table
        new_token_table = {}
        for tok,freq in token_table.items():
            if freq == 0 :
                continue
            if len(tok)<2:
                new_token_table[tok] = new_token_table.get(tok, 0) + freq
                continue
            if (p0 not in tok) or (p1 not in tok):
                new_token_table[tok] = new_token_table.get(tok, 0) + freq
                continue
            #=====
            out = []
            i=0
            n=len(tok)
            while i < n:
                if i+1 < n and tok[i] == p0 and tok[i+1] == p1:
                    out.append(merged)
                    i+=2
                else:
                    out.append(tok[i])
                    i += 1

            out = tuple(out)
            new_token_table[out]= new_token_table.get(out,0)+freq
        token_table = new_token_table

  
        
    return vocab,merges
     

        
    