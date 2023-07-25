import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparams
batch_size = 32
block_size = 8
max_iter = 5000
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 32

torch.manual_seed(42)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()


#Prepare chars
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


#Mappings
char_to_idx = { char:idx for idx, char in enumerate(chars)}
idx_to_char = { idx:char for idx, char in enumerate(chars)}
encode = lambda chars: [char_to_idx[char] for char in chars]
decode = lambda idxs: ''.join([idx_to_char[idx] for idx in idxs])

#Test Train split
data = torch.tensor(encode(corpus), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

#Batchloader
def get_batch(split):
    #Build batch for train or eval
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X = X.to(device) 
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Self attention head"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # attention dot product
        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, C) * (B, C, T) results in # (B, T, T)
        # get triangular shape as mask and normalize
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # Final multiply with v
        v = self.value(x)
        out = wei @ v # (B, T, T) * # (B, T, C) results in # (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Stacked Heads of SelfAttention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim =-1)

class BigramLanguageModel(nn.Module):

    #Only uses embdding table and no additional architecture
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb) #positional encoding
        self.sa_heads = MultiHeadAttention(4, n_emb//4) # 4 heads of 8-dim self attention concats to 32
        self.lm_head = nn.Linear(n_emb, vocab_size)

    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.token_embedding_table(torch.arange(T, device=device)) # (T, C)
        x       = tok_emb + pos_emb # (B, T, C)
        x       = self.sa_heads(x)
        logits  = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)


        return logits, loss
    
    def generate(self, idx, max_new_tokens):
            #idx is (B, T) array of indices 
        for _ in range(max_new_tokens):
            # crop idx to the last block size head
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on last element
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # get sample form the prob - distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # apppend new sample index to existing sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

#Initialize model and optimizer
model = BigramLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

#Very simple train and eval loop
for steps in range(max_iter):

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample a batch
    xb, yb, = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    #evaluate the loss

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


