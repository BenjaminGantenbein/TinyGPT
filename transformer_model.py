import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparams
batch_size = 64
block_size = 256
max_iter = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 384
n_head = 6
n_layer = 6
dropout = 0.2

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
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Self attention head"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key        = nn.Linear(n_emb, head_size, bias=False)
        self.query      = nn.Linear(n_emb, head_size, bias=False)
        self.value      = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # attention dot product
        wei = q @ k.transpose(-2, -1) *  k.shape[-1]**-0.5 # (B, T, C) * (B, C, T) results in # (B, T, T)
        # get triangular shape as mask and normalize
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Final multiply with v
        v = self.value(x)
        out = wei @ v # (B, T, T) * # (B, T, C) results in # (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Stacked Heads of SelfAttention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads      = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj       = nn.Linear(head_size * num_heads, n_emb)
        self.dropout    = nn.Dropout(dropout)  
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    """Introducing non-linear ReLu"""

    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )  

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Actual Transformer Block: First create attention then compute"""
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size   = n_emb // n_head
        self.sa     = MultiHeadAttention(n_head, head_size)
        self.ffwd   = FeedForward(n_emb)
        self.ln1    = nn.LayerNorm(n_emb)
        self.ln2    = nn.LayerNorm(n_emb)


    def forward(self, x):
        x           = x + self.sa(self.ln1(x))
        x           = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table      = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table   = nn.Embedding(block_size, n_emb) #positional encoding
        self.blocks                     = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.ln_f                       = nn.LayerNorm(n_emb)
        self.lm_head                    = nn.Linear(n_emb, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x       = tok_emb + pos_emb # (B, T, C)
        x       = self.blocks(x)    # (B, T, C)
        x       = self.ln_f(x)
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

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

#Very simple train and eval loop
for steps in range(max_iter):

    if steps % eval_interval == 0 or steps == max_iter-1:
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

#print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))