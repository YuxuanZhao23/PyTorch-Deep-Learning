import torch
import torch.nn as nn
from torch.nn import functional as F

# region utils
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
model_path = 'blm.pth'
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
s2i = {ch:i for i,ch in enumerate(chars)}
i2s = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [s2i[c] for c in s]
decode = lambda l: ''.join([i2s[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
test = data[n:]

def get_batch(isTrain):
    data = train if isTrain == 'train' else test
    randomIndex = torch.randint(len(data) - block_size, (batch_size, )) # block size 4, batch size 2: 
    x = torch.stack([data[i: i + block_size] for i in randomIndex]) # [[55, 36, 84, 92], [63, 45, 13, 62]]
    y = torch.stack([data[i + 1: i + block_size + 1] for i in randomIndex]) # [[36, 84, 92, 10], [45, 13, 62, 33]]
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval() # set to evaluation phase, 为什么需要？因为我们只在训练期间使用 dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set back to training phase
    return out

def save_model(model, optimizer, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    print("Model saved!")

def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded!")

def train_model():
    for steps in range(max_iters):
        if steps % eval_iters == 0 or steps == max_iters - 1:
            losses = estimate_loss()
            print(f"{100*steps/max_iters}%: step {steps}: train loss {losses['train']:.4f}, validate loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # 因为默认gradient会累加
        loss.backward()
        optimizer.step()
    save_model(model, optimizer, model_path)

def inference():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=5000)[0].tolist()))
# endregion

class Head(nn.Module):
    def __init__(self, head_size): # 什么叫 self-attention？也就是说 key, value, query 都是基于相同的 x 产生的
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 不参与 gradient
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)，计算 attention scores，也称 affinities
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        return wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads * head_size)
        out = self.proj(out) # (B, T, n_embd) 只是 Block 调用的时候恰好设置了 head_size = n_embd // n_head
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # transformer paper 里面的 feed forward 就是差不多 4 倍，这个扩大很重要
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) # 这是不同于原始paper的实现，原始paper是在attention和feed forward之后lay norm，大家发现改到前面会更好
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding 可以理解成key是token，value是长度为384的向量的dict
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # key现在是block size index
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None): # index 和 targets 都是 (batch, block)
        B, T = index.shape
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) token embedding 和 position embedding 是相加的关系
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) logit 是向量的意思，在这里可以理解成预测值

        if targets is None: # inference 模式
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # 整理一下格式，Pytorch 的格式是 (batch, class, ... )
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(index[:, -block_size:]) # predict, 加了 attention 之后，我们现在只能往回看 block 这么多的内容了
            logits = logits[:, -1, :] # the last character prediction, (batch, character)
            probs = F.softmax(logits, dim=-1) # 把输出转换成概率
            index_next = torch.multinomial(probs, num_samples=1) # (batch, 1)，multinomial 会基于 prob distribution 来 sample
            index = torch.cat((index, index_next), dim=1) # (batch, block + 1)
        return index
    
model = BigramLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train_model()

load_model(model, optimizer, model_path)
inference()