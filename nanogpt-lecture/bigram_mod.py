# Check the dependencies

# n_embd = 256, n_head = 4 → head_size = 64. Valid.
# batch_size = 8, block_size = 128 → each batch has 8 × 128 = 1,024 input tokens. Valid.
# n_layer = 4 → four Transformer blocks, assuming your model actually uses n_layer to construct them.
# dropout = 0.2 → valid; dropout is active during training and disabled during evaluation.
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters -- external config settings defined here for easy tweaking
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 2000 #increased the number of iterations since self attention is more complex and needs more iterations to converge
eval_interval = 200
learning_rate = 3e-4 #decreased the lr after self attention since self attention is more complex and needs a smaller learning rate to converge
#if you have a GPU, put the model on GPU -- CUDA is much faster
# to_device in the cuda, need to make sure that the model and the data are on the same device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 256 # increased the embedding dimension since self attention is more complex and needs more capacity to learn
#added dropout to the model to prevent overfitting since self attention is more complex and needs
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#averages up the loss over multiple batches of data
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

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # all the linear projections will be done in a batch inside a single linear layer of c_attn, which is more efficient
        # self.key = nn.Linear(n_embd, head_size, bias=False)
        # self.query = nn.Linear(n_embd, head_size, bias=False)
        # self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.n_head = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        #calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B,T,3*hs) -> 3 x (B,T,hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5) # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B,T,C)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            #The expansion gives the network more "working space" to do computation before compressing back down. This 4x ratio isn't arbitrary — it comes from the original Attention is All You Need paper and has stuck ever since.
            nn.Linear(n_embd,4*n_embd), #times 4 due to feed forward layer expansion, this is a common practice in transformer architectures
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), #this is the projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Step 1: attention (communication). Positions look at each other and gather information. At =, this is where it fetches the ones digits of a and b.

# Step 2: feed-forward (computation). Each position processes what it gathered, on its own. This is where “4 and 5 give 9” or “a carry happened” can be computed, thanks to the ReLU.

# So a block is “gather, then think”, and one block does it once.

# The x = x + ... part (residual). The sublayer’s output is added to x instead of replacing it. The original information is always preserved, and each sublayer only contributes a correction. This also makes deep stacks train much more easily, since gradients can flow straight through the additions.

# The LayerNorm part. ln1(x) rescales each token’s vector to have mean 0 and variance 1 across its n_embd numbers, with learned scale and shift. It’s applied to the input of each sublayer (not the residual path itself). It keeps the numbers at a stable size as you stack layers, so training doesn’t blow up or stall.
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, num_heads):
        # n_embd: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd),
        )
        #at this point, we are not satisfied because the neural network has become very deep and optimization issues come to the scene
        #so here are 2 things that we can do to make the optimization easier:
        #1. we can add a residual connection to the self attention and feed forward layers
        #2. we can add layer normalization to the self attention and feed forward layers
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        #x = self.sa_head(x) # (B,T,C)
        #x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #self attention looks at all the tokens
            #therefore we crop the context to the last block_size tokens
            #earlier it was a simpler model so we didnt need it
            idx_cond = idx[:, -block_size:] # crop context to the last block_size tokens
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Device:", device)
print("Starting training...")

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        print("Starting evaluation...")
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))