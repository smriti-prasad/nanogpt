# EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? 
# (A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left too. 
# You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. 
# You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index).
#  Does your Transformer learn to add?

# vocab_size is the number of different tokens the model can ever see or output
# EDIT: removed `from xml.parsers.expat import model` (stray auto-import)

import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 32       # EDIT: moved up from the later `batch_size = 32` line (4 was too small)
block_size = 8
dropout = 0.0         # EDIT: 0.2 -> 0.0, every batch is fresh random data so nothing to regularize

# Training setup (next-token prediction):
# Build one full sequence: a + b = c_reversed  (e.g. 07+45=250 -> "0 7 + 4 5 = 2 5 0")
# x = seq[:-1], y = seq[1:]  -> y is x shifted left by one, so both have the same length.
# At every position i, the model sees only x[0..i] (causal mask) and predicts y[i].
#
# Teacher forcing: the answer digits in x are the TRUE previous digits, so each
# answer digit is predicted from the question + earlier answer digits, and all
# positions train in parallel. At generation time, the model's own outputs are
# fed back in instead.
#
# Loss mask: y = -1 for the problem part (a, +, b, =). The model still makes
# predictions there, but ignore_index=-1 drops them from the loss, because
# a and b are random and can't be predicted. Only the 3 answer digits are learned.
#
# Digits of c are reversed (ones first) to match right-to-left addition with carry.

def get_batch_train():
# ID:       0 1 2 3 4 5 6 7 8 9 10 11
# Meaning:  0 1 2 3 4 5 6 7 8 9  +  =      (EDIT: PAD removed)

    a = torch.randint(0, 100, (batch_size, 1), dtype=torch.long)
    b = torch.randint(0, 100, (batch_size, 1), dtype=torch.long)
    c = a + b

    # Reverse the digits of each c
    c_reversed = []

    for i in range(len(c)):
        number = c[i].item()
        reversed_number = 0
        element_wise_tensor = []
        while number > 0:
            digit = number % 10
            reversed_number = reversed_number * 10 + digit
            number = number // 10
            element_wise_tensor.append(digit)

        c_reversed.append(element_wise_tensor)

    def pad_sequence(seq, param,  max_length):
        padded_seq = seq + [param] * (max_length - len(seq))
        return padded_seq

    # Input 
    x = []

    for i in range(len(a)):
        first = [int(d) for d in str(a[i].item()).zfill(2)]
        second = [int(d) for d in str(b[i].item()).zfill(2)]

        sequence = first + [10] + second + [11] 
        x.append(sequence)

    x1 = x

    # Target
    y1 = [pad_sequence(seq, 0, 3) for seq in c_reversed]

    # join row by row, then convert to tensor
    seq = torch.tensor([xi + yi for xi, yi in zip(x1, y1)])    # (B, 9)

    # shift to get x and y, mask the problem part
    x = seq[:, :-1]                                            # (B, 8)
    y = seq[:, 1:].clone()                                     # (B, 8)
    y[:, :5] = -1

    return x, y

print("Example batch:")
x,y  = get_batch_train()
print("Input shape:", x.shape)
print("Target shape:", y.shape) 
print("Input (a, b):", x)
print("Target:", y)

vocab_size = 12  # 0-9, +, =        # EDIT: removed stale "PAD"
n_embd = 64  # embedding dimension
# EDIT: the long "notepad" comment that was inside the attention class lives here now:
# n_embd is the size of each token's "notepad" vector: room to hold which digit it is,
# where it sits, the partial sum, whether there's a carry. You choose it.
head_size = 16
num_heads = n_embd // head_size  # number of attention heads (=4)
n_layer = 3                      # EDIT: number of stacked blocks (was hard-coded as 3 Block(...) lines)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_head = num_heads                                     # EDIT: store n_head
        self.head_size = head_size
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)                       # EDIT: replaces lm_head; output stays n_embd wide
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # EDIT: removed self.lm_head and self.n_embd (lm_head belongs only in calculatorGPT)

    def forward(self, x):
        B, T, C = x.shape                                           # C = n_embd

        q, k, v = self.c_attn(x).split(C, dim=2)                    # EDIT: split(C); 3 x (B,T,C)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)   # EDIT: (B, nh, T, hs), was view(B,T,head_size,head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)    # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v                                               # EDIT: was `logits = wei @ v`; (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)        # re-assemble all head outputs side by side
        return self.dropout(self.proj(out))                         # EDIT: project back to n_embd and RETURN (was lm_head, no return)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            #The expansion gives the network more "working space" to do computation before compressing back down. This 4x ratio comes from the original Attention is All You Need paper.
            nn.Linear(n_embd,4*n_embd), #times 4 due to feed forward layer expansion
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), #this is the projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# EDIT: deleted the duplicate Block class; kept one
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, num_heads):
        # n_embd: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embd // num_heads        # EDIT: restored (the 2nd copy had it commented out)
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class calculatorGPT(nn.Module):
    def __init__(self): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # EDIT: build blocks from n_layer instead of 3 hard-coded lines
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)    # EDIT: final LayerNorm now its own attribute (was inside the Sequential)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=idx.device)) # EDIT: idx.device, `device` was undefined
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)   # EDIT: apply final LayerNorm
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # compute the loss over all valid positions
            logits_flat = logits.reshape(-1, logits.size(-1))   # EDIT: reshape instead of view (safer)
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)

        return idx


m = calculatorGPT()          # EDIT: no vocab_size argument (the class reads the global)
logits, loss = m(x, y)
print("logits shape:", logits.shape)
print("loss:", loss)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# EDIT: removed the later `batch_size = 32` (now set at the top)
losses = []                                   # EDIT: collect losses to print an average
for steps in range(5000):
    x, y = get_batch_train()
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())                # EDIT

    if steps % 100 == 0:
        print(f"step {steps}: avg loss {sum(losses)/len(losses):.4f}")   # EDIT: averaged, less noisy
        losses = []                                                      # EDIT

@torch.no_grad()
def predict(a, b):
    m.eval()
    # build the prompt: a + b =  (same fixed layout as training)
    idx = torch.tensor([[a // 10, a % 10, 10, b // 10, b % 10, 11]])   # (1, 6)

    for _ in range(3):                                   # 3 answer digits
        logits, _ = m(idx[:, -block_size:])          # (1, T, 12)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # best next token
        idx = torch.cat([idx, nxt], dim=1)               # feed it back in

    ones, tens, hundreds = idx[0, 6:].tolist()           # generated digits, reversed order
    m.train()
    return ones + 10 * tens + 100 * hundreds

print(predict(37, 45), "expected", 37 + 45)
print(predict(76, 67), "expected", 76 + 67)

# EDIT: added accuracy check over many random problems (the real test of "did it learn to add")
correct = 0
n_test = 1000
for _ in range(n_test):
    a, b = torch.randint(0, 100, (2,)).tolist()
    correct += (predict(a, b) == a + b)
print(f"accuracy on {n_test} random problems: {correct / n_test:.3f}")