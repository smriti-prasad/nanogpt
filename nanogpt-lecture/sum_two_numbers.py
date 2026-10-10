# EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? 
# (A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left too. 
# You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. 
# You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index).
#  Does your Transformer learn to add?

# vocab_size is the number of different tokens the model can ever see or output
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 4
block_size = 8

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
# ID:       0 1 2 3 4 5 6 7 8 9 10 11 12
# Meaning:  0 1 2 3 4 5 6 7 8 9  +  =  PAD

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
            

        #print(element_wise_tensor)
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


    x1 = x #[pad_sequence(seq, 12, block_size) for seq in x]

    # Target
    y1 = [pad_sequence(seq, 0, 3) for seq in c_reversed]

    # CHANGED: join row by row, then convert to tensor
    seq = torch.tensor([xi + yi for xi, yi in zip(x1, y1)])    # (B, 9)

    # CHANGED: shift to get x and y, mask the problem part
    x = seq[:, :-1]                                            # (B, 8)
    y = seq[:, 1:].clone()                                     # (B, 8)
    y[:, :5] = -1

    return x, y

print("Example batch:")
x,y  = get_batch_train()
print("Input shape:", x.shape)
print("Target shape:", y.shape) 
#print("total", len(seq1))
print("Input (a, b):", x)
print("Target:", y)

# what GPT is expecting:
# its idx is in the shape of B, T --- and since it requires to be rectangular, we will pad the target with -1 to ignore the loss at those positions

vocab_size = 12  # 0-9, +, =, PAD
n_embd = 64  # embedding dimension
head_size = 16

class calculatorGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # creating lookup table for the vocab size
# Problem 1: an integer isn’t useful by itself
# The id 7 is only a row number in a table. The model can’t add, compare, or attend using “7” as a bare integer. So we give each token id a learned vector of numbers, and training adjusts those numbers until they’re useful. That’s what an embedding is.
# Problem 2: the model needs a workspace
# Think of each position’s vector as a small notepad that gets passed through the layers. While computing the ones digit of the answer, the notepad has to hold things like:
# which digit this token is
# where it sits (ones of a, tens of b, ...)
# the partial sum so far
# whether there’s a carry

# If the notepad has only 12 slots, there is little room to hold all that at once. With 64 slots, there’s plenty. n_embd is just the size of that notepad. You choose it; bigger gives more room but costs more compute.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # create a single head of self-attention
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # create a linear layer to project the output of the attention head to the vocab size
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        # get the logits from the embedding table
        token_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
    
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        logits = wei @ v  # (B, T, head_size)
        logits = self.lm_head(logits)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # compute the loss over all valid positions
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
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


m = calculatorGPT(vocab_size)
logits, loss = m(x, y)
print("logits shape:", logits.shape)
print("loss:", loss)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(5000):
    x, y = get_batch_train()
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(f"step {steps}: loss {loss.item()}")

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

print(predict(34, 45), "expected", 34 + 45)
print(predict(58, 67), "expected", 58 + 67)
        
        