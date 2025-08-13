import torch
import torch.nn as nn
from torch.nn import functional as F

#---HYPERPARAMETERS---
batch_size = 32                                         # how many independent sequences will we process in parallel?
block_size = 8                                          # what is the maximum context length for predictions?
max_iters = 3000                                        # how many steps to take
eval_interval = 300                                     # how often to evaluate the loss on the val set
learning_rate = 1e-2                                    # learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device to train on
eval_iters = 200                                        # how many iterations to evaluate the loss on the val set
n_embed = 32                                            # size of each embedding vector

torch.manual_seed(1337)                                 # seed for reproducibility

#---DATA---
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create mapping from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#---TRAIN/TEST SPLIT---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

#---DATA LOADER---
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,)) # batch_size num. of random offsets in dataset
    x = torch.stack([data[i:i + block_size] for i in idx]) # (batch_size, block_size)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx]) # (batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y # (batch_size, block_size), (batch_size, block_size)

"""
Evaluate the loss on the train and val sets every few iterations.
"""
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

#---DEFINE THE BIGRAM LANGUAGE MODEL---
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # embedding table for the tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # embedding table for the positions | adds the notion of space

        self.ln_head = nn.Linear(n_embed, vocab_size) # linear layer to project the embedding to the vocabulary size

    def forward(self, idx, targets=None):
        B, T = idx.shape # B: batch size, T: block size
        
        # 1) Predictions: idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch, time, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (time, n_embed)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.ln_head(x) # (batch, time, vocab_size)

        # 2) Evaluate loss
        if targets is None:
            loss = None
        else:
            # need to reshape to fix Pytorch's expectation of the shape of the logits
            B, T, C = logits.shape
            logits = logits.view(B * T, C) 
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) # we have the identity of the next character, how well are the logits predicting it?
        
        return logits, loss
    
    """
    For each sequence in the batch, we will predict and concatenate the next character 
    along the Time dimension until max_new_tokens.
    """
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 1) get the predictions
            logits, loss = self(idx)

            # 2) focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # 3) apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # 4) sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 5) append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

#---OPTIMIZER---
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#---TRAINING LOOP---
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Sample from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))