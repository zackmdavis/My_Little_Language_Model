# following along with https://www.youtube.com/watch?v=kCc8FmEb1nY to start
# (but making my own idiosyncratic changes for flavor, of course)

import torch
import torch.nn as nn

with open("corpus.txt") as f:
    corpus = f.read()

# exclude a few characters that are too weird, even for me
exclusions = ['\u200a', '\u200b', '\u202d', 'ا', 'ر', 'غ', 'ف', 'م', 'ن', 'ي']

# length 155: >2.3 times Karpathy's, because I use special characters
chars = sorted(c for c in list(set(corpus)) if c not in exclusions)
vocab_size = len(chars)

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(string):
    output = []
    for c in string:
        if c not in exclusions:
            output.append(char_to_int[c])
    return output


def decode(tokens):
    return ''.join(int_to_char[t] for t in tokens)


data = torch.tensor(encode(corpus), dtype=torch.long)
print(data.shape, data.dtype)
print(data)

cutoff = int(0.92*len(data))
training_set = data[:cutoff]
validation_set = data[cutoff:]

batch_size = 5
block_size = 10
embedding_dimensions = 32
dropout = 0.2

def get_batch(dataset):
    indices = torch.randint(high=len(dataset) - block_size, size=(batch_size,))
    inputs = torch.stack([dataset[i:i+block_size] for i in indices])
    targets = torch.stack([dataset[i+1:i+block_size+1] for i in indices])
    return inputs, targets



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(embedding_dimensions, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # TODO—rewrite this line-by-line to check understanding

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dimensions)
        self.self_attention_head = Head(embedding_dimensions)
        self.head = nn.Linear(embedding_dimensions, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        token_embeddings = self.token_embedding_table(inputs)
        position_embeddings = self.position_embedding_table(torch.arange(T))
        x = token_embeddings + position_embeddings
        x = self.self_attention_head(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            # What are B, T, and C?! I don't like imitating opaque magic—"batch, time, channels"
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_length):
        for _ in range(max_length):
            inputs_cropped = inputs[:, -block_size:]
            logits, loss = self(inputs_cropped)
            logits = logits[:, -1, :]
            probabilities = nn.functional.softmax(logits, dim=1)
            following = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat((inputs, following), dim=1)
        return inputs


inputs, targets = get_batch(training_set)
m = LanguageModel(vocab_size)
logits, loss = m(inputs, targets)
print(logits.shape)
print(loss)

print(decode(m.generate(inputs = torch.zeros((1, 1), dtype=torch.long), max_length=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    inputs, targets = get_batch(training_set)
    logits, loss = m(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

print(decode(m.generate(inputs = torch.zeros((1, 1), dtype=torch.long), max_length=100)[0].tolist()))
