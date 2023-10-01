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

def get_batch(dataset):
    indices = torch.randint(high=len(dataset) - block_size, size=(batch_size,))
    inputs = torch.stack([dataset[i:i+block_size] for i in indices])
    targets = torch.stack([dataset[i+1:i+block_size+1] for i in indices])
    return inputs, targets


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs)
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
            logits, loss = self(inputs)
            logits = logits[:, -1, :]
            probabilities = nn.functional.softmax(logits, dim=1)
            following = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat((inputs, following), dim=1)
        return inputs


inputs, targets = get_batch(training_set)
m = BigramLanguageModel(vocab_size)
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

# up to 42:20 in the video
