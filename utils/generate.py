import torch
import torch.nn.functional as F
import numpy as np


def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return probs


def top_k_sampling(logits, k=50):
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    return probs, indices


def top_p_sampling(logits, p=0.95):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    cutoff_index = torch.where(cumulative_probs >= p)[0][0].item() + 1
    selected_logits = sorted_logits[:cutoff_index]
    selected_indices = sorted_indices[:cutoff_index]

    probs = F.softmax(selected_logits, dim=-1)
    return probs, selected_indices


def beam_search_generate(model, start_token, max_length, vocab_size, num_beams=3, eos_token=None, temperature=1.0,
                         top_k=50, top_p=0.95):
    beams = [(torch.tensor([start_token]).long(), 0)]
    for step in range(max_length):
        new_beams = []
        for seq, score in beams:
            if eos_token is not None and seq[-1] == eos_token:
                new_beams.append((seq, score))
                continue

            logits = model(seq.unsqueeze(0))
            logits = logits.squeeze(0)

            logits = temperature_sampling(logits, temperature)

            if top_k > 0:
                probs, top_indices = top_k_sampling(logits, top_k)
            elif top_p < 1.0:
                probs, top_indices = top_p_sampling(logits, top_p)
            else:
                probs = F.softmax(logits, dim=-1)
                top_indices = torch.topk(probs, num_beams).indices

            for idx in range(len(probs)):
                new_seq = torch.cat([seq, top_indices[idx].unsqueeze(0)])
                new_score = score + torch.log(probs[idx])
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]

        if eos_token is not None and all(seq[-1] == eos_token for seq, _ in beams):
            break

    return beams[0][0]


class SimpleLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLanguageModel, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.LSTM(embed_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embedded = self.embeddings(input_seq)
        rnn_out, _ = self.rnn(embedded.unsqueeze(0))
        logits = self.fc(rnn_out.squeeze(0))
        return logits


if __name__ == "__main__":
    vocab_size = 10
    embed_size = 16
    hidden_size = 32
    start_token = 0
    eos_token = 9
    max_length = 20

    model = SimpleLanguageModel(vocab_size, embed_size, hidden_size)
    model.eval()

    generated_seq = beam_search_generate(
        model=model,
        start_token=start_token,
        max_length=max_length,
        vocab_size=vocab_size,
        num_beams=3,
        eos_token=eos_token,
        temperature=1.0,
        top_k=5,
        top_p=0.95
    )

    print(generated_seq)
