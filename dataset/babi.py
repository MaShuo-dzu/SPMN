import torch
from torchtext.datasets import BABI20


def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab


train_iter, valid_iter, test_iter, vocab = dataloader(2, 50, 1, True, False)

for _, batch in enumerate(train_iter, start=1):
    story = batch.story
    for sentence in story[0]:
        for token in sentence:
            print(vocab.itos[token], end=" ")
        print("")

    query = batch.query
    for token in query[0]:
        print(vocab.itos[token], end=" ")
    print("")

    answer = batch.answer
    for token in answer[0]:
        print(vocab.itos[token], end="")

    # print(story.shape, query.shape, answer.shape)
    input()
