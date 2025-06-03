import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from memory import DynamicMemory


class EntNet(nn.Module):
    def __init__(self, vocab_size, memory_slots=20, emb_size=100, bow_encoding=False, max_sentence_length=50,
                 gate_penalty=0.0):
        super(EntNet, self).__init__()
        self.bow_encoding = bow_encoding
        self.memory_slots = memory_slots

        # Gate penalty
        self.gate_penalty = gate_penalty

        # Initialize embedding
        self.embedding = nn.Embedding(vocab_size, emb_size,
                                      _weight=nn.init.normal_(torch.empty(vocab_size, emb_size), std=0.1))

        # Initialize Sentence Encoder Parameters with all ones (BoW)
        init_weight = torch.ones(max_sentence_length, emb_size)
        if self.bow_encoding:
            self.mult_mask = nn.Embedding.from_pretrained(init_weight, freeze=True)
        else:
            self.mult_mask = nn.Embedding(max_sentence_length, emb_size, _weight=init_weight)

        # Initialize the memory network - emb_size == hidden_size
        self.memory_net = DynamicMemory(hidden_size=emb_size, memory_slots=memory_slots)

        # Output module parameters
        self.R = Parameter(nn.init.normal_(torch.empty(emb_size, vocab_size), mean=0, std=0.1))
        self.H = Parameter(nn.init.normal_(torch.empty(emb_size, emb_size), mean=0, std=0.1))
        self.activation = nn.PReLU(num_parameters=emb_size, init=1.0)

    def encode_sent(self, batch_sent):
        """
        Encode a batch of sentences.
        batch_sent: B x L
        """
        # Embed Sentence
        sent_tensor = self.embedding(batch_sent)  # B x L x E
        # Get non-zero indices (Will only select indices >= 1)
        mask_tensor = torch.unsqueeze(torch.ge(batch_sent, 1), dim=2).cuda()  # B x L x 1
        # Cast the mask_tensor
        mask_tensor = mask_tensor.type(dtype=torch.cuda.FloatTensor)
        # Mask out padded symbols
        sent_tensor = sent_tensor * mask_tensor  # B x L x E

        # Get embedding for multiplicative masks
        _, sentence_length = list(batch_sent.size())
        idx_tensor = torch.arange(sentence_length).cuda()
        mult_mask_tensor = self.mult_mask(idx_tensor)  # L x E

        batch_sent_emb = sent_tensor * mult_mask_tensor  # B x L x E
        batch_sent_emb = torch.sum(batch_sent_emb, 1)  # B x E

        return batch_sent_emb

    def encode_story(self, batch_story):
        """Encode a batch of stories.
        batch_story: B x T x L (T is # of sentences and L is max length of each sentence)

        Return:-
        enc_stories: list of B x H tensors with length T.
        mask_stories: list of length T with tensors of size (B,)
        """
        batch_story = torch.transpose(batch_story, 0, 1)  # T x B x L

        # Split stories along sentences
        seq_batch_sent = torch.unbind(batch_story)  # Tuple of B x L tensors
        enc_stories = []
        mask_stories = []
        for batch_sent in seq_batch_sent:
            enc_stories.append(self.encode_sent(batch_sent))
            # Check if all the symbols are pad or not
            mask_at_t = torch.ge(torch.sum(batch_sent, dim=1), 1.0).type(dtype=torch.cuda.FloatTensor)
            mask_stories.append(mask_at_t)
        return enc_stories, mask_stories

    def read_story_and_answer_question(self, batch_story, batch_question):
        """Read stories via memory network and answer related questions.
        batch_story: B x T x L (T is # of sentences and L is sentence length)
        batch_question: B x L

        Return:- answer
        """
        enc_stories, mask_stories = self.encode_story(batch_story)
        enc_query = self.encode_sent(batch_question)  # B x H

        memorized_stories, gate_list = self.memory_net.read_story(enc_stories, mask_stories)  # M x B x H

        # Get softmax scores for memories
        memory_scores = torch.sum(memorized_stories * enc_query, dim=2)  # M x B
        softmax_scores = nn.functional.softmax(memory_scores, dim=0)  # M x B

        # Get story representation corresponding to query
        softmax_scores = torch.unsqueeze(softmax_scores, 2)  # M x B x 1
        weighted_memories = softmax_scores * memorized_stories  # M x B x H
        story_repr = torch.sum(weighted_memories, dim=0)  # B x H

        # Output scores
        activation_input = enc_query + torch.mm(story_repr, self.H)  # B x H
        pred = torch.mm(self.activation(activation_input), self.R)  # B x V

        return pred, gate_list

    def get_loss(self, preds, target, gate_list):
        """Return the loss function given the prediction and correct labels.
        preds: B x V
        target: B (integer valued)
        """
        loss = nn.functional.cross_entropy(input=preds, target=target)

        if self.gate_penalty:
            gate_cat = torch.cat(gate_list, 1)  # M x (B x T)
            num_entries = gate_cat.size()[1]  # [(B x T)]

            gate_cov = torch.mm(gate_cat, torch.transpose(gate_cat, 0, 1))  # M x M
            # Remove diagonal entries
            mask = torch.eye(self.memory_slots).byte().cuda()
            gate_cov.masked_fill_(mask, 0)
            inter_memory_interaction = torch.sum(torch.abs(gate_cov))

            mean_interaction = inter_memory_interaction / num_entries
            loss = loss + self.gate_penalty * mean_interaction
        return loss

    def update_gate_penalty(self, gamma=0.5):
        """Exponential gate updation."""
        self.gate_penalty = self.gate_penalty * gamma
