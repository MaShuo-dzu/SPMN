import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

class DynamicMemory(nn.Module):
    def __init__(self, hidden_size=100, memory_slots=20, keys=None):
        super(DynamicMemory, self).__init__()
        self.hidden_size = hidden_size

        self.keys = Parameter(nn.init.normal_(torch.empty(memory_slots, hidden_size), mean=0, std=0.1)) # M x H
        # Matrices for candidate
        self.U = Parameter(nn.init.normal_(torch.empty(hidden_size, hidden_size), mean=0, std=0.1))
        self.W = Parameter(nn.init.normal_(torch.empty(hidden_size, hidden_size), mean=0, std=0.1))
        self.V = Parameter(nn.init.normal_(torch.empty(hidden_size, hidden_size), mean=0, std=0.1))

        # PRelu initialization
        self.activation = nn.PReLU(num_parameters=hidden_size, init=1.0)

    def init_memory(self, batch_size):
        """Initialize the content memory to keys."""
        init_memory = self.keys.repeat(batch_size, 1, 1)  # B x M x H
        return init_memory

    def calc_gate(self, batch_sent, memory):
        """Calculate gate scalars.
        batch_sent: B x H
        memory: M x B x H
        """
        memory_term = torch.sum(memory * batch_sent, 2)  # M x B
        key_term = torch.mm(self.keys, torch.transpose(batch_sent, 1, 0))  # M x B
        return torch.sigmoid(memory_term + key_term)  # M x B

    def calc_candidate(self, batch_sent, memory):
        """Calculate the candidate term.
        batch_sent: B x H
        memory: M x B x H

        Return:- candidate: M x B x H
        """
        memory_slots, batch_size, hidden_size = list(memory.size())
        memory_resh = torch.reshape(memory, (-1, hidden_size))  # (M x B) x H
        memory_term = torch.mm(memory_resh, self.U)  # (M x B) x H
        memory_term = torch.reshape(memory_term, (memory_slots, batch_size, hidden_size))  # M x B x H

        key_term = torch.mm(self.keys, self.V)  # M x H
        key_term = torch.unsqueeze(key_term, dim=1) # M x 1 x H
        sent_term = torch.mm(batch_sent, self.W)  # B x H

        sum_term = memory_term + key_term + sent_term  # M x B x H
        sum_term = torch.reshape(sum_term, (-1, hidden_size))
        activated_inp =  self.activation(sum_term)
        return torch.reshape(activated_inp, (memory_slots, batch_size, hidden_size))

    def read_story(self, stories, mask):
        """Read story and return memories.
        stories: List of length T of B x H tensors
        mask: List of length T of tensors of size B

        Return:- memory: M x B x H
        """
        batch_size, _ = list(stories[0].size())

        # Initialize memory
        # memory_store = []
        gate_list = []
        memory = self.init_memory(batch_size)  # B x M x H
        # memory_store.append(memory)

        # Transpose memory and story to batch minor
        memory = torch.transpose(memory, 0, 1)  # M x B x H
        # print (mask)
        for batch_sent, cur_mask in zip(stories, mask):
            mask_sum = torch.sum(cur_mask).item()
            if mask_sum == 0:
                # All the entries in the current and following masks are zero
                break
            gate = self.calc_gate(batch_sent, memory)  # M x B
            gate_list.append(gate)
            masked_gate = gate * cur_mask  # M x B
            candidate = self.calc_candidate(batch_sent, memory) # M x B x H

            # Update memory after reading in ith sentence
            memory = memory + masked_gate.unsqueeze(dim=2) * candidate  # M x B x H
            # Normalize memory
            memory = nn.functional.normalize(memory, dim=2, p=2)

        # Transpose memory to make it batch major
        # batch_memory = torch.transpose(memory, 0, 1)  # B x M x H

        # return batch_memory
        return memory, gate_list
