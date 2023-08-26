import os
import torch
from torch.utils.data import Dataset


class ReplayMemory(Dataset):
    def __init__(self, capacity, output_dir):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        self.output_dir = output_dir
        self.num_episode = 0
        self.num_episode_resume = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

    def save(self, file_name, num_episode):
        memory_path = os.path.join(self.output_dir, file_name)
        state = {'memory':self.memory, 'num_episode': num_episode, 'position':self.position}
        torch.save(state, memory_path)

    def load(self, file_name):
        memory_file = os.path.join(self.output_dir, file_name)
        if os.path.exists(memory_file):
            state = torch.load(memory_file)
            self.memory = state['memory']
            self.position = state['position']
            self.num_episode = state['num_episode']
            self.num_episode_resume = state['num_episode']
            return self.num_episode
        else:
            return 0