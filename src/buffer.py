import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, observation_shape, action_size, capacity, device):
        self.device = device
        self.capacity = int(capacity)

        self.observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

        self._index = 0
        self._full = False

    def __len__(self):
        return self.capacity if self._full else self._index

    def add(self, observation, action, reward, done):
        self.observations[self._index] = observation
        self.actions[self._index] = action
        self.rewards[self._index] = reward
        self.dones[self._index] = done

        self._index = (self._index + 1) % self.capacity
        self._full = self._full or self._index == 0

    def sample(self, batch_size, sequence_length):
        last_filled_index = self._index - sequence_length + 1
        assert self._full or (last_filled_index > batch_size), "not enough data in the buffer to sample"

        max_index = self.capacity if self._full else last_filled_index
        sample_index = np.random.randint(0, max_index, batch_size).reshape(-1, 1)
        offsets = np.arange(sequence_length).reshape(1, -1)
        indices = (sample_index + offsets) % self.capacity

        observations = torch.as_tensor(self.observations[indices], device=self.device).float()
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        is_first = torch.zeros_like(dones, dtype=torch.bool)
        is_first[:, 0] = True
        is_first[:, 1:] = dones[:, :-1].bool()

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "is_first": is_first,
        }
