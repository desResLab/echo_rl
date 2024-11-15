from collections import deque
MEMORY_SIZE = 100000
class Memory():
    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """

    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, terminal):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, terminal))