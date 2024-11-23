import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class Agent:
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025,  # learning rate
                 gamma=0.9,   # discount factor
                 epsilon=1.00,  # initial epsilon
                 eps_decay=0.99999975,  # epsilon decay factor
                 eps_min=0.1,  # minimum epsilon value
                 replay_buffer_capacity=100_000,   # number of state action pair tuples that can be stored
                 batch_size=32,  # training batch size
                 sync_network_rate=10000   # number of episodes per updating network weight
                 ):
        
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Online and target networks for training
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Replay buffer with a capacity for 10,000 experiences
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

# epsilon greedy implementation of the function to choose an action
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()

    # reduces epsilon by our decay factor without going under the min value
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    # takes tensors we want to store in the replay buffer and organizes them in a dictionary
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))

    # Checks if enough learning steps have occurred
    # if so it will copy weight from target to online network
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    # saves the model to allow for the ability to pause training
    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    # loads a saved model to resume training
    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    # function to process the learning step
    def learn(self):
        # validate sufficient experiences in replay buffer to sample a batch from
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()

        # clear gradients
        self.optimizer.zero_grad()

        # sample from replay buffer and storing result is proper variable
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # Pass states tensor to online network to get predicted q values and index by actions taken
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]

        # Calculating target values from future states
        # the (1 - dones.float()) portion ensures that no future rewards are considered if the episode is done
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # Calculate loss and preform back propagation to calculate gradients
        # then preform a step of gradient decent with those values
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Increment learning counter and decay epsilon
        self.learn_step_counter += 1
        self.decay_epsilon()


        


