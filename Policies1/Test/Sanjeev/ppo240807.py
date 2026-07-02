import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import gym
import numpy as np
import matplotlib.pyplot as plt

# #class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(PolicyNetwork, self).__init__()
        
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.mean = nn.Linear(64, output_dim)
#         self.log_std = nn.Linear(64, output_dim)
#         self.initialize_weights()

#     def initialize_weights(self):
#         nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.mean.weight, nonlinearity='linear')
#         nn.init.kaiming_normal_(self.log_std.weight, nonlinearity='linear')
#         nn.init.constant_(self.fc1.bias, 0.0)
#         nn.init.constant_(self.fc2.bias, 0.0)
#         nn.init.constant_(self.mean.bias, 0.0)
#         nn.init.constant_(self.log_std.bias, 0.0)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         mean = self.mean(x)
#         log_std = self.log_std(x)
#         log_std = torch.clamp(log_std, -2, 2)
#         #print(f'log,{log_std}')
#         std = torch.exp(log_std)
#         #print(mean,std)
#         return mean, std

# class ValueNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.value = nn.Linear(64, 1)
#         self.initialize_weights()

#     def initialize_weights(self):
#         nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.value.weight, nonlinearity='linear')
#         nn.init.constant_(self.fc1.bias, 0.0)
#         nn.init.constant_(self.fc2.bias, 0.0)
#         nn.init.constant_(self.value.bias, 0.0)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         value = self.value(x)
#         #print(f'value:{value}')
#         return value
HID_SIZE =64
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(input_dim, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, output_dim),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        mean = self.mu(x)
        log_std = self.logstd
        #log_std = torch.clamp(log_std, -2, 2)
        #print(f'log,{log_std}')
        std = torch.exp(log_std)
        #print(mean,std)
        return mean, std
        


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(input_dim, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)

class PPOAgent:
    def __init__(self, env, policy_lr=3e-4, value_lr=1e-3, gamma=0.99, clip_ratio=0.2, epochs=5):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        # Initialize lists for storing losses
        self.policy_losses = []
        self.value_losses = []

    def get_action(self, state):    
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #print(f'state,{state}')  # Add batch dimension
        mean, std = self.policy_net(state)
        #print(f'mean,std {mean,std}')
        #print(f"Mean: {mean.mean().item()}, Std: {std.mean().item()}")
        dist = Normal(mean, std)
        action = dist.sample()
        #print(f'Action from get sample, {action}')
        log_prob = dist.log_prob(action).sum(axis=-1)
        action = action.squeeze(0).detach().numpy()
        actionplot =action[1]
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high) 
        #print(f'Action from get clip, {action}') # Clip action
        return action, log_prob.squeeze(0).detach(),actionplot

    def normalize_observation(self, observation):
        return np.clip(observation, self.env.observation_space.low, self.env.observation_space.high)

    def train(self, num_episodes=500, batch_size=64):
        episode_rewards = []
        n=0
        for episode in range(num_episodes):
            n+=1
            state = self.env.reset()
            state = self.normalize_observation(state)  # Ensure initial state is normalized
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], [] #clear memory
            total_reward = 0.0
            while True:
                action, log_prob,actionplot = self.get_action(state)
                
                #print(f'Action from get action, {action}')
                results = self.env.step(action)
                
                if len(results) == 4:
                    next_state, reward, done, info = results
                    truncated = False  # Assume not truncated if only 'done' is provided
                else:
                    next_state, reward, done, truncated, info = results

                next_state = self.normalize_observation(next_state)  # Normalize observation
                
                #store observations
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done or truncated)
                values.append(self.value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())
                log_probs.append(log_prob.item())
                
                total_reward += reward
                if done or truncated:
                    episode_rewards.append(total_reward)
                    rewards = (rewards[:-1] - np.mean(rewards[:-1])) / (np.std(rewards[:-1]))#,unbiased=False)
                    next_value = self.value_net(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).item()
                    advantages = self.compute_advantages(rewards, values, next_value, dones)
                    
                    # Convert to tensors
                    returns = advantages + torch.tensor(values, dtype=torch.float32)
                    states = np.array(states)
                    states = torch.tensor(states, dtype=torch.float32)
                    #print(f'states:{states}')
                    actions =np.array(actions)
                    actions = torch.tensor(actions, dtype=torch.float32)
                    advantages = advantages#torch.tensor(advantages, dtype=torch.float32)
                    log_probs = torch.tensor(log_probs, dtype=torch.float32)
                    
                    # Ensure tensors have correct shape and type
                    #print(f"States: {states.shape}, Actions: {actions.shape}, Advantages: {advantages.shape}, Log_probs: {log_probs.shape}, Returns: {returns.shape}")
                    
                    policy_loss, value_loss = self.update_policy_value(states, actions, advantages, log_probs, returns)
                    self.policy_losses.append(policy_loss)
                    self.value_losses.append(value_loss)
                    episode_rewards2 =np.array(episode_rewards)
                    writer.add_scalar('Total Reward',total_reward,episode)
                    writer.add_scalar('Policy Loss',policy_loss,episode)
                    writer.add_scalar('Value Loss',value_loss,episode)
                    writer.add_scalar('actions',actionplot,(episode*10000))
                    break
                
                state = next_state

            #print(f"Episode {episode}: Total Reward = {total_reward}")

        self.plot_results(episode_rewards)
        
        #writer.add_scalar('Total Reward',episode_rewards,episode)

    def plot_results(self, episode_rewards):
        plt.figure(figsize=(12, 4))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(self.policy_losses, label='Policy Loss')
        plt.plot(self.value_losses, label='Value Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages.insert(0, gae)
            next_value = values[i]
        
        # Debugging the advantages list
        #print(f"Raw Advantages List: {advantages}")
        
        advantages = np.array(advantages)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std())
        
        return advantages

    def update_policy_value(self, states, actions, advantages, log_probs_old, returns):
        policy_loss = 0
        value_loss = 0

        sum_loss_policy = 0.0
        
        for _ in range(self.epochs):
            mean, std = self.policy_net(states)
            #print(f"Mean2: {mean.mean().item()}, Std2: {std.mean().item()}")
            #print(f'states2:{states}')
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(axis=-1)
            ratios = torch.exp(log_probs - log_probs_old)
            #print(f'ratios:{ratios}')
            #print(f"Ratios: Mean={ratios.mean().item()}, Min={ratios.min().item()}, Max={ratios.max().item()}")
            #print(f"log prob: {log_probs},log_prob_old:{log_probs_old}")
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            self.policy_optimizer.zero_grad()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss 
            self.value_optimizer.zero_grad() 
            values = self.value_net(states).squeeze(-1)
            value_loss = (returns - values).pow(2).mean()
            #print(f'return:{returns},value:{values}')
            #print(value_loss)
        
            # Update policy
            torch.autograd.set_detect_anomaly(True)
            
            policy_loss.backward(retain_graph=True) 
            self.policy_optimizer.step()

            # Update value network
                        
            value_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.value_optimizer.step()

            sum_loss_policy+=policy_loss
            #count_steps+=1
            #n+=1
        # writer = SummaryWriter()
            #writer.add_scalar(policy_loss,n)
      
        return policy_loss.item(), value_loss.item()

# Example usage
env = gym.make('gym_fracture:fracsurg-v0')
writer = SummaryWriter()
steps = 10000
n=0
agent = PPOAgent(env)
agent.train()
