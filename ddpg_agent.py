import numpy as np
import progressbar as pb
import random
import copy
from collections import namedtuple, deque
import time
from utilities import list_to_tensor, transpose_list

from model import Actor, Critic
from colorama import Fore, Back, Style

import torch
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Conversion from numpy to tensor
def ten(x): return torch.from_numpy(x).float().to(device)

# empty class to use like Matlab struct
class struct_class(): pass
    
class ddpg_agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, config):
        """Initialize an Agent object.
        
        Params
        ======
            env : environment to be handled
            config : configuration given a variety of parameters
        """
        self.env = env
        self.config = config

        # set parameter for ML
        self.set_parameters(config)
        # Q-Network
        self.create_networks()
        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
    
    def set_parameters(self, config):
        # Base agent parameters
        self.gamma = config['gamma']                    # discount factor 
        self.tau = config['tau']
        self.max_episodes = config['max_episodes']      # max numbers of episdoes to train
        self.env_file_name = config['env_file_name']    # name and path for env app
        self.brain_name = config['brain_name']          # name for env brain used in step
        self.train_mode = config['train_mode']
        self.num_agents = config['num_agents']
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.hidden_size = config['hidden_size']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.learn_every = config['learn_every']
        self.critic_learning_rate = config['critic_learning_rate']
        self.actor_learning_rate = config['actor_learning_rate']
        self.noise_decay = config['noise_decay']
        self.seed = (config['seed'])
        self.noise_scale = 1
        self.results = struct_class()
        # Some debug flags
        self.Do_debug_values = False
        self.Do_debug_qr = True
        
    def create_networks(self):
        # Actor Network (local & Target Network)
        self.actor_local = Actor(self.state_size, self.hidden_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.hidden_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic Network (local & Target Network)
        self.critic_local = Critic(self.state_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.critic_target = Critic(self.state_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # print('step : Next States : ',next_state.shape)
        self.memory.add(states, actions, rewards, next_states, dones)
        # print('New step added to memory, length : ',len(self.memory))
            
    # def update_noise_scale(self, cur_reward, scale_min = 0.2, scale_noise=False):
    #     """ If scale_noise == True the self.noise_scale will be decreased in relation to rewards
    #         Currently hand coded  as rewards go up noise_scale will go down from 1 to scale_min"""
        
    #     if scale_noise:
    #         rewlow = 2 # below rewlow noise_scale is 1 from there on it increases linearly down to scale_min + 0.5*(1 - scale_min) until rewhigh is reached
    #         rewhigh = 10 # above rewhigh noise_scale falls linearly down to scale_min until rewrd = 30 is reached. Beyond 30 it stays at scale_min
    #         if cur_reward > rewlow:
    #             if cur_reward < rewhigh:
    #                 self.noise_scale = (1 - scale_min)*(0.5*(rewhigh-cur_reward)/(rewhigh - rewlow) + 0.5) + scale_min
    #             else:
    #                 self.noise_scale = (1 - scale_min)*np.min(0.5*(30-cur_reward)/((30-rewhigh)),0) + scale_min
                    
    #         print('Updated noise scale to : ',self.noise_scale)
                
    #     return                    
        

    def act_by_agent(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = ten(state)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_scale * self.noise.sample()
        print(type(flat_action))
        action = np.clip(flat_action, -1, 1)
        return action

    def act(self, state, add_noise=True): # act_complete
        """Returns actions for given state as per current policy."""
        state = ten(state)
        self.actor_local.eval()
        with torch.no_grad():
            flat_action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            flat_action += self.noise_scale * self.noise.sample()
        flat_action = np.clip(flat_action, -1, 1)
        action =np.reshape(flat_action,[2,2])
        return action
    
    def init_results(self):
        """ Keeping different results in list in self.results, being initializd here"""
        self.results.reward_window = deque(maxlen=100)
        self.results.all_rewards = []
        self.results.avg_rewards = []
        self.results.critic_loss = []
        self.results.actor_loss = []

    def episode_reset(self):
        self.noise_reset()
        self.noise_scale *= self.noise_decay

    def noise_reset(self):
        # for a_i in range(self.num_agents):
        self.noise.reset()

    def train(self):
        print('Running on device : ',device)
        # if False:
        #     filename = 'trained_reacher_a_e100.pth'
        #     self.load_agent(filename)
        self.init_results()

        # training loop
        # show progressbar
        widget = ['episode: ', pb.Counter(),'/',str(self.max_episodes),' ', 
                  pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
        
        timer = pb.ProgressBar(widgets=widget, maxval=self.max_episodes).start()

        for i_episode in range(self.max_episodes): 
            timer.update(i_episode)
            tic = time.time()
            self.episode = i_episode

            # per episode resets
            self.episode_reset()
            total_reward = np.zeros(self.num_agents)
            # Reset the enviroment
            env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
            states = env_info.vector_observations
            t = 0
            dones = np.zeros(self.num_agents, dtype = bool)

            # loop over episode time steps
            while not any(dones):
                # act and collect data
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = (env_info.local_done)
                # increment stuff
                t += 1
                total_reward += rewards
                # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                # print('Episode {} step {} taken action {} reward {} and done is {}'.format(i_episode,t,actions,rewards,dones))
                # Proceed agent step
                self.step(states, actions, rewards, next_states, dones)
                # prepare for next round
                states = next_states
            # while not done
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size and (i_episode % self.learn_every == 0): # 
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            toc = time.time()
            # keep track of rewards:
            self.results.all_rewards.append(total_reward)
            self.results.avg_rewards.append(np.mean(self.results.reward_window))
            self.results.reward_window.append(np.max(total_reward))
            
            # Output Episode info : 
            if (i_episode % 20 == 0) or (np.max(total_reward) > 0.01):
                if np.max(total_reward) > 0.01:
                    StyleString = Back.RED
                else:
                    StyleString = ''
                print(StyleString + 'Episode {} || Reward : {} || avg reward : {:6.3f} || Noise {:6.3f} || {:5.3f} seconds, mem : {}'.format(i_episode,total_reward,np.mean(self.results.reward_window),self.noise_scale,toc-tic,len(self.memory)))
                print(Style.RESET_ALL, end='')                
        # for i_episode
            
        return self.results

    # def stable_update(self):
    #     """ Update Hyperparameters which proved more stable """
    #     self.buffer_size = 400000
    #     self.memory.enlarge(self.buffer_size)
    #     self.noise_sigma = 0.05
    #     self.noise.sigma = 0.05
    
    def unzip_experiences(self, experiences):
        """ experiences is list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
            returns states... as tensors(shape[num_agents, batch_size, item_size(e.g. state_size)])"""
        texperiences = transpose_list(experiences)

        states = torch.stack([ten(np.vstack([e.state for e in te])) for te in texperiences], dim=0)
        actions = torch.stack([ten(np.vstack([e.action for e in te])) for te in texperiences], dim=0)
        rewards = torch.stack([ten(np.vstack([np.asarray(e.reward) for e in te])) for te in texperiences], dim=0)
        next_states = torch.stack([ten(np.vstack([e.next_state for e in te])) for te in texperiences], dim=0)
        dones = torch.stack([ten(np.vstack([np.asarray(e.done) for e in te])) for te in texperiences], dim=0)
    
        return  states, actions, rewards, next_states, dones

    def experiences_by_agent(self, agent_num, states, actions, rewards, next_states, dones):
        
        states = states[agent_num,:,:]
        actions = actions[agent_num,:,:]
        rewards = rewards[agent_num,:,:]
        next_states = next_states[agent_num,:,:]
        dones = dones[agent_num,:,:]
        
        return  states, actions, rewards, next_states, dones
            
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        q_target = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            self.gamma (float): discount factor
        """
        # epxereinces is list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
        agent_num = 0
        states, actions, rewards, next_states, dones = self.unzip_experiences(experiences)
        states, actions, rewards, next_states, dones = self.experiences_by_agent(agent_num,states, actions, rewards, next_states, dones)
        # print('Learning shape : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # print('learn : Next States : ',next_states.shape)
        actions_next = self.actor_target(next_states)
        q_next = self.critic_target(next_states, actions_next)
        # print('learn : Actions : ',actions_next.shape)
        # print('learn : Q_target_next : ',q_next.shape)
        # Compute Q targets for current states (y_i)
        q_target = rewards + (self.gamma * q_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)

        td_error = q_target - q_expected
        # if max_reward > 0.01:
        self.debug_values(q_next, q_target, q_expected, td_error, rewards, dones)
        self.debug_update_qr(agent_num, q_expected, rewards,td_error)

        critic_loss = F.mse_loss(q_expected, q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        
        self.results.actor_loss.append(al)
        self.results.critic_loss.append(cl)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    # Some debug functions
    def debug_values(self, q_next, q_target, q_expected, td_error, rewards, dones):
        if self.Do_debug_values and (self.episode % 20 == 0):
            print('Found reward :')
            # print(type(q_next))
            # print(type(q_target))
            # print(type(q_expected))
            # print(q_next.shape)
            # print(q_target.shape)
            # print(q_expected.shape)
            # print(td_error.shape)
            # print(rewards.shape)
            # print(dones.shape)
            torch.set_printoptions(precision=4,threshold=10_000, sci_mode = False, linewidth = 120)
            print('q_next, q_target, q_expected, td_error')
            qlist = torch.cat([q_next,q_target,q_expected,td_error,rewards,dones],1)
            print(qlist)
        
    def debug_update_qr(self, agent_number, q_expected, rewards,td_error):
        if self.Do_debug_qr and (self.episode % 10 == 0) or (torch.max(rewards) > 0.01):
            print('--------------------------------------')
            print('Agent {} and episode {} '.format(agent_number, self.episode))
            print('update - q expected : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(q_expected),torch.std(q_expected),torch.min(q_expected),torch.max(q_expected)))
            if torch.max(rewards) > 0.01:
                print(Back.GREEN+'update - reward : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(rewards),torch.std(rewards),torch.min(rewards),torch.max(rewards)))
                print(Style.RESET_ALL, end='')                
            else:
                print('update - reward : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(rewards),torch.std(rewards),torch.min(rewards),torch.max(rewards)))
                
            print('update - TD-Error : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(td_error),torch.std(td_error),torch.min(td_error),torch.max(td_error)))

            
    def save_agent(self,i_episode):
        filename = 'trained_reacher_e'+str(i_episode)+'.pth'
        torch.save({
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            }, filename)
        
        print('Saved Networks in ',filename)
        return
        
    def load_agent(self,filename):
        savedata = torch.load(filename)
        self.critic_local.load_state_dict(savedata['critic_local'])
        self.critic_target.load_state_dict(savedata['critic_target'])
        self.actor_local.load_state_dict(savedata['actor_local'])
        self.actor_target.load_state_dict(savedata['actor_target'])
        return

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            self.batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
            
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        # print('Adding experiences to memory : ')
        # print('State shape : ',states.shape)
        # print(type(states))
        new_experience = []
        # new_experience_set = [self.experience() for e in np.arange(states.shape[0])]
        for a in range(states.shape[0]): # a as num_agents
            e = self.experience(states[a,:], actions[a,:], rewards[a], next_states[a,:], dones[a])
            new_experience.append(e)
        # print('New Experience : ',new_experience)
        self.memory.append(new_experience)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)