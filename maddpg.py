import numpy as np
import progressbar as pb
from collections import namedtuple, deque
from utilities import transpose_list
import time

from colorama import Back, Style
import torch
from ddpg_agent import ddpg_agent
from BufferPER import ReplayBuffer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Conversion from numpy to tensor
def ten(x): return torch.from_numpy(x).float().to(device)

# empty class to use like Matlab struct
class struct_class(): pass
    
class maddpg():
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
        # Replay memory
        self.memory = ReplayBuffer(config)
        # Q-Network
        self.create_agents(config)
    
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
        self.emin = 0.0001
        config['emin'] = self.emin
        self.min_batch_size = 256                      # if batch_size is large, this will be used as batchsize for early samples until buffer is filled with > batchsize experiences
        self.use_PER = True
        self.dropout = config['dropout']
        self.learn_every = config['learn_every']
        self.joined_states = config['joined_states']
        self.critic_learning_rate = config['critic_learning_rate']
        self.actor_learning_rate = config['actor_learning_rate']
        self.noise_decay = config['noise_decay']
        self.seed = (config['seed'])
        self.noise_scale = 1
        self.results = struct_class()
        # Some Debug flags
        self.debug_show_memory_summary = False
        
    def create_agents(self, config):
        self.maddpg_agent = [ddpg_agent(config), 
                             ddpg_agent(config)]
        
        for a_i in range(self.num_agents):
            self.maddpg_agent[a_i].id = a_i
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)
                    

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy 
        shuold only get single or single joined states from train"""
        state = ten(state)
        if self.joined_states:
            joined_state = torch.reshape(state,[48])
            actions = np.vstack([agent.act(joined_state) for agent in self.maddpg_agent])
        else:
            actions = np.vstack([agent.act(state[a_i,:]) for agent,a_i in zip(self.maddpg_agent, range(self.num_agents))])
        return actions
    
    def actor_target(self, states):
        """Returns actions for given state as per current target_policy without noise.
           should only get batch_size states from learn"""
        if self.joined_states:
            actions = np.hstack([agent.act(states) for agent in self.maddpg_agent])
        else:
            actions = np.vstack([agent.act(states[a_i,:]) for agent,a_i in zip(self.maddpg_agent, range(self.num_agents))])
        return ten(actions)

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
        for agent in self.maddpg_agent:
            agent.noise_scale = self.noise_scale
            agent.episode = self.episode
        
    def noise_reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset() 

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
            if (i_episode % self.learn_every == 0):
                if len(self.memory) > self.batch_size: # 
                    experiences, exp_indices, probs = self.memory.sample()
                    self.learn(experiences,exp_indices, probs)
                elif len(self.memory) > self.min_batch_size: # while few experiences draw ajust min_batch_size
                    print('Until buffer filled batches are smaller ({} vs. later {})'.format(self.min_batch_size, self.batch_size))
                    self.memory.batch_size = self.min_batch_size
                    experiences, exp_indices, probs = self.memory.sample()
                    self.learn(experiences,exp_indices, probs)
                    self.memory.batch_size = self.batch_size
                    

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
                print(StyleString + 'Episode {} with {} steps || Reward : {} || avg reward : {:6.3f} || Noise {:6.3f} || {:5.3f} seconds, mem : {}'.format(i_episode,t,total_reward,np.mean(self.results.reward_window),self.noise_scale,toc-tic,len(self.memory)))
                print(Style.RESET_ALL, end='')                
                if self.debug_show_memory_summary:self.memory.mem_print_summary()
        # for i_episode
            
        return self.results
    
    def unzip_experiences(self, experiences):
        """ experiences is list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
            returns states... as tensors(shape[num_agents, batch_size, item_size(e.g. state_size)])"""

        if self.joined_states:
            states = ten(np.vstack([np.concatenate([e.state for e in batch],axis=0) for batch in experiences]))
            actions = ten(np.vstack([np.concatenate([e.action for e in batch],axis=0) for batch in experiences]))
            rewards = ten(np.vstack([np.hstack([e.reward for e in batch]) for batch in experiences]))
            next_states = ten(np.vstack([np.concatenate([e.next_state for e in batch],axis=0) for batch in experiences]))
            dones = ten(np.vstack([sum([int(e.done) for e in batch]) for batch in experiences]))
        else:
            texperiences = transpose_list(experiences)
            states = torch.stack([ten(np.vstack([e.state for e in te])) for te in texperiences], dim=0)
            actions = torch.stack([ten(np.vstack([e.action for e in te])) for te in texperiences], dim=0)
            rewards = torch.stack([ten(np.vstack([np.asarray(e.reward) for e in te])) for te in texperiences], dim=0)
            next_states = torch.stack([ten(np.vstack([e.next_state for e in te])) for te in texperiences], dim=0)
            dones = torch.stack([ten(np.vstack([np.asarray(e.done) for e in te])) for te in texperiences], dim=0)
    
        # print('unzip shape : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        return  states, actions, rewards, next_states, dones

    def ununzip_experiences(self, experiences):
        """ experiences from PER is list(len=num_agents) list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
            as both agent do not use the same experiences anymore, as their deltaQ in the buffer are different
            returns list(len=num_agents) of states... as tensors(shape[num_agents, batch_size, item_size(e.g. state_size)])"""
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []
        for a in range(self.num_agents):
            states, actions, rewards, next_states, dones = self.unzip_experiences(experiences[a])
            state_list.append(states)
            action_list.append(actions)
            reward_list.append(rewards)
            next_state_list.append(next_states)
            done_list.append(dones)
        return state_list, action_list, reward_list, next_state_list, done_list
    
    def experiences_by_agent(self, agent_num, states, actions, rewards, next_states, both_next_actions, dones):
        
        states = states[agent_num,:,:]
        actions = actions[agent_num,:,:]
        rewards = rewards[agent_num,:,:]
        next_states = next_states[agent_num,:,:]
        dones = dones[agent_num,:,:]
        
        print('Learn : ',both_next_actions)
        next_actions = both_actions_next[agent.id]
        print(next_actions)

        return  states, actions, rewards, next_states, next_actions, dones
            
    def learn(self, experiences, exp_indices, probs):
        """Update policy and value parameters using given batch of experience tuples.
        q_target = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value """

        # epxeriences is list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
        # for PER it is list(len=num_agents) list(len=num_agents) of list(len=batch_size) of experiences(namedtuple of states, actions, rewards, next_states, dones)
        if self.use_PER:
            # ununzip will unzip both lists, return will be list(len_num_agents) of the second version
            states, actions, rewards, next_states, dones = self.ununzip_experiences(experiences)
        else:
            # without PER there is only one list to unzip
            states, actions, rewards, next_states, dones = self.unzip_experiences(experiences)
        # print('Learning shape : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        
        actor_loss = []
        critic_loss = []
        if not self.use_PER:
            both_next_actions = self.actor_target(next_states)
        else:
           both_next_actions = [self.actor_target(next_states[a]) for a in range(self.num_agents)]
            
        # print('Learn both',both_next_actions.shape)
        for agent in self.maddpg_agent:
            # In case of joined_states, we want actions_next from both agents for learning
            if self.joined_states:
                if not self.use_PER:
                    al, cl = agent.learn(states, actions, rewards[:,agent.id].unsqueeze(1), next_states, both_next_actions, dones)
                else:
                    al, cl, NewQ = agent.learn(states[agent.id], actions[agent.id], rewards[agent.id][:,agent.id].unsqueeze(1), next_states[agent.id], both_next_actions[agent.id], dones[agent.id], probs[agent.id], len(self.memory))
                    self.memory.updatedeltaQ(NewQ, exp_indices[agent.id], agent.id)
            else:
                al, cl = agent.learn(*(self.experiences_by_agent(agent.id,states, actions, rewards, next_states, both_next_actions, dones)))
            actor_loss.append(al)
            critic_loss.append(cl)
            
        self.results.actor_loss.append(np.mean(actor_loss))
        self.results.critic_loss.append(np.mean(critic_loss))

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
