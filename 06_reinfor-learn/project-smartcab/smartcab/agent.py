import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.prior_state = None
        self.prior_action = None
        self.action_list = [None] # used in Q learning algorithm
        self.reward_list = [0] # used in Q learning algorithm
        self.current_state = ('', '', '', '')
        self.reward_count = 0.0
        self.penalty_count = 0.0
        self.none_count = 0.0
        self.net_reward = 0.0

        self.states = {}
        waypoints = ['forward', 'left', 'right']
        trafficlight = ['red','green']
        oncoming = [None, 'left', 'right', 'forward']
        left = [None, 'left', 'right', 'forward']

        self.pct_stats = pd.DataFrame({'trial_num':[],'pct_reward':[],'pct_penalty':[],'pct_none':[],'pct_reward_or_none':[]})
        
        # Create dictionary of all possible states
        for w in waypoints:
            for t in trafficlight:
                for o in oncoming:
                    for l in left:
                        new_key = (w,t,o,l)
                        if w=='forward':
                            self.states[new_key] = {None:0.5, 'forward':0.5, 'right':0, 'left':0}
                        elif w=='right':
                            self.states[new_key] = {None:0.5, 'forward':0, 'right':0.5, 'left':0}
                        elif w=='left':
                            self.states[new_key] = {None:0.5, 'forward':0, 'right':0, 'left':0.5}
                        #self.states[new_key] = {None:0.0, 'forward':0.0, 'right':0.0, 'left':0.0}

        self.states[self.current_state] = {None:0.0, 'forward':0.0, 'right':0.0, 'left':0.0}
        self.action_count = 0.0
        self.stats = {'net_reward':[], 'reward_count':[], 'penalty_count':[], 'none_count':[], 'reward_or_none_count':[],'successes':[], 'total_trials':0}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.action_count = 0.0
        self.prior_state = None
        self.prior_action = None
        self.action_list = [None] # used in Q learning algorithm
        self.reward_list = [0] # used in Q learning algorithm
        self.reward_count = 0
        self.penalty_count = 0
        self.none_count = 0
        self.net_reward = 0
        self.stats['total_trials'] += 1
		
    def update(self, t):
        print_move = False

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.prior_state = tuple(self.current_state)
        self.current_state = (self.next_waypoint, inputs['light'],inputs['oncoming'],inputs['left'])
		
        prior_key = tuple(self.prior_state)
        current_key = tuple(self.current_state)
		
        prior_line = dict(self.states[prior_key])
        current_line = dict(self.states[current_key])
        current_line_str = str(current_line)
		
        current_values = current_line.values()
        
        # TODO: Select action according to your policy
        prior_reward = self.reward_list[len(self.reward_list)-1]
        prior_action = self.action_list[len(self.action_list)-1]
        prior_q = float(self.states[prior_key][prior_action])
        max_q_current_state = float(max(current_values))
		
        # TODO: Learn policy based on state, action, reward
        alpha = 1.0
        gamma = 1.0
        
        # ACTUAL Q LEARNING
        #updated_q_value = prior_q + alpha * (prior_reward + gamma * max_q_current_state - prior_q) 
        
		# AGENT LEARNS ON FIRST ENCOUNTER
        updated_q_value = prior_q + alpha * (prior_reward + gamma * prior_q - prior_q) # sub prior_q for max_q_current_state to do just rewards only q learning
        self.states[prior_key][prior_action] = updated_q_value # prior_q + alpha * (reward + gamma * prior_q - prior_q) 
		
        ### Implement a basic driving agent - random choice
        #action = random.choice([None,'forward','right','left']) # random action
		
        ### Select action with the largest Q-value
		# First check all the Q values for the given state
        best_action_value = max(current_values)
        num_best = sum(np.array(current_values) == best_action_value)
        
        # If there is a best Q value, choose that action
        if num_best==1:
            action = current_line.keys()[current_line.values().index(best_action_value)]
        # Otherwise randomly sample between actions with Q-values = max Q-value for that state, except exclude None
        elif num_best==2 or num_best==3:
            filt_dict = {}
            for k,v in current_line.iteritems():
                if v == best_action_value and k != None: # We never want to randomly choose None, it will provide no exploration
                    filt_dict[k] = v
                else:
                    continue
            possible_actions = filt_dict.keys()
            action = random.choice(possible_actions)
        else:
            action = random.choice(['forward', 'right', 'left']) # Again we never randomly choose None

        # Execute action and get reward
        self.action_list.append(action)
        reward = self.env.act(self, action)
        self.reward_list.append(reward)

        if reward>=2.0:
            self.reward_count += 1
        elif reward<0:
            self.penalty_count += 1
        elif reward==0:
            self.none_count += 1
        else:
            print "There's an error in the reward if/else, reward not in (-1.0, -0.5, 0, 2.0+), reward is: " + str(reward)

        self.net_reward += reward

        ### Update end of trip information
        if reward > 2.0:
            self.stats['net_reward'].append(self.net_reward)
            self.stats['reward_count'].append(self.reward_count)
            self.stats['penalty_count'].append(self.penalty_count)
            self.stats['none_count'].append(self.none_count)
            self.stats['reward_or_none_count'].append(self.reward_count+self.none_count)
            self.stats['successes'].append(1)
            # Add new row to pcts table
            trial_num = self.stats['total_trials']
            tot_actions = (self.reward_count + self.penalty_count + self.none_count)*1.0
            pct_reward = (self.reward_count * 1.0) / tot_actions
            pct_penalty = (self.penalty_count * 1.0) / tot_actions
            pct_none = (self.none_count * 1.0) / tot_actions
            pct_reward_or_none = (self.reward_count+self.none_count*1.0) / tot_actions
            new_row = {'trial_num':trial_num,'pct_reward':pct_reward, 'pct_penalty':pct_penalty,'pct_none':pct_none, 'pct_reward_or_none':pct_reward_or_none}
            self.pct_stats = self.pct_stats.append(new_row, ignore_index=True)
        elif deadline == 0:
            self.stats['net_reward'].append(self.net_reward)
            self.stats['reward_count'].append(self.reward_count)
            self.stats['penalty_count'].append(self.penalty_count)
            self.stats['none_count'].append(self.none_count)
            self.stats['reward_or_none_count'].append(self.reward_count+self.none_count)
            self.stats['successes'].append(0)
            # Add new row to pcts table
            trial_num = self.stats['total_trials']
            tot_actions = (self.reward_count + self.penalty_count + self.none_count)*1.0
            pct_reward = (self.reward_count * 1.0) / tot_actions
            pct_penalty = (self.penalty_count * 1.0) / tot_actions
            pct_none = (self.none_count * 1.0) / tot_actions
            pct_reward_or_none = (self.reward_count+self.none_count*1.0) / tot_actions
            new_row = {'trial_num':trial_num,'pct_reward':pct_reward, 'pct_penalty':pct_penalty,'pct_none':pct_none, 'pct_reward_or_none':pct_reward_or_none}
            self.pct_stats = self.pct_stats.append(new_row, ignore_index=True)

        if self.stats['total_trials']==100 and (reward > 2.0 or deadline==0):
            success_count = sum(self.stats['successes'])*1.0
            success_count_last10 = sum(self.stats['successes'][90:100])*1.0
            success_rate = success_count / 100.0
            success_rate_last10 = success_count_last10 / 10.0
            # Plot the percent of each move
            fileName = 'Different Initialization, Explore When Required'
            #plt.plot(self.pct_stats['trial_num'],self.pct_stats[['pct_none','pct_reward','pct_penalty']])
            #plt.legend(['None','Reward','Penalty'],bbox_to_anchor=(0.87,-0.1),ncol=3)
            #plt.xlabel('Trial Number')
            #plt.ylabel('Percent of Actions in Trial')
            #plt.title('Percent of Actions over Time (' + fileName + ')')
            #plt.text(0.75,-0.11,'Success Rate: '+str(success_rate), weight='bold')
            #plt.savefig('plots/'+fileName+'.png', bbox_inches='tight')
            #COMBINE REWARD AND NONE
            make_plots=False
            if make_plots:
                plt.plot(self.pct_stats['trial_num'],self.pct_stats['pct_reward_or_none'], color='green')
                plt.plot(self.pct_stats['trial_num'],self.pct_stats['pct_penalty'], color='red')
                plt.legend(['Reward or None','Penalty'],bbox_to_anchor=(0.87,-0.1),ncol=3)
                plt.xlabel('Trial Number')
                plt.ylabel('Percent of Actions in Trial')
                plt.title('Percent of Actions over Time (' + fileName + ')')
                plt.text(0.75,-0.11,'Success Rate: '+str(success_rate), weight='bold')
                plt.savefig('plots/'+fileName+'.png', bbox_inches='tight')
            #plt.show()
            print ""
            print "=====Summary Stats====="
            print "Success Rate of Last 10 Trips: %f" % success_rate_last10
            print "Success Rate Overall: %f" % success_rate
            print "Total Trials: " + str(self.stats['total_trials'])
            #print "=====Percents Table====="
            #print self.pct_stats
			
        if print_move:
            print ""
            print "===UPDATE===" + str(self.action_count) + "==="
            print "Prior state: " + str(self.prior_state)
            print "Prior line: " + str(prior_line)
            print "Prior action: " + str(self.action_list[len(self.action_list)-2])
            print "Prior reward: " + str(self.reward_list[len(self.reward_list)-2])
            print "Start state: " + str(self.current_state) # + str(start_state)
            print "Current line: " + str(current_line_str)
            print "Max Q of Current Line: " + str(max_q_current_state)
            print "Action: " + str(action)
            print "Reward: " + str(reward)
            print "Updated prior line: " + str(self.prior_state) + " : " + str(self.states[self.prior_state])

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
