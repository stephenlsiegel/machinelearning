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
        self.state = ('', '', '', '', '')
        self.route = []
        
        self.states = {}
        waypoints = ['forward', 'left', 'right']
        trafficlight = ['red','green']
        oncoming = [None, 'left', 'right', 'forward']
        right = [None, 'left', 'right', 'forward']
        left = [None, 'left', 'right', 'forward']
        
        # Create dictionary of all possible states
        for w in waypoints:
		        for t in trafficlight:
				        for o in oncoming:
						        for r in right:
								        for l in left:
										        new_key = (w,t,o,r,l)
										        self.states[new_key] = {None:0, 'forward':0, 'right':0, 'left':0}

        self.successes = []
        self.trial_num = 0.0
        self.action_count = 0.0
        self.shortest_route = 0.0
        self.route_efficiency = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.successes.append(0)
        self.trial_num += 1.0
        self.action_count = 0.0
        self.shortest_route = self.env.compute_dist(self.env.get_start(),self.planner.get_destination()) * 1.0
        #print self.planner.get_destination()
        #print self.env.get_start()
        #print self.env.compute_dist(self.env.get_start(),self.planner.get_destination())
        self.route = []
		
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'],inputs['oncoming'],inputs['right'],inputs['left'])
        
        # TODO: Select action according to your policy
        #action = random.choice([None,'forward','right','left']) # random action
		
		# Implement Q-Learning
        current_key = self.state
        current_line = self.states[current_key]
        current_values = current_line.values()
        best_action_value = max(current_values)
        num_best = sum(np.array(current_values) == best_action_value)
        
        if num_best==1:
            action = current_line.keys()[current_line.values().index(best_action_value)]
        elif num_best==2 or num_best==3:
            filt_dict = {}
            for k,v in current_line.iteritems():
                #if v == best_action_value: # before enhancing
                if v == best_action_value and k != None: # after enhancing
                    filt_dict[k] = v
                else:
                    continue
            possible_actions = filt_dict.keys()
            action = random.choice(possible_actions)
        else:
            #action = random.choice([None, 'forward', 'right', 'left']) # before enhancing
            action = random.choice(['forward', 'right', 'left']) # after enhancing

        # Execute action and get reward
        self.route.append(action)
        reward = self.env.act(self, action)
        self.action_count += 1

        # TODO: Learn policy based on state, action, reward
        self.states[current_key][action] += reward
		
        # Update successes if arrived at destination
        if reward > 2.0:
            self.successes[len(self.successes)-1] = 1
            self.success_count = sum(self.successes)*1.0
            self.accuracy = self.success_count / self.trial_num
            self.route_efficiency.append((self.shortest_route / self.action_count))
            print "Total trips: %d" % self.trial_num
            print "Successful trips: %d" % self.success_count
            print "Accuracy: %f" % self.accuracy
            print "Actions taken: %d" % self.action_count
            print "Shortest route: %d" % self.shortest_route
            print "Route efficiency: %f" % (self.shortest_route / self.action_count)
            #print "Route efficiency avg: %f" % np.array(self.route_efficiency).mean()
            #print self.route_efficiency
            #if self.trial_num == 10000:
                #re_ar = np.array(self.route_efficiency)
                #np.savetxt(fname='route_eff.txt', X=re_ar)
                #csv = pd.DataFrame(np.array(self.route_efficiency))
                #csv.to_csv("route_eff.csv")
                #plt.plot(range(1,2501), self.route_efficiency)
                #plt.ylim(ymax=1)
                #plt.xlabel('trial number')
                #plt.ylabel('route efficiency')
                #plt.title('Route Efficiency over 2,500 Trials')
                #plt.savefig('tripeffs/2500.png', bbox_inches='tight') #plt.show()"""
        elif deadline == 0:
            self.success_count = sum(self.successes)*1.0
            self.accuracy = self.success_count / self.trial_num
            self.route_efficiency.append((self.shortest_route / self.action_count))
            print "Total trips: %d" % self.trial_num
            print "Successful trips: %d" % self.success_count
            print "Accuracy: %f" % self.accuracy
            print "Actions taken: %d" % self.action_count
            print "Shortest route: %d" % self.shortest_route
            print "Route efficiency: %f" % (self.shortest_route / self.action_count)
            #print "Route efficiency avg: %f" % np.array(self.route_efficiency).mean()
            #print self.route_efficiency
            #if self.trial_num == 10000:
                #re_ar = np.array(self.route_efficiency)
                #np.savetxt(fname='route_eff.txt', X=re_ar)
                #csv = pd.DataFrame(np.array(self.route_efficiency))
                #csv.to_csv("route_eff.csv")
                #print csv
                #plt.plot(range(1,2501), self.route_efficiency)
                #plt.ylim(ymax=1)
                #plt.xlabel('trial number')
                #plt.ylabel('route efficiency')
                #plt.title('Route Efficiency over 2,500 Trials')
                #plt.savefig('tripeffs/2500.png', bbox_inches='tight') #plt.show()"""
			
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print self.states[current_key]
        #print action
        #print ""

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
