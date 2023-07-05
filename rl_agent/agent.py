import random 
import numpy as np 
import state as st
import dt
from sklearn import tree
from matplotlib import pyplot as plt
import copy
import random 
import graphviz
class Soccer_Agent:
    def __init__(self,
                actions,
                inital_state,
                state_dim, 
                dis_steps = 5000,
                rmax=40,
                max_steps=8,
                lr = 0.99
                ):
        
        self.max_steps = max_steps
        self.rmax= rmax 
        self.actions = actions
        self.decision_trees = {}
        self.lr = lr
        self.state_dim = state_dim
        self.dis_steps = dis_steps
        self.action_decoding = {"left": 0, "right": 1, "kick": 2}
        # Initalize the n decision trees for every state variable 
        for sd in self.state_dim.keys():
            for j in range(state_dim[str(sd)]["dim"]):
                self.decision_trees[sd  +"_" + str(j)] = dt.DT()

        # Initalize the decision tree of the reward function 
        self.decision_trees["reward"] = dt.DT()
       
      
        assert type(inital_state["joint_values"]) == np.ndarray, "Inital state is missing joint values field or its not a np array "
        assert type(inital_state["goal_keeper_position"])  == np.ndarray, "Inital state is missing goal keeper position field or its not a np array "
        self.states = [st.State(inital_state["joint_values"], inital_state["goal_keeper_position"], actions=actions)]
        self.current_state = self.states[0]
        self.terminal_state = st.Terminal_State()

    def take_action(self,):
        act = self.current_state.choose_action()
        self.current_state.state_visited(act)
        return act 

     
    def action_taken(self, next_state, reward:float):
        """Given a experienced state, this function transformes this state into an internal representation 

        Args:
            next_state (dict): dictonary containing the state dimensions and values as key-value pairs 

        Returns:
            State: internal representation of that state
        """
        experiece_input = []

        # To create a new experience for every decision tree we extract every state dimension s_1,...,s_n, a
        for i in self.current_state.joint_values:
            experiece_input.append(i)
        
        for i in self.current_state.goal_keeper_position:
            experiece_input.append(i)
        # We decode the action 
        experiece_input.append(self.action_decoding[self.current_state.last_action])

        if self.current_state.last_action == "kick":
            next_state = self.terminal_state

        # Only performed for "left" or "right"
        else:
            next_state, _ = self._check_state(next_state)
            x_i = {}
            x_i["joint_values"] =  np.array(next_state.joint_values) - np.array(self.current_state.joint_values)
            if  abs(x_i["joint_values"]) > 1:
                print("ERROR")
            x_i["goal_keeper_position"] = np.array(next_state.goal_keeper_position) -np.array(self.current_state.goal_keeper_position)
            # We will know add this experience to every decision tree with the respective output 
            for sd in self.state_dim.keys():
                for j in range(self.state_dim[str(sd)]["dim"]):
                    dt = self.decision_trees[sd  +"_" + str(j)] 
                    dt.add_experience(experiece_input, x_i[sd][j])

        # Same for the reward DT
        # This will be performed for the all actions
        dt = self.decision_trees["reward"] 
        dt.add_experience(experiece_input, reward)

        # Set the current state to the next_state after transition is completed
        self.current_state = next_state

    def RM(self, s, a):
        """This function is the reward estimate based on the decision tree

        Args:
            s (_type_): _description_
            a (_type_): _description_
        """
        # Later this needs to call the DT and return its reward
        input = np.array([s.joint_values[0], s.goal_keeper_position[0], a])
        pred_reward = self.decision_trees["reward"].predict(input)
        return pred_reward
    
    def PM(self, s:st.State,a ):
        """This function is the transition function estimate based on the decision tree 
        Args:
            s (_type_): _description_
            a (_type_): _description_
        """
        if a == 2:
            return [(self.terminal_state,1.0)]
        else:
            input = np.array([s.joint_values[0], s.goal_keeper_position[0], a])
            input_change = np.array([])
            for sd in self.state_dim.keys():
                for j in range(self.state_dim[str(sd)]["dim"]):
                    dt = self.decision_trees[sd  +"_" + str(j)] 
                    rel = dt.predict(input)
                    input_change = np.append(input_change, rel)
            nxt = (input[:-1] + input_change).tolist()
            #nxt[0] = np.clip(nxt[0],self.state_dim["joint_values"]["min_max"][0][0],
            #                self.state_dim["joint_values"]["min_max"][0][1])
            #nxt[1] = np.clip(nxt[1],self.state_dim["goal_keeper_position"]["min_max"][0][0],
            #                self.state_dim["goal_keeper_position"]["min_max"][0][1])
            return [([nxt], [1.0])]
        # Later this needs to return a list of (state, probability pairs)
        
    def check_model(self, inital_state):
        curr_state, _ = self._check_state(inital_state)
        max_steps = 30
        exp = True
        for i in range(max_steps):
                act = curr_state.choose_action()
                reward =  self.RM(curr_state, self.action_decoding[act])
                if reward > 0:
                  exp = False
                else:
                    nxt  =  self.PM(curr_state, self.action_decoding[act])[0]
                    nxt = nxt[0]
                    if type(nxt) == st.Terminal_State:
                        break
                    else:
                        nxt  = {"joint_values": np.array(nxt[0][0],), "goal_keeper_position": np.array(nxt[0][1],)}
                        nxt, _ = self._check_state(nxt)
                        curr_state = nxt
               
        return exp 
                 
        return nxt, reward
    def _check_state(self, state):
        """
        This function is used to check a incoming state for its existance in the list of states.
        If found this state will be returned, otherwise a new state object will be created. 
        Args:
            state (state.State): incoming state to be checked
            terminal (bool): Check if state is terminal 
        Returns: 
            State: Retrieved State Object 
        """
      
        if not state["joint_values"].shape and not state["goal_keeper_position"].shape:
            state["joint_values"] = [int(state["joint_values"])]
            state["goal_keeper_position"] = [int(state["goal_keeper_position"])]

        for s in self.states:
            if s.joint_values == state["joint_values"] and \
               s.goal_keeper_position == state["goal_keeper_position"]:
                return s, True
        new_state = st.State(joint_values=state["joint_values"],
                        goal_keeper_position=state["goal_keeper_position"],
                        actions=self.actions)
        self.states.append(new_state)
        return new_state, False
      
    def update_model(self):
        for sd in self.state_dim.keys():
            for j in range(self.state_dim[str(sd)]["dim"]):
                dt = self.decision_trees[sd  +"_" + str(j)]
                dt.update()
                #if dt.clf:
                    #tree.plot_tree(dt.clf)
                    #plt.show()
                
        dt = self.decision_trees["reward"]
        dt.update()
        #if dt.clf:
                #tree.plot_tree(dt.clf)
                #plt.show()
    """
    def check_model(self, start_state):
        state = self._check_state(start_state)
    """   

    def _setup_iteration(self):
        Ks = {}
        min_visits = None
        q_values = {}
        for s in self.states:
            # We check if a state has been at least visited once
            # if at least visited once we will set Ks to 0 
            if s.total_visits() > 0:
                Ks[s] = 0
            # Otherwise we will initalize it to infinity 
            else:
                Ks[s] = np.inf
            # Next we initalizte the minimum visits among all states 
            if not min_visits:
                min_visits = s.total_visits()
            else:
                if s.total_visits() < min_visits:
                    min_visits = s.total_visits()
            # Initalize all q values with 0 
            q_values[s] = {k : 0 for k in self.actions}
        return Ks, min_visits, q_values

    def value_iteration(self, exploration=False):
        """
        this function implements the compute_values algorithm 
        """
        # We setup an iteration 
        Ks, min_visits, q_values = self._setup_iteration()
        step = 0
        last_q_values = {}
        for s in q_values:
            last_q_values[s] = {a : 0 for a in self.actions}
        while True: 
            
        
            # First we initalize the
            inital_states = self.states.copy()
            # We set the values for the terminal state to 0 (since no more reward will be expeced after reaching it)
            
            q_values[self.terminal_state] = {a: 0 for a in self.actions}
            for s in self.states:
                    for a in self.actions:
                        # Setting states with maximum reward 
                        if exploration:
                            pass
                        if exploration and s.total_visits() == min_visits or Ks[s] > self.max_steps:
                            q_values[s][a] = self.rmax
                        else:    
                            q_values[s][a] = self.RM(s,self.action_decoding[a])
                            
                            predicted_next = self.PM(s, self.action_decoding[a])
                            update_q = 0.0
                            for p, prob in predicted_next: 
                                if p == self.terminal_state: 
                                    state = p 
                                    update_q = 0
                                else: 
                                    prob = prob[0]
                                    p_state  = {"joint_values": np.array(p[0][0],), "goal_keeper_position": np.array(p[0][1],)}
                                    state, _ = self._check_state(p_state)
                                    if not _ :
                                        q_values[state] = {k : 0 for k in self.actions}
                                        last_q_values[state] = {k : 0 for k in self.actions}
                                    if state in Ks:
                                        maximum_q = np.NINF
                                        for ac in self.actions: 
                                            if q_values[state][ac] > maximum_q:
                                                maximum_q =  q_values[state][ac]
                                        update_q = prob*maximum_q
                                        if Ks[state] > Ks[s] +1:
                                            Ks[state] = Ks[s] +1
                                    else: 
                                        Ks[state] = Ks[s] +1
                                        update_q = 0
                                
                                q_values[s][a] += self.lr * update_q
            update_amount = 0.
            for s in self.states:
                for a in self.actions:
                    try:
                        up_i = (q_values[s][a] - last_q_values[s][a])**2
                        update_amount += up_i
                    except KeyError:
                        print("HERE")
            for s in q_values:
                last_q_values[s] = copy.deepcopy(q_values[s])
            if abs(update_amount) <= 0.1:
                for s in self.states:
                    for a in self.actions:
                        s.int_state[a]["q_value"] = q_values[s][a]
                return q_values
            step += 1

sucessfull_pos = {1: 2, 2: 4, 3: 8}
gk_pos = 1
hip_pos = 0
inital_state = {"joint_values": np.array([hip_pos]), "goal_keeper_position": np.array([gk_pos])}
agent = Soccer_Agent(actions=["left", "right", "kick"], inital_state=inital_state, state_dim={"joint_values": {"dim": 1, "min_max": np.array([[0,10]])},
                                                                         "goal_keeper_position":  {"dim": 1, "min_max": np.array([[1,3]])}})


reward = 0
terminated = False
episode_number = 1
steps = 0 
while episode_number < 40:
    """
    if i % 50 == 0 and i != 0:
        tree.plot_tree(agent.decision_trees["joint_values_0"].clf)
        plt.show()
    """
    if not terminated and steps < 10: 

        action = agent.take_action()
        if action == "right":
            if hip_pos > 0:
                hip_pos -= 1
            reward = -1
        elif action == "left": 
            if hip_pos < 10:
                hip_pos += 1
            reward = -1
        else:

            if hip_pos == sucessfull_pos[gk_pos]:
                reward = 20
            else: 
                reward = -2
            terminated = True
        next_state = {"joint_values": np.array([hip_pos]), "goal_keeper_position": np.array([gk_pos])}
        #print("\nSTATE {} \nACTION TAKEN {} \nNEXT STATE {}\n Reward {}\n Step Count {}\n".format(str(agent.current_state), action, next_state, reward, steps))
        agent.action_taken(next_state, reward)
        agent.update_model()
        exp = agent.check_model(inital_state)
        #print("Explore mode ", exp)
        q_values = agent.value_iteration(exploration=exp)
        steps += 1
        for i in q_values:
            
            if type(i) != st.Terminal_State and \
               i.joint_values[0] >= 0 and \
               i.joint_values[0] < 11 and \
               i.goal_keeper_position[0] > 0 and\
               i.goal_keeper_position[0] < 4:
                #print(str(i), q_values[i])
                pass
    else:
        """
        cf = agent.decision_trees["joint_values_0"].clf
        if cf: 
            tree.plot_tree(cf)
            plt.show()
        """
        exp = agent.check_model(inital_state)
        if not exp:
            pass
        hip_pos = random.randint(0,10)
        if not exp:
            #gk_pos = random.randint(1,3)
            #hip_pos = random.randint(0,10)
            #inital_state = {"joint_values": np.array([hip_pos]), "goal_keeper_position": np.array([gk_pos])}
            pass
        inital_state = {"joint_values": np.array([hip_pos]), "goal_keeper_position": np.array([gk_pos])}
        state, _ = agent._check_state(inital_state)
        episode_number += 1
        print("KICKED, Starting new episode {} with {}\n\n".format(episode_number, str(state)))
        agent.current_state = state
        terminated = False
        steps = 0
        


# Save the transition graph 
dot_data = tree.export_graphviz(agent.decision_trees["joint_values_0"].clf, out_file=None, 
                 feature_names=["Hip Pos", "Gk Pos", "Action"])
graph = graphviz.Source(dot_data)
graph.render("transition", format="png")
# Save the reward graph 
dot_data = tree.export_graphviz(agent.decision_trees["reward"].clf, out_file=None, 
                 feature_names=["Hip Pos", "Gk Pos", "Action"])
graph = graphviz.Source(dot_data)
graph.render("reward", format="png")
"""
    Rewards:

        -20 when falling over
        20 for goal 
        -2 if no goal 
        -1 for moving the leg without falling over      
"""
