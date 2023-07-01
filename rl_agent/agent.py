import random 
import numpy as np 
import state as st
import dt
    
class Soccer_Agent:
    def __init__(self,
                actions,
                inital_state,
                state_dim, 
                dis_steps = 5000,
                rmax=10,
                max_steps=5,
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

    def take_action(self,):
        act = self.current_state.choose_action()
        self.current_state.state_visited(act)
        return act 

    def discretize_state(self, state):
        dis_state = {}
        for sd in self.state_dim.keys():
            dis_state[sd] = np.empty(self.state_dim[str(sd)]["dim"])
            for j in range(self.state_dim[str(sd)]["dim"]):
                value = state[str(sd)][j]
                # We first clip the value to lie in the desired min-max range
                value = np.clip(value, a_min = self.state_dim[str(sd)]["min_max"][j][0],  a_max = self.state_dim[str(sd)]["min_max"][j][1])
                # Then we bin it
                bin_bounds = np.linspace(start=self.state_dim[str(sd)]["min_max"][j][0],stop=self.state_dim[str(sd)]["min_max"][j][1], num=self.dis_steps+1, endpoint=True)
            
                value = np.digitize(value, bin_bounds)
                print(value)
                value = (bin_bounds[value-1] +  bin_bounds[value]) / 2
                dis_state[sd][j] = value
        return dis_state
    
    def action_taken(self, next_state, reward:float):
        """Given a experienced state, this function transformes this state into an internal representation 

        Args:
            next_state (dict): dictonary containing the state dimensions and values as key-value pairs 

        Returns:
            State: internal representation of that state
        """

        # We discretize the state
        print("Next state ", next_state)
        next_state, _ = self._check_state(next_state)

        x_i = {}
        x_i["joint_values"] =  next_state.joint_values - self.current_state.joint_values
        x_i["goal_keeper_position"] = next_state.goal_keeper_position -self.current_state.goal_keeper_position 

        experiece_input = []

        # To create a new experience for every decision tree we extract every state dimension s_1,...,s_n, a
        for i in self.current_state.joint_values:
           
            experiece_input.append(i)
        
        for i in self.current_state.goal_keeper_position:
          
            experiece_input.append(i)
        # We decode the action 
        experiece_input.append(self.action_decoding[self.current_state.last_action])
        print(experiece_input)
        # We will know add this experience to every decision tree with the respective output 
        for sd in self.state_dim.keys():
            for j in range(self.state_dim[str(sd)]["dim"]):
                dt = self.decision_trees[sd  +"_" + str(j)] 
                print("Adding Experience to {}-{} DT with Input: {} Output: [{},]".format(sd, j, experiece_input,  x_i[sd][j]))
                dt.add_experience(experiece_input, x_i[sd][j])

        # Same for the reward DT
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

        return 1.0
    
    def PM(self, s:st.State,a ):
        """This function is the transition function estimate based on the decision tree 
        Args:
            s (_type_): _description_
            a (_type_): _description_
        """
        print(s.joint_values)
        print(s.goal_keeper_position)
        print(a)
        input = np.array([s.joint_values[0], s.goal_keeper_position[0], a])
        input_change = np.array([])
        for sd in self.state_dim.keys():
            for j in range(self.state_dim[str(sd)]["dim"]):
                dt = self.decision_trees[sd  +"_" + str(j)] 
                rel = dt.predict(input)
                input_change = np.append(input_change, rel)
        return [([(input[:-1] + input_change).tolist()], [1.0])]
        # Later this needs to return a list of (state, probability pairs)
        
    
    def _check_state(self, state):
        """
        This function is used to check a incoming state for its existance in the list of states.
        If found this state will be returned, otherwise a new state object will be created. 
        Args:
            state (_type_): incoming state to be checked

        Returns: 
            State: Retrieved State Object 
        """
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
        dt = self.decision_trees["reward"].update()
    def check_model(self, start_state):
        state = self._check_state(start_state)
        
    def value_iteration(self, exploration=False):
        """
        this function implements the compute_values algorithm 
        """
        Ks = {}
        min_visits = None
        for s in self.states:
            internal_state = s.int_state
            one_action_visited = False
            for a in self.actions:
                if internal_state[a]["visits"] > 0:
                    one_action_visited = True
            if one_action_visited:
                Ks[s] = 0
            else:
                Ks[s] = np.inf
            if not min_visits:
                min_visits = s.total_visits()
            else:
                if s.total_visits() < min_visits:
                    min_visits = s.total_visits()
        # Perform value iteration 
        q_values = {}
        for s in self.states:
                q_values[s] = {k : 0 for k in self.actions}
        while True: 
            total_learning_amount = 0.0
            learning_number = 0
            # We first initalize the q function 
           
                    
            # First we initalize the
            for s in self.states:
                for i,a in enumerate(self.actions):
                    # Setting states with maximum reward 
                    if exploration and s.total_visits() == min_visits or Ks[s] > self.max_steps:
                        q_values[s][a] = self.rmax
                    else:    
                        q_values[s][a] = self.RM(s,a)
                        
                    predicted_next = self.PM(s, self.action_decoding[a])
                    update_q = 0.0
                    for p, prob in predicted_next: 
                        prob = prob[0]
                        p_state  = {"joint_values": np.array(p[0][0]), "goal_keeper_position": np.array(p[0][1])}
                        state, _ = self._check_state(p_state)
                        if not _ :
                            q_values[state] = {k : 0 for k in self.actions}
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
                        total_learning_amount += self.lr * update_q
                        q_values[s][a] += self.lr * update_q
            learning_number += 1
            convergerence_cricerion = total_learning_amount / learning_number
            if convergerence_cricerion <= 0.1:
                return q_values
            
inital_state = {"joint_values": np.array([0]), "goal_keeper_position": np.array([0])}
agent = Soccer_Agent(actions=["left", "right", "kick"], inital_state=inital_state, state_dim={"joint_values": {"dim": 1, "min_max": np.array([[0,12]])},
                                                                         "goal_keeper_position":  {"dim": 1, "min_max": np.array([[1,6]])}})
action = agent.take_action()

if action == "right":

    next_state = {"joint_values": np.array([1]), "goal_keeper_position": np.array([0])}
else: 
     next_state = {"joint_values": np.array([0]), "goal_keeper_position": np.array([0])}
agent.action_taken(next_state, -1.)
agent.update_model()
print(agent.value_iteration())

"""
    Rewards:

        -20 when falling over
        20 for goal 
        -2 if no goal 
        -1 for moving the leg without falling over      
"""
