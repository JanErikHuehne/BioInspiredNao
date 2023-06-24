import random 
import numpy as np 
import state as st
import dt
    
class Soccer_Agent:
    def __init__(self,
                actions,
                inital_state,
                state_dim,
                rmax=10,
                max_steps=5,
                lr = 0.01
                ):
        
        self.max_steps = max_steps
        self.rmax= rmax 
        self.actions = actions
        self.state_dim = state_dim
        self.decision_trees = {}
        self.lr = 0.01
       

        # Initalize the n decision trees for every state variable 
        for sd in self.state_dim.keys():
            for j in range(state_dim[str(sd)]):
                self.decision_trees[sd  +"_" + str(j)] = dt.DT()

        # Initalize the decision tree of the reward function 
        self.decision_trees["reward"] = dt.DT()
       
      
        assert type(inital_state["joint_values"]) == np.ndarray, "Inital state is missing joint values field or its not a np array "
        assert type(inital_state["ball_position"]) == np.ndarray, "Inital state is missing ball position field or its not a np array "
        assert type(inital_state["goal_keeper_position"])  == np.ndarray, "Inital state is missing goal keeper position field or its not a np array "
        self.states = [st.State(inital_state["joint_values"], inital_state["ball_position"], inital_state["goal_keeper_position"], actions)]
        self.current_state = self.states[0]

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
        next_state, _ = self._check_state(next_state)

        x_i = {}
        x_i["joint_values"] = self.current_state.joint_values - next_state.joint_values
        x_i["ball_position"] = self.current_state.ball_position - next_state.ball_position
        x_i["goal_keeper_position"] = self.current_state.goal_keeper_position - next_state.goal_keeper_position

        experiece_input = []

        # To create a new experience for every decision tree we extract every state dimension s_1,...,s_n, a
        for i in self.current_state.joint_values:
            experiece_input.append(i)
        
        for i in self.current_state.ball_position:
            experiece_input.append(i)
        
        for i in self.current_state.goal_keeper_position:
            experiece_input.append(i)

        experiece_input.append(self.current_state.last_action)

        # We will know add this experience to every decision tree with the respective output 
        for sd in self.state_dim.keys():
            for j in range(self.state_dim[str(sd)]):
                dt = self.decision_trees[sd  +"_" + str(j)] 
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
    
    def PM(self, s,a ):
        """This function is the transition function estimate based on the decision tree 
        Args:
            s (_type_): _description_
            a (_type_): _description_
        """

        # Later this needs to return a list of (state, probability pairs)
        return [({"joint_values": np.array([0.2]), "ball_position": np.array([0,0]), "goal_keeper_position": np.array([0,0])},0.4)]
    
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
               s.ball_position == state["ball_position"] and \
               s.goal_keeper_position == state["goal_keeper_position"]:
                return s, True
       
        new_state = st.State(joint_values=state["joint_values"],
                        ball_position=state["ball_position"],
                        goal_keeper_position=state["goal_keeper_position"],
                        actions=self.actions)
        self.states.append(new_state)
        return new_state, False
      
    
    def value_iteration(self, exploration=True):
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
     
        while True: 
            total_learning_amount = 0.0
            learning_number = 0
            for s in self.states:
                for i,a in enumerate(self.actions):
                    # Setting states with maximum reward 
                    if exploration and s.total_visits() == min_visits or Ks[s] > self.max_steps:
                        if i == 0:
                            q_values[s] = {a : self.rmax}
                        else:
                             q_values[s][a] = self.rmax
                    else:
                        if i == 0: 
                            q_values[s] = {a : self.RM(s,a)}
                        else: 
                            q_values[s][a] = self.RM(s,a)
                        
                        predicted_next = self.PM(s, a)
                        update_q = 0.0
                        for p, prob in predicted_next: 
                            state, _ = self._check_state(p)
                            if state in Ks:
                                maximum_q = np.NINF
                                for ac in self.actions: 
                                    if q_values[state][ac] > maximum_q:
                                        maximum_q =  q_values[state][ac]
                                update_q += prob*maximum_q
                                if Ks[state] > Ks[s] +1:
                                     Ks[state] = Ks[s] +1
                            else: 
                                Ks[state] = Ks[s] +1
                        total_learning_amount += self.lr * update_q
                        learning_number += 1
                        q_values[s,a] += self.lr * update_q
            convergerence_cricerion = total_learning_amount / learning_number
            if convergerence_cricerion <= 0.1:
                return q_values
            
inital_state = {"joint_values": np.array([0]), "ball_position": np.array([0,0]), "goal_keeper_position": np.array([0,0])}
agent = Soccer_Agent(["left", "right", "kick"], inital_state, state_dim={"joint_values": 1, "ball_position": 2,  "goal_keeper_position": 2})
agent.take_action()

next_state = {"joint_values": np.array([0.2]), "ball_position": np.array([0,0]), "goal_keeper_position": np.array([0,0])}
agent.action_taken(next_state, -1.)
agent.value_iteration()
