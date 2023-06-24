import random 

class State:
    def __init__(self, joint_values, ball_position, goal_keeper_position, actions):
        self.joint_values = joint_values
        self.ball_position = ball_position
        self.goal_keeper_position = goal_keeper_position
        self.int_state = {a : {"visits": 0, "q_value": 0.0} for a in actions}
        self.last_action = None

    def choose_action(self,):
        """
        This method is called to get the action to take in the state object based on the specific q values of each (s,a) pair.
        If multiple (s,a) pairs for this state have the same q-value, a random action out of these posibilities is selected. 
        """
        max_actions = []
        max_q = 0
        for a in self.int_state:
            if self.int_state[a]["q_value"] >= max_q:
                max_q = self.int_state[a]["q_value"]
                max_actions.append(a)
        if len(max_actions) > 1:
            selected_action = str(max_actions[random.randint(0, len(max_actions)-1)])
        else:
            selected_action = str(max_actions[0])
        return selected_action

    def state_visited(self, action):
        self.int_state[action]["visits"] += 1
        self.last_action = action
    
    def total_visits(self):
        total_visits = 0
        for a in self.int_state.keys():
            total_visits += self.int_state[a]["visits"]
        return total_visits
    
    def __str__(self,):
        return self.int_state.__str__()