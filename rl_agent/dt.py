

class Experience:
    def __init__(self, input, output):
        self.input = input
        self.output = output
    
    def __str__(self):
        return str(self.input + self.output)
class DT:
    def __init__(self,):
        self.experiences = []
    
    def add_experience(self, input, output):
        exp = Experience(input=input, output=output)  
        exists = False
        for exp in self.experiences:
            if exp.input == input and exp.output == output:
                exists = True
        if not exists: 
            self.experiences.append(exp)
      