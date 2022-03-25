from queue import PriorityQueue

# Base class used to store all important steps

class State(object):
    def __init__(self, value, parent, start = 0, goal = 0):
        self.children = [] # List of all membering possibilities
        self.parent = parent
        self.value = value
        self.dist = 0
        
        # Check if parent is plucked in
        if parent:
            self.start = parent.start
            self.goal = parent.goal
            self.path = parent.path[:] # [:] ensures that self.path is a copy of parent.paths
            
            self.path.append(value) # Store all vals in path
        
        # If no parent
        else:
            self.path = [value]
            self.start = start
            self.goal = goal
            
class State_String(State):
    def __init__(self, value, parent, start = 0, goal = 0):
        super(State_String, self).__init__(value, parent, start, goal) # Create constructor
        self.dist = self.GetDistance() # Override dist var 
        
    def GetDistance(self):
        # Check if already reached goal
        if self.value == self.goal.value:
            return 0
        dist = 0
        
        # Loop to go through each letter of goal
        for i in range(len(self.goal)):
            letter = self.goal[i]
            
            dist += abs(i - self.value.index(letter))
            
        return dist
        
    # Generates children    
    def CreateChildren(self):
        # If no children, generate children
        
        if not self.children:
            for i in range(len(self.goal)-1):
                val = self.value
                
                # Switching second and first letter of every pair of letters
                val = val[:i] + val[i+1] + val[i] + val[i+2: ]
                
                # Create child and store value of child and pass self to store parent
                child = State_String(val, self)
                self.children.append(child)
                
                
class A_Star:
    def __init__(self, start, goal):
        # Store final solution from start to goal state
        self.path = []
        self.visitedQueue = []
        self.priorityQueue = PriorityQueue()
        self.start = start
        self.goal = goal
        
    def Solve(self):
        startState = State_String(self.start, 0, self.start, self.goal)
        
        count = 0
        
        # Add children
        self.priorityQueue.put((0, count, startState))
        
        while(not self.path and self.priorityQueue.qsize()):
            closestChild = self.priorityQueue.get()[2]
            closestChild.CreateChildren()
            self.visitedQueue.append(closestChild.value)
            for child in closestChild.children:
                if child.value not in self.visitedQueue:
                    count+=1
                    if not child.dist:
                        self.path = child.path
                        break
                    self.priorityQueue.put((child.dist, count, child))
        if not self.path:
            print("Goal not possible")
        return self.path
    
if __name__=="__main__":
    start1 = "hema"
    goal1 = "mahe"
    a = A_Star(start1, goal1)
    a.solve()
    for i in range(len(a.path)):
        print("{0}{1}".format(i, a.path[i]))