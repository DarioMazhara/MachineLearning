# Adjaceny matrix representation

class Graph(object):
    # Initialize graph
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size
        
    # Add edge
    def add_edge(self, v1, v2):
        if v1==v2:
            print ("Same vertex")
            return
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1
    
    # Remove edge
    def remove_edges(self, v1, v2):
        if v1==v2:
            print ("Same vertex")
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0
        
    def __len__(self):
        return self.size
    # Print matrix
    
    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print('{:4}'.format(val))


def main():
    graph = Graph(5)
    
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 1)
    
    graph.print_matrix()
    
    
if __name__=='__main__':
    main()
            