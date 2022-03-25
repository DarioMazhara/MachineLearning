from numpy import vsplit


class Graph:
    adj = []
    
    # Fill empty adj matrix
    def __init__(self, v, e):
        self.v = v
        self.e = e
        Graph.adj = [[0 for i in range(v)], [0 for j in range(v)]]
        
    def add_edge(self, start, e):
        Graph.adj[start][e] = 1
        Graph.adj[e][start] = 1
        
    def BFS(self, start):
        visited = [False] * self.v
        q = [start]
        visited[start] = True
        
        while q:
            vis = q[0]
            print (vis, end = " ")
            q.pop(0)
            
            for i in range(self.v):
                if (Graph.adj[vis][1] == 1 and not visited[i]):
                    q.append(i)
                    visited[i] = True
    
    def DFS(self, start, visited):
        print(start, end = " ")
        visited[start] = True
        
        for i in range(self.v):
            if(Graph.adj[start][i] == 1 and not visited[i]):
                self.DFS(i, visited)
            