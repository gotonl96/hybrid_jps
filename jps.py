import math, time, heapq

def heuristic(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def blocked(cx, cy, dx, dy, matrix) :
    # outbound check
    if cx + dx < 0 or cx + dx >= matrix.shape[0]:
        return True
    if cy + dy < 0 or cy + dy >= matrix.shape[1]:
        return True
    
    # cornor check
    if dx !=0 and dy != 0 :
        if matrix[cx + dy][cy] == 1 and matrix[cx][cy + dy] == 1:
            return True
    else:
        if dx != 0:
            if matrix[cx + dx][cy] == 1:
                return True
        else:
            if matrix[cx][cy + dy] == 1:
                return True
            
    return False
    # 

def nodeNeighbours(cx, cy, parent, matrix):
    neighbours = []
    if type(parent) != tuple :
        for i, j in [
            (-1, 0),
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            if not blocked(cx, cy, i, j, matrix):
                neighbours.append(cx + i, cy + j)
            
        return neighbours



def identifySuccessors(cx, cy, came_from, matrix, goal):
    successors = []
    neighbours = nodeNeighbours(cx, cy, came_from.get((cx, cy), 0), matrix)



came_from = {}
close_set = set()
start = (0, 0)
goal = (100, 100)

gscore = {start: 0}
fscore = {start: heuristic(start, goal)}

pqueue = []

heapq.heappush(pqueue, (fscore[start], start))

starttime = time.time()

while pqueue :
    current = heapq.heappop(pqueue)[1]
    
    if current == goal:
        data = []
        while current in came_from:
            data.append(current)
            current = came_from[current]
        data.append(start)
        data = data[::-1]
        endtime = time.time()
        
        # return (data, round(endtime - starttime, 6))
    
    successors = identifySuccessors(

    )

        

