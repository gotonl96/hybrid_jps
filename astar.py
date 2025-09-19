import math, heapq, time

def blocked(cY, cX, dY, dX, matrix):
    ny, nx = cY + dY, cX + dX
    if nx < 0 or nx >= matrix.shape[1]:
        return True
    if ny < 0 or ny >= matrix.shape[0]:
        return True
    if dX != 0 and dY != 0:
        if matrix[ny][cX] == 1 and matrix[cY][nx] == 1:
            return True
        if matrix[ny][nx] == 1:
            return True
    else:
        if dX != 0:
            if matrix[cY][nx] == 1:
                return True
        else:
            if matrix[ny][cX] == 1:
                return True
    return False

def heuristic(a, b, hchoice):
    # a, b: (y, x)
    if hchoice == 1:
        xdist = abs(b[1] - a[1])
        ydist = abs(b[0] - a[0])
        if xdist > ydist:
            return 14 * ydist + 10 * (xdist - ydist)
        else:
            return 14 * xdist + 10 * (ydist - xdist)
    if hchoice == 2:
        return math.sqrt((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2)

def method(matrix, start, goal, hchoice):
    # start, goal: (y, x)
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal, hchoice)}

    pqueue = []
    heapq.heappush(pqueue, (fscore[start], start))
    starttime = time.time()

    while pqueue:
        current = heapq.heappop(pqueue)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]
            endtime = time.time()
            return (path, round(endtime - starttime, 6))

        close_set.add(current)
        for dY, dX in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            if blocked(current[0], current[1], dY, dX, matrix):
                continue

            neighbour = (current[0] + dY, current[1] + dX)

            if hchoice == 1:
                if dX != 0 and dY != 0:
                    tentative_g_score = gscore[current] + 14
                else:
                    tentative_g_score = gscore[current] + 10
            elif hchoice == 2:
                if dX != 0 and dY != 0:
                    tentative_g_score = gscore[current] + math.sqrt(2)
                else:
                    tentative_g_score = gscore[current] + 1

            if neighbour in close_set:
                continue

            if tentative_g_score < gscore.get(neighbour, float('inf')) or neighbour not in [i[1] for i in pqueue]:
                came_from[neighbour] = current
                gscore[neighbour] = tentative_g_score
                fscore[neighbour] = tentative_g_score + heuristic(neighbour, goal, hchoice)
                heapq.heappush(pqueue, (fscore[neighbour], neighbour))
        endtime = time.time()
    return (0, round(endtime - starttime, 6))