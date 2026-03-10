from collections import deque # 队列
import heapq # 优先队列（最小堆）

#       A
#    3/   \1
#    B     C
# 2/  5\ 2/ \3
# D ___ E     F
#    4  6\   /2
#          G
graph = {
    'A': {'B': 3, 'C': 1},
    'B': {'D': 2, 'E': 5},
    'C': {'E': 2, 'F': 3},
    'D': {'E': 4},
    'E': {'G': 6},
    'F': {'G': 2},
    'G': {}
}

# DFS（深度优先搜索）
def dfs(graph, start, target):
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        current, path = stack.pop()
        if current == target:
            return path
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in reversed(list(graph[current].keys())):  # reversed保证顺序与添加顺序一致
            stack.append((neighbor, path + [neighbor]))
    
    return None

# BFS（广度优先搜索）
def bfs(graph, start, target):
    queue = deque([(start, [start])]) # (node, path to this node)
    visited = set()
    
    while queue:
        current, path = queue.popleft() # popleft从左端弹出
        if current == target:
            return path
        if current in visited: # 再次访问时就不是最短的了，直接跳过
            continue
        visited.add(current)
        
        for neighbor in graph[current]:
            queue.append((neighbor, path + [neighbor]))
    
    return None

# UCS（一致代价搜索）
def ucs(graph, start, target):
    heap = [(0, start, [start])] # (total cost, node, path)，第一个参数是排序指标
    visited = {} # {node: total cost}
    
    while heap:
        cost, current, path = heapq.heappop(heap) # 弹出排序指标最小的节点
        print(f"[UCS] considering: {current} with g {cost}")
        if current == target: # 取出时判断，而不要在放入时判断，不然不一定最优
            return path, cost
        if current in visited: # 先前已经有cost更小的路径访问此节点
            continue
        visited[current] = cost # 将当前节点与对应代价加入visited
        
        for neighbor, edge_cost in graph[current].items():
            new_cost = cost + edge_cost
            heapq.heappush(heap, (new_cost, neighbor, path + [neighbor])) # 加入节点并根据cost自动排序
    
    return None, float('inf')

# A*搜索

# g: start -> current 用于路径计算
# h: current -> goal 由启发函数估计
# f: total f = g + h 用于节点排序

# 启发式函数（估计当前节点到目标的代价，这里利用距离估计）
# 可接受：对于移动到目标的代价，如果0 <= 估计值 <= 真实值，认为启发函数可接受（避免估计值太大了而破坏最优性）
# 一致性：对于一次移动的代价，如果0 <= 估计值 h(p1)-h(p2) <= 真实值 cost(p1->p2)，认为启发函是一致的（保证一条路上f不变小，比可接受更严格）
# 当所有估计值都是0，就成了UCS
# 可证明，启发函数可接受或是一致的时，搜索路径就是最优的
heuristic = {
    'A': 5, 'B': 4, 'C': 3, 
    'D': 3, 'E': 2, 'F': 1, 'G': 0 # 终点启发值固定为0
}

def a_star(graph, start, target, heuristic):
    heap = [(heuristic[start], 0, start, [start])] # (f, g, node, path)
    visited = {}
    
    while heap:
        _, g_cost, current, path = heapq.heappop(heap) # f仅用于排序，不用读取，弹出f最小的节点
        print(f"[A*] considering: {current}")
        if current == target:
            return path, g_cost
        if current in visited and visited[current] < g_cost:
            continue
        visited[current] = g_cost
        
        for neighbor, edge_cost in graph[current].items():
            new_g = g_cost + edge_cost
            f_cost = new_g + heuristic[neighbor]
            heapq.heappush(heap, (f_cost, new_g, neighbor, path + [neighbor]))
    
    return None, float('inf')

# 执行搜索
target = 'G'
print("DFS路径:", dfs(graph, 'A', target))          # 一条路走到黑
print("BFS路径:", bfs(graph, 'A', target))          # 最短的，但不考虑cost
ucs_path, ucs_cost = ucs(graph, 'A', target)
print(f"UCS路径: {ucs_path}, 总代价: {ucs_cost}")  # 最优的，比较慢
a_path, a_cost = a_star(graph, 'A', target, heuristic)
print(f"A*路径: {a_path}, 总代价: {a_cost}")       # 最优的，更高效（效率受启发函数影响）