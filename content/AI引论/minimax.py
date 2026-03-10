import math

# 模拟的搜索树

#           A              max
#        /     \
#      B        C          min
#    /   \    /   \ 
#   D     E  F     G       max
#   10    8  4     50

graph = {
    'A': ['B', 'C'],   
    'B': ['D', 'E'],   
    'C': ['F', 'G'],   
    'D': [],           
    'E': [],        
    'F': [],        
    'G': []            
}

# 节点的预设评分（叶子节点）
node_scores = {
    'D': 10,
    'E': 8,
    'F': 4,
    'G': 50,
    # 非叶子节点（如 A, B, C 等）的评分将通过 Minimax 推导得出
}

# 存储被剪枝的节点
pruned_nodes = []

# 使用 Minimax 和 Alpha-Beta 剪枝的函数
# alpha: 上方的max层目前至少能取多少
# beta: 上方的min层目前至多能取多少
def minimax(node, depth, alpha, beta, is_maximizing):
    print(f"searching {node}...")
    # 如果当前节点是叶子节点，直接返回其预设评分（实际中为 depth == 0 时使用评估函数获取）
    if node in node_scores:
        return node_scores[node]

    # 当前是最大化的情况
    if is_maximizing:
        max_eval = -math.inf
        for child in graph[node]:
            eval = minimax(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            if max_eval >= beta: # 当前max值，即当前max层至少可取的值，如果比beta大，就不再考虑了
                pruned_nodes.extend(graph[node][graph[node].index(child) + 1:])  # 添加被剪枝的节点
                return max_eval  # Beta 剪枝，剪掉后没必要更新alpha，因为这里的alpha只会传递给下层的节点
            alpha = max(alpha, eval)
        return max_eval
    
    # 当前是最小化的情况
    else:
        min_eval = math.inf
        for child in graph[node]:
            eval = minimax(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            if min_eval <= alpha: # 当前min值，即当前min层至多可取的值，如果比alpha小，就不再考虑了
                pruned_nodes.extend(graph[node][graph[node].index(child) + 1:])  # 添加被剪枝的节点
                return min_eval  # Alpha 剪枝
            beta = min(beta, eval)
        return min_eval


# 主程序
if __name__ == "__main__":
    # 根节点是 'A'，假设从最大化玩家开始
    root = 'A'
    max_depth = 2  # 设置最大搜索深度，视具体应用场景调整

    # 调用 Minimax 算法计算根节点的最佳分数
    best_score = minimax(root, max_depth, -math.inf, math.inf, True)
    print(f"根节点 {root} 的最佳分数为: {best_score}")
    print(f"被剪枝的节点为: {pruned_nodes}")