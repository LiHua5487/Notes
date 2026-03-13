import math
import random
import matplotlib.pyplot as plt
import networkx as nx


# 模拟搜索树
#                    A
#          /         |           \
#        B           C            D
#      /   \       /   \        /   \
#     E     F     G     H      I     J  
#    / \         / \   / \    / \   / \
#   K   L       M   N O   P  Q   R S   T

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': ['G', 'H'],
    'D': ['I', 'J'],
    'E': ['K', 'L'],
    'F': [],
    'G': ['M', 'N'],
    'H': ['O', 'P'],
    'I': ['Q', 'R'],
    'J': ['S', 'T'],
    'K': [],
    'L': [],
    'M': [],
    'N': [],
    'O': [],
    'P': [],
    'Q': [],
    'R': [],
    'S': [],
    'T': []
}


# 底层结果：为叶节点定义 'win' 或 'lose'
leaf_results = {
    'F': 'lose',
    'K': 'win',
    'L': 'win',
    'M': 'lose',
    'N': 'lose',
    'O': 'win',
    'P': 'win',
    'Q': 'win',
    'R': 'lose',
    'S': 'lose',
    'T': 'win'
}


class Node:
    def __init__(self, name):
        self.name = name      # 节点名称
        self.visits = 0       # 访问次数
        self.wins = 0         # 胜利次数
        self.children = []    # 子节点


# 构建树结构
def build_tree(graph, node_name):
    node = Node(node_name)
    for child_name in graph[node_name]:
        child_node = build_tree(graph, child_name)
        node.children.append(child_node)
    return node


# MCTS
def mcts(root, iterations):
    for _ in range(iterations):
        # 1. 选择
        node_path = []
        current_node = root
        while current_node.children:  # 未达到叶子节点
            node_path.append(current_node)
            current_node = select(current_node)
        node_path.append(current_node)

        # 2. 扩展
        if not is_leaf(current_node.name):  # 如果不是叶子节点，可以扩展
            for child in current_node.children:
                if child.visits == 0:  # 找到一个还没模拟过的节点
                    current_node = child
                    node_path.append(child)
                    break

        # 3. 模拟
        result = simulate(current_node.name)

        # 4. 回传
        for node in reversed(node_path):
            node.visits += 1
            if result == 'win':
                node.wins += 1


# 节点选择（UCB1）
def select(node):
    c = 2

    best_score = -float('inf')
    best_child = None
    for child in node.children:
        if child.visits == 0:
            score = float('inf')
        else:
            score = child.wins / child.visits + c * math.sqrt(2 * math.log(node.visits + 1) / child.visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


# 判断是否为叶子节点
def is_leaf(node_name):
    return node_name in leaf_results


# 模拟过程
def simulate(node_name):
    # 从当前节点开始模拟，直到到达叶节点
    current_node_name = node_name
    while current_node_name not in leaf_results:
        children = graph[current_node_name]
        if not children:  # 当前节点没有子节点却不是叶节点，则终止
            break
        current_node_name = random.choice(children)  # 随机选择一个子节点
    # 最终返回叶节点的结果
    return leaf_results.get(current_node_name, 'lose')  # 默认返回 'lose' 防止异常



# 可视化树的搜索情况
def visualize_tree(root):
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """利用递归获取树的分层位置"""
        pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        return pos

    def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) or parent is None:
            if root in children:
                children.remove(root)  # 防止自循环
        if parent is not None:
            if parent in children:
                children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                     xcenter=nextx, pos=pos, parent=root, parsed=parsed)
        return pos

    # 构建树状图
    g = nx.DiGraph()
    node_labels = {}
    add_to_graph(g, root, node_labels)

    # 使用层级布局
    pos = hierarchy_pos(g, root=root.name)
    plt.figure(figsize=(12, 8))
    nx.draw(g, pos, with_labels=False, node_size=2000, node_color="skyblue", font_size=10, font_weight="light")
    nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): '' for u, v in g.edges()})
    nx.draw_networkx_labels(g, pos, node_labels, font_size=8)
    plt.show()


def add_to_graph(g, node, node_labels):
    g.add_node(node.name)
    node_labels[node.name] = f'{node.name}\nVisits: {node.visits}\nWins: {node.wins}'
    for child in node.children:
        g.add_edge(node.name, child.name)
        add_to_graph(g, child, node_labels)


# 输出最优的下一步
def get_best_next_move(root):
    best_score = -float('inf')
    best_move = None
    for child in root.children:
        if child.visits > 0:
            score = child.wins / child.visits  # 胜率 = 胜利次数 / 访问次数
            if score > best_score:
                best_score = score
                best_move = child
    return best_move


# 创建搜索树
root = build_tree(graph, 'A')

# 执行 MCTS
mcts(root, 5000)  # 增加迭代次数，进行更精细的统计

# 获取并输出最优的下一步
best_move = get_best_next_move(root)
if best_move:
    print(f"The best next move is: {best_move.name}")
    print(f"Visits: {best_move.visits}, Wins: {best_move.wins}, Win Rate: {best_move.wins / best_move.visits:.2f}")
else:
    print("No valid next move found.")

# 可视化搜索结果
visualize_tree(root)
