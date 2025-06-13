import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 在plt.show()前设置后端
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei"]


def create_ieee24_graph():
    """创建IEEE 24节点可靠性测试系统图"""
    G = nx.Graph()

    # IEEE 24节点系统节点列表
    nodes = list(range(1, 25))

    # 节点类型（1-5: 发电节点, 6-10: 大型负荷节点, 11-24: 中小型负荷节点）
    node_types = {
        '发电节点': [1, 2, 3, 4, 5],
        '大型负荷节点': [6, 7, 8, 9, 10],
        '中小型负荷节点': list(range(11, 25))
    }

    # 为每种节点类型分配不同的颜色
    node_colors = {
        '发电节点': '#FF5733',  # 橙色
        '大型负荷节点': '#3366FF',  # 蓝色
        '中小型负荷节点': '#33FF57'  # 绿色
    }

    # 节点坐标（手动调整以获得更好的可视化效果）
    pos = {
        1: (1, 5), 2: (2, 5), 3: (3, 5), 4: (4, 5), 5: (5, 5),
        6: (1, 3), 7: (2, 3), 8: (3, 3), 9: (4, 3), 10: (5, 3),
        11: (1, 1), 12: (2, 1), 13: (3, 1), 14: (4, 1), 15: (5, 1),
        16: (6, 1), 17: (7, 1), 18: (8, 1), 19: (9, 1), 20: (10, 1),
        21: (6, 3), 22: (7, 3), 23: (8, 3), 24: (9, 3)
    }

    # 添加节点及其属性
    for node in nodes:
        node_type = None
        for type_name, type_nodes in node_types.items():
            if node in type_nodes:
                node_type = type_name
                break
        G.add_node(node, type=node_type, color=node_colors[node_type])

    # IEEE 24节点系统的边（输电线路）
    edges = [
        (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9),
        (5, 9), (5, 10), (6, 7), (7, 8), (8, 9), (9, 10),
        (6, 11), (6, 12), (7, 12), (7, 13), (8, 13), (8, 14),
        (9, 14), (9, 15), (10, 15), (10, 16), (11, 12), (12, 13),
        (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
        (19, 20), (16, 21), (17, 21), (17, 22), (18, 22), (18, 23),
        (19, 23), (19, 24), (20, 24), (21, 22), (22, 23), (23, 24)
    ]

    # 为每条边添加容量属性（基于IEEE RTS-24标准值）
    # 实际应用中应根据具体研究调整这些值
    line_capacities = {
        (1, 6): 1000, (2, 6): 1000, (2, 7): 1000, (3, 7): 1000,
        (3, 8): 1000, (4, 8): 1000, (4, 9): 1000, (5, 9): 1000,
        (5, 10): 1000, (6, 7): 800, (7, 8): 800, (8, 9): 800,
        (9, 10): 800, (6, 11): 600, (6, 12): 600, (7, 12): 600,
        (7, 13): 600, (8, 13): 600, (8, 14): 600, (9, 14): 600,
        (9, 15): 600, (10, 15): 600, (10, 16): 600, (11, 12): 400,
        (12, 13): 400, (13, 14): 400, (14, 15): 400, (15, 16): 400,
        (16, 17): 600, (17, 18): 600, (18, 19): 600, (19, 20): 600,
        (16, 21): 600, (17, 21): 600, (17, 22): 600, (18, 22): 600,
        (18, 23): 600, (19, 23): 600, (19, 24): 600, (20, 24): 600,
        (21, 22): 400, (22, 23): 400, (23, 24): 400
    }

    # 添加边及其容量属性
    for u, v in edges:
        # 确保边的容量在两个方向上相同
        capacity = line_capacities.get((u, v)) or line_capacities.get((v, u))
        G.add_edge(u, v, capacity=capacity)

    return G, pos


def plot_power_system(G, pos, title="IEEE 24节点可靠性测试系统"):
    """绘制电力系统图"""
    plt.figure(figsize=(15, 10))

    # 获取节点颜色
    colors = [G.nodes[node]['color'] for node in G.nodes()]

    # 获取节点标签
    labels = {node: node for node in G.nodes()}

    # 获取边的容量作为权重
    weights = [G[u][v]['capacity'] / 200 for u, v in G.edges()]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=colors, alpha=0.8)

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='#999999', alpha=0.6)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_family="SimHei")

    # 绘制边标签（显示容量）
    edge_labels = {(u, v): f"{G[u][v]['capacity']}MW" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # 设置标题和坐标轴
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=node_type)
                       for node_type, color in {
                           '发电节点': '#FF5733',
                           '大型负荷节点': '#3366FF',
                           '中小型负荷节点': '#33FF57'
                       }.items()]

    plt.legend(handles=legend_elements, loc='upper right')

    return plt


# 创建并绘制IEEE 24节点系统图
G, pos = create_ieee24_graph()
plt = plot_power_system(G, pos)

# 显示图形
plt.show()