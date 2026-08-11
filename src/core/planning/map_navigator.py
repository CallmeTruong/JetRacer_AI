import os
import json
import networkx as nx

class ShortestPathFinder:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.start_node = self._find_node_by_type('start')
        self.end_node = self._find_node_by_type('end')
        self.graph = self._build_graph()  # {u: {v: (weight, label)}}
        self.dinh = self._convert_edges_to_nodes()

    def _find_node_by_type(self, node_type):
        for node in self.data['nodes']:
            if node['type'] == node_type:
                return node['id']
        return None

    def _build_graph(self):
        graph = {}
        for edge in self.data['edges']:
            u = edge['source']
            v = edge['target']
            w = 1  # Trọng số mặc định, có thể thay đổi nếu cần
            if u not in graph:
                graph[u] = {}
            graph[u][v] = (w, edge['label'])
        return graph

    def _convert_edges_to_nodes(self):
        cost_xoay = 1
        dinh = {}
        graph = self.graph
        for edge in self.data['edges']:
            u = edge['source']
            v = edge['target']
            w = graph[u][v][0]
            huong_u_v = graph[u][v][1]
            for v2, value in graph[v].items():
                if v2 == u: continue
                huong_v_v2 = value[1]
                cost = value[0]
                if huong_u_v != huong_v_v2:
                    cost += cost_xoay
                if (u, v) not in dinh: dinh[(u, v)] = {}
                dinh[(u, v)][(v, v2)] = cost
        return dinh

    def convert_to_graph(self, dinh):
        G = nx.DiGraph()
        for u in dinh:
            for v in dinh[u]:
                cost = dinh[u][v]
                G.add_edge(u, v, weight=cost)
        return G

    def shortest_path(self, start, end):
        G = self.convert_to_graph(self.dinh)
        try:
            path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
            cost = nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
            return path, cost
        except nx.NetworkXNoPath:
            return None, float('inf')

    def find(self, start_node=None, end_node=None):
        # Trả về đường đi ngắn nhất từ start_node đến end_node
        if start_node is None:
            start_node = self.start_node
        if end_node is None:
            end_node = self.end_node
        # Tìm các node kề để tạo tuple (start, next_node), (end, next_node2)
        res_cost = float('inf')
        result_path = []
        for u in self.graph[start_node].keys():
            bd = (start_node, u)
            for v in self.graph[end_node].keys():
                kt = (v, end_node)
                path, cost = self.shortest_path(bd, kt)
                cost += self.graph[start_node][u][0]
                if cost < res_cost:
                    result_path = path
                    res_cost = cost
        result = []
        for u, v in result_path:
            result.append((u, v, self.graph[u][v][1]))

        return result, res_cost


class MapNavigator:
    """Compatibility wrapper so older code importing MapNavigator keeps working.

    This wraps ShortestPathFinder and provides:
    - start_node, end_node properties
    - find_path(start, end, banned_edges=None) -> list of node ids (like original)
    - get_next_direction_label(current_node_id, path)
    - get_neighbor_by_direction(current_node_id, direction_label)
    """
    def __init__(self, json_path):
        self.finder = ShortestPathFinder(json_path)
        # build node lookup
        self.node_lookup = {n['id']: n for n in self.finder.data.get('nodes', [])}

    @property
    def start_node(self):
        return self.finder.start_node

    @property
    def end_node(self):
        return self.finder.end_node

    def _build_nx_graph(self):
        G = nx.DiGraph()
        for u, nbrs in self.finder.graph.items():
            for v, (w, label) in nbrs.items():
                G.add_edge(u, v, weight=w, label=label)
        return G

    def _heuristic(self, a, b):
        # Use Euclidean distance on node coordinates if available
        na = self.node_lookup.get(a)
        nb = self.node_lookup.get(b)
        if not na or not nb:
            return 0
        try:
            dx = na.get('x', 0) - nb.get('x', 0)
            dy = na.get('y', 0) - nb.get('y', 0)
            return (dx*dx + dy*dy) ** 0.5
        except Exception:
            return 0

    def find_path(self, start, end, banned_edges=None):
        """Return list of node ids from start to end. Honors banned_edges if provided."""
        G = self._build_nx_graph()
        if banned_edges:
            try:
                G.remove_edges_from(banned_edges)
            except Exception:
                pass
        try:
            path = nx.astar_path(G, source=start, target=end, heuristic=self._heuristic)
            return path
        except Exception:
            return None

    def get_next_direction_label(self, current_node_id, path):
        """Given a node path (list of node ids), return the edge label from current to next."""
        if not path or current_node_id not in path:
            return None
        idx = path.index(current_node_id)
        if idx + 1 >= len(path):
            return None
        next_node = path[idx + 1]
        nbrs = self.finder.graph.get(current_node_id, {})
        entry = nbrs.get(next_node)
        if entry:
            return entry[1]
        return None

    def get_neighbor_by_direction(self, current_node_id, direction_label):
        for v, (w, label) in self.finder.graph.get(current_node_id, {}).items():
            if label == direction_label:
                return v
        return None

if __name__ == "__main__":
    # Sử dụng file map.json đã cung cấp
    finder = ShortestPathFinder(json_path="d:/basic_Python/traning-12-jetbot-nav/map.json")
    path, cost = finder.find()
    print(f"Đường đi ngắn nhất từ {finder.start_node} đến {finder.end_node}: {path}")
    print(f"Tổng trọng số: {cost}")

#    Đường đi ngắn nhất từ 2 đến 11: [(2, 1), (1, 4), (4, 8), (8, 11)]
#Tổng trọng số: 5