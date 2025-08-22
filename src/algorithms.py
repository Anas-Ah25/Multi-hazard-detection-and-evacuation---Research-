"""
Original Algorithm Implementations for Evacuation Pathfinding Simulation
Based on original academic implementations from GitHub repositories
"""

import heapq
import networkx as nx
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

class DijkstraAlgorithm:
    """
    Classic Dijkstra's Algorithm - Original Implementation
    Based on standard academic implementation
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes_explored = 0
        self.computation_time = 0
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using classic Dijkstra's algorithm"""
        start_time = time.time()
        
        # Ensure all edges have weight attribute
        self._ensure_edge_weights()
        
        # Initialize distances: set distance to start = 0, others = infinity
        distances = {vertex: float('inf') for vertex in self.graph.nodes()}
        distances[self.start] = 0
        
        # Priority queue for selecting the vertex with the minimum distance
        pq = [(0, self.start)]  # (distance, vertex)
        
        # Keep track of visited nodes and predecessors
        visited = set()
        predecessors = {}
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
            visited.add(current_vertex)
            self.nodes_explored += 1
            
            # If we reached the target, stop early
            if current_vertex == self.goal:
                break
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_vertex):
                edge_data = self.graph[current_vertex][neighbor]
                weight = self._get_edge_weight(edge_data)
                
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))
        
        self.computation_time += time.time() - start_time
        
        # Reconstruct path
        if self.goal not in predecessors and self.goal != self.start:
            return []
        
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        
        path.reverse()
        return path if path[0] == self.start else []
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """Replan from current position by running Dijkstra again"""
        if current_position >= len(current_path):
            return []
        
        current_node = current_path[current_position]
        
        # Dijkstra replans completely from current position
        temp_start = self.start
        self.start = current_node
        new_path = self.find_initial_path()
        self.start = temp_start
        
        return new_path[1:] if new_path and len(new_path) > 1 else []
    
    def _ensure_edge_weights(self):
        """Ensure all edges have a 'weight' attribute"""
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data:
                # Use enhanced weight calculation if base_time available
                if 'base_time' in data and 'lanes' in data:
                    t_base = data['base_time']
                    lanes = data['lanes']
                    severity = data.get('severity', 0.0)
                    K_hazard = 4.0
                    data['weight'] = (t_base * (1 + K_hazard * severity)) / (lanes + 1)
                else:
                    data['weight'] = data.get('length', 100)
    
    def _get_edge_weight(self, edge_data) -> float:
        """Get edge weight from edge data - unified across all algorithms"""
        if isinstance(edge_data, dict):
            return edge_data.get('weight', edge_data.get('length', 100))
        else:
            # For multigraphs
            for key in edge_data:
                return edge_data[key].get('weight', edge_data[key].get('length', 100))
        return 100


class DStarLiteAlgorithm:
    """
    D* Lite Algorithm - Based on original implementation
    Adapted from: https://github.com/mdeyo/d-star-lite/blob/master/d_star_lite.py
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes_explored = 0
        self.computation_time = 0
        
        # D* Lite specific data structures
        self.g = {}  # g values
        self.rhs = {}  # rhs values
        self.k_m = 0  # key modifier
        self.queue = []  # priority queue
        
        self._initialize()
    
    def _initialize(self):
        """Initialize D* Lite data structures"""
        # Initialize g and rhs values
        for node in self.graph.nodes():
            self.g[node] = float('inf')
            self.rhs[node] = float('inf')
        
        # Goal has rhs = 0
        self.rhs[self.goal] = 0
        
        # Add goal to queue with its key
        key = self._calculate_key(self.goal)
        heapq.heappush(self.queue, key + (self.goal,))
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using D* Lite"""
        start_time = time.time()
        
        # Compute shortest path
        self._compute_shortest_path()
        
        # Extract path
        path = self._extract_path()
        
        self.computation_time += time.time() - start_time
        return path
    
    def _calculate_key(self, node: int) -> Tuple[float, float]:
        """Calculate priority key for a node"""
        min_val = min(self.g[node], self.rhs[node])
        heuristic = self._heuristic(node, self.start)
        return (min_val + heuristic + self.k_m, min_val)
    
    def _heuristic(self, node1: int, node2: int) -> float:
        """Heuristic function (Euclidean distance)"""
        if node1 in self.graph.nodes() and node2 in self.graph.nodes():
            try:
                x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
                x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
                return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            except KeyError:
                # Fallback if coordinates not available
                return 0
        return 0
    
    def _get_edge_cost(self, u: int, v: int) -> float:
        """Get cost of edge between two nodes - unified across all algorithms"""
        if self.graph.has_edge(u, v):
            edge_data = self.graph[u][v]
            if isinstance(edge_data, dict):
                return edge_data.get('weight', edge_data.get('length', 100))
            else:
                for key in edge_data:
                    return edge_data[key].get('weight', edge_data[key].get('length', 100))
        return float('inf')
    
    def _update_vertex(self, node: int):
        """Update vertex in D* Lite algorithm"""
        if node != self.goal:
            min_rhs = float('inf')
            for neighbor in self.graph.neighbors(node):
                cost = self._get_edge_cost(node, neighbor) + self.g[neighbor]
                min_rhs = min(min_rhs, cost)
            self.rhs[node] = min_rhs
        
        # Remove node from queue if present
        self.queue = [item for item in self.queue if item[2] != node]
        heapq.heapify(self.queue)
        
        # Add to queue if inconsistent
        if self.g[node] != self.rhs[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.queue, key + (node,))
    
    def _compute_shortest_path(self):
        """Compute shortest path using D* Lite"""
        start_key = self._calculate_key(self.start)
        
        while (self.queue and 
               (self._top_key() < start_key or self.rhs[self.start] != self.g[self.start])):
            
            k_old = self._top_key()
            u = heapq.heappop(self.queue)[2]
            self.nodes_explored += 1
            
            k_new = self._calculate_key(u)
            
            if k_old < k_new:
                heapq.heappush(self.queue, k_new + (u,))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for predecessor in self.graph.neighbors(u):
                    self._update_vertex(predecessor)
            else:
                self.g[u] = float('inf')
                self._update_vertex(u)
                for predecessor in self.graph.neighbors(u):
                    self._update_vertex(predecessor)
            
            start_key = self._calculate_key(self.start)
    
    def _top_key(self) -> Tuple[float, float]:
        """Get top key from priority queue"""
        if self.queue:
            return self.queue[0][:2]
        return (float('inf'), float('inf'))
    
    def _extract_path(self) -> List[int]:
        """Extract path from start to goal"""
        if self.g[self.start] == float('inf'):
            return []
        
        path = [self.start]
        current = self.start
        
        while current != self.goal:
            min_cost = float('inf')
            next_node = None
            
            for neighbor in self.graph.neighbors(current):
                cost = self._get_edge_cost(current, neighbor) + self.g[neighbor]
                if cost < min_cost:
                    min_cost = cost
                    next_node = neighbor
            
            if next_node is None:
                break
            
            path.append(next_node)
            current = next_node
            
            # Prevent infinite loops
            if len(path) > len(self.graph.nodes()):
                break
        
        return path
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """Replan using D* Lite incremental search"""
        if current_position >= len(current_path):
            return []
        
        start_time = time.time()
        
        # Update affected edges due to hazards
        for hazard in hazards:
            for u, v in hazard.affected_edges:
                if self.graph.has_edge(u, v):
                    self._update_vertex(u)
                    self._update_vertex(v)
        
        # Update k_m for replanning
        current_node = current_path[current_position]
        self.k_m += self._heuristic(self.start, current_node)
        self.start = current_node
        
        # Recompute shortest path
        self._compute_shortest_path()
        
        # Extract new path
        new_path = self._extract_path()
        
        self.computation_time += time.time() - start_time
        return new_path[1:] if new_path and len(new_path) > 1 else []


class SSSPAlgorithm:
    """
    Bounded Multi-Source Shortest Paths (BMSSP) Algorithm
    Based on: https://github.com/DiogoRibeiro7/bmssp/blob/main/implementations/python/bmssp.py
    Original implementation with proper BMSSP algorithm
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes_explored = 0
        self.computation_time = 0
        
        # Convert NetworkX graph to BMSSP format
        self.bmssp_graph = self._convert_to_bmssp_graph()
        
        # BMSSP specific parameters
        n = len(self.graph.nodes())
        self.k = max(1, int(math.log(n, 2) ** (1/3))) if n > 1 else 1
        self.t = max(1, int(math.log(n, 2) ** (2/3))) if n > 1 else 1
        self.l = math.ceil(math.log(n, 2) / self.t) if n > 1 and self.t > 0 else 1
    
    def _convert_to_bmssp_graph(self):
        """Convert NetworkX graph to BMSSP Graph format"""
        bmssp_graph = BMSSPGraph()
        
        for u, v, data in self.graph.edges(data=True):
            weight = self._get_edge_weight(data)
            bmssp_graph.add_edge(u, v, weight)
            bmssp_graph.add_edge(v, u, weight)  # For undirected graph
        
        return bmssp_graph
    
    def _get_edge_weight(self, edge_data) -> float:
        """Get edge weight from edge data - unified across all algorithms"""
        if isinstance(edge_data, dict):
            return edge_data.get('weight', edge_data.get('length', 100))
        else:
            # For multigraphs
            for key in edge_data:
                return edge_data[key].get('weight', edge_data[key].get('length', 100))
        return 100
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using BMSSP algorithm"""
        start_time = time.time()
        
        # Run BMSSP algorithm
        distances = self._run_bmssp()
        
        # Reconstruct path using the distances
        path = self._reconstruct_path_from_distances(distances)
        
        self.computation_time += time.time() - start_time
        return path
    
    def _run_bmssp(self) -> Dict[int, float]:
        """Run the BMSSP algorithm with fallback to ensure connectivity"""
        n = len(self.bmssp_graph.adj)
        k = max(1, int(math.log(n, 2) ** (1/3))) if n > 1 else 1
        t = max(1, int(math.log(n, 2) ** (2/3))) if n > 1 else 1
        l = math.ceil(math.log(n, 2) / t) if n > 1 and t > 0 else 1
        
        # For smaller graphs, use more conservative parameters
        if n < 50000:
            k = max(k, 10)  # Ensure minimum exploration
            t = max(t, 3)
            l = max(l, 5)
        
        d_hat = {v: math.inf for v in self.bmssp_graph.adj}
        d_hat[self.start] = 0.0
        complete = {self.start}
        
        try:
            # Run BMSSP
            bmssp(self.bmssp_graph, l, math.inf, {self.start}, d_hat, complete, k, t)
            
            # Check if goal was reached
            if d_hat[self.goal] == math.inf:
                print(f"BMSSP failed to reach goal, using Dijkstra fallback")
                return self._fallback_dijkstra()
            
            # Count explored nodes
            self.nodes_explored = len(complete)
            return d_hat
            
        except Exception as e:
            print(f"BMSSP error: {e}, using Dijkstra fallback")
            return self._fallback_dijkstra()
    
    def _fallback_dijkstra(self) -> Dict[int, float]:
        """Fallback to Dijkstra when BMSSP fails"""
        distances = {v: math.inf for v in self.bmssp_graph.adj}
        distances[self.start] = 0.0
        self.predecessors = {}
        
        pq = [(0, self.start)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.nodes_explored += 1
            
            if current == self.goal:
                break
            
            for neighbor, weight in self.bmssp_graph.neighbors(current):
                new_distance = current_dist + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    self.predecessors[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        return distances
    
    def _reconstruct_path_from_distances(self, distances: Dict[int, float]) -> List[int]:
        """Reconstruct path from distances using greedy approach or predecessors"""
        if self.goal not in distances or distances[self.goal] == math.inf:
            return []
        
        # If we have predecessors (from fallback Dijkstra), use them
        if hasattr(self, 'predecessors') and self.goal in self.predecessors:
            path = []
            current = self.goal
            while current is not None:
                path.append(current)
                current = self.predecessors.get(current)
            path.reverse()
            return path if path and path[0] == self.start else []
        
        # Otherwise use greedy reconstruction from distances
        path = [self.start]
        current = self.start
        
        while current != self.goal:
            best_neighbor = None
            best_distance = math.inf
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in distances:
                    neighbor_dist = distances[neighbor]
                    edge_weight = self._get_edge_weight(self.graph[current][neighbor])
                    
                    # Check if this creates a valid shortest path step
                    if (abs(distances[current] + edge_weight - neighbor_dist) < 1e-9 and 
                        neighbor_dist < best_distance):
                        best_neighbor = neighbor
                        best_distance = neighbor_dist
            
            if best_neighbor is None:
                # Fallback: choose closest neighbor to goal
                for neighbor in self.graph.neighbors(current):
                    if neighbor in distances and distances[neighbor] < best_distance:
                        best_neighbor = neighbor
                        best_distance = distances[neighbor]
            
            if best_neighbor is None:
                break
            
            path.append(best_neighbor)
            current = best_neighbor
            
            # Prevent infinite loops
            if len(path) > len(self.graph.nodes()):
                break
        
        return path if path[-1] == self.goal else []
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """Replan using BMSSP from current position"""
        if current_position >= len(current_path):
            return []
        
        current_node = current_path[current_position]
        
        # Update start position and recompute
        temp_start = self.start
        self.start = current_node
        
        # Re-convert graph to account for any changes
        self.bmssp_graph = self._convert_to_bmssp_graph()
        
        new_path = self.find_initial_path()
        self.start = temp_start
        
        return new_path[1:] if new_path and len(new_path) > 1 else []


# BMSSP Support Classes and Functions
class BMSSPGraph:
    """Graph class for BMSSP algorithm"""
    
    def __init__(self):
        self.adj = {}
    
    def add_edge(self, u, v, w):
        if w < 0:
            raise ValueError("Only non-negative weights allowed")
        self.adj.setdefault(u, []).append((v, float(w)))
        self.adj.setdefault(v, self.adj.get(v, []))
    
    def neighbors(self, u):
        return self.adj.get(u, [])


class DQueue:
    """DQueue implementation for BMSSP"""
    
    def __init__(self, M: int, B: float):
        self.M = M
        self.B = B
        self.data = []
        self.prepend = []
    
    def insert(self, node, dist: float):
        if dist >= self.B:
            return
        heapq.heappush(self.data, (dist, node))
    
    def batch_prepend(self, items):
        self.prepend.extend((dist, node) for node, dist in items if dist < self.B)
    
    def pull(self):
        if self.prepend:
            group = self.prepend[:self.M]
            self.prepend = self.prepend[self.M:]
            B_i = group[0][0] if group else self.B
            return B_i, {node for _, node in group}
        
        if not self.data:
            return self.B, set()
        
        out = []
        while self.data and len(out) < self.M:
            dist, node = heapq.heappop(self.data)
            out.append((dist, node))
        B_i = out[0][0] if out else self.B
        return B_i, {node for _, node in out}
    
    def non_empty(self):
        return bool(self.data or self.prepend)


def find_pivots(graph, B: float, S, d_hat, complete, k: int):
    """Find pivots implementation for BMSSP"""
    W = set(S)
    W_prev = set(S)
    
    for _ in range(1, k + 1):
        W_i = set()
        for u in W_prev:
            du = d_hat[u]
            if du >= B:
                continue
            for v, w_uv in graph.neighbors(u):
                if du + w_uv <= d_hat[v]:
                    d_hat[v] = du + w_uv
                    if du + w_uv < B:
                        W_i.add(v)
        W |= W_i
        if len(W) > k * len(S):
            return set(S), W
        W_prev = W_i
    
    F_children = defaultdict(list)
    indeg = {u: 0 for u in W}
    for u in W:
        du = d_hat[u]
        for v, w_uv in graph.neighbors(u):
            if v in W and abs(d_hat[v] - (du + w_uv)) < 1e-12:
                F_children[u].append(v)
                indeg[v] += 1
    
    def subtree_size(u):
        size = 1
        for child in F_children.get(u, []):
            size += subtree_size(child)
        return size
    
    P = {u for u in S if indeg.get(u, 0) == 0 and subtree_size(u) >= k}
    return P, W


def base_case(graph, B: float, S, d_hat, complete, k: int):
    """Base case implementation for BMSSP"""
    x = next(iter(S))
    U0 = {x}
    H = [(d_hat[x], x)]
    visited = set()
    
    while H and len(U0) < k + 1:
        du, u = heapq.heappop(H)
        if u in visited:
            continue
        visited.add(u)
        U0.add(u)
        complete.add(u)
        for v, w_uv in graph.neighbors(u):
            if du + w_uv <= d_hat[v] and du + w_uv < B:
                d_hat[v] = du + w_uv
                heapq.heappush(H, (d_hat[v], v))
    
    if len(U0) <= k:
        return B, U0
    else:
        B_prime = max(d_hat[v] for v in U0)
        return B_prime, {v for v in U0 if d_hat[v] < B_prime}


def bmssp(graph, l: int, B: float, S, d_hat, complete, k: int, t: int):
    """Main BMSSP algorithm implementation"""
    if l == 0:
        return base_case(graph, B, S, d_hat, complete, k)
    
    P, W = find_pivots(graph, B, S, d_hat, complete, k)
    M = 2 ** ((l - 1) * t)
    D = DQueue(M, B)
    for x in P:
        D.insert(x, d_hat[x])
    
    U = set()
    B0_prime = min(d_hat[x] for x in P) if P else B
    
    while len(U) < k * (2 ** (l * t)) and D.non_empty():
        Bi, Si = D.pull()
        B_prime_i, U_i = bmssp(graph, l - 1, Bi, Si, d_hat, complete, k, t)
        U |= U_i
        K = []
        for u in U_i:
            for v, w_uv in graph.neighbors(u):
                if d_hat[u] + w_uv <= d_hat[v]:
                    d_hat[v] = d_hat[u] + w_uv
                    if Bi <= d_hat[v] < B:
                        D.insert(v, d_hat[v])
                    elif B_prime_i <= d_hat[v] < Bi:
                        K.append((v, d_hat[v]))
        prepend_items = K + [(x, d_hat[x]) for x in Si if B_prime_i <= d_hat[x] < Bi]
        D.batch_prepend(prepend_items)
    
    B_prime = min(B0_prime, B)
    U |= {x for x in W if d_hat[x] < B_prime}
    complete |= U
    return B_prime, U
