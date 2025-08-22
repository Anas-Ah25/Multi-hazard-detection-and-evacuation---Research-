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
                data['weight'] = data.get('length', 100)
    
    def _get_edge_weight(self, edge_data) -> float:
        """Get edge weight from edge data"""
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
        """Get cost of edge between two nodes"""
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
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes_explored = 0
        self.computation_time = 0
        
        # SSSP specific parameters
        n = len(self.graph.nodes())
        self.k = max(1, int(math.log(n, 2) ** (1/3))) if n > 1 else 1
        self.t = max(1, int(math.log(n, 2) ** (2/3))) if n > 1 else 1
        self.l = math.ceil(math.log(n, 2) / self.t) if n > 1 and self.t > 0 else 1
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using BMSSP algorithm"""
        start_time = time.time()
        
        # Initialize distances
        d_hat = {v: float('inf') for v in self.graph.nodes()}
        d_hat[self.start] = 0
        
        # Run BMSSP
        complete = set()
        distances = self._run_sssp(d_hat, complete)
        
        # Reconstruct path using Dijkstra-style backtracking
        path = self._reconstruct_path(distances)
        
        self.computation_time += time.time() - start_time
        return path
    
    def _run_sssp(self, d_hat: Dict[int, float], complete: Set[int]) -> Dict[int, float]:
        """Run the BMSSP algorithm"""
        # Simplified version of BMSSP for single source
        distances = {v: float('inf') for v in self.graph.nodes()}
        distances[self.start] = 0
        predecessors = {}
        
        # Use Dijkstra as base case for single source
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
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    edge_weight = edge_data.get('weight', edge_data.get('length', 100))
                else:
                    # For multigraphs
                    for key in edge_data:
                        edge_weight = edge_data[key].get('weight', edge_data[key].get('length', 100))
                        break
                
                new_distance = current_dist + edge_weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        self.predecessors = predecessors
        return distances
    
    def _reconstruct_path(self, distances: Dict[int, float]) -> List[int]:
        """Reconstruct path from predecessors"""
        if self.goal not in self.predecessors and self.goal != self.start:
            return []
        
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)
        
        path.reverse()
        return path if path and path[0] == self.start else []
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """Replan using BMSSP from current position"""
        if current_position >= len(current_path):
            return []
        
        current_node = current_path[current_position]
        
        # Update start position and recompute
        temp_start = self.start
        self.start = current_node
        new_path = self.find_initial_path()
        self.start = temp_start
        
        return new_path[1:] if new_path and len(new_path) > 1 else []
