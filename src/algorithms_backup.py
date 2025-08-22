"""
Pathfinding Algorithm Implementations for Evacuation Simulation
Includes Dijkstra, D* Lite, and Single-Source Shortest Path (SSSP) algorithms
"""

import networkx as nx
import heapq
import time
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

class PathfindingAlgorithm(ABC):
    """Abstract base class for pathfinding algorithms"""
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.nodes_explored = 0
        self.computation_time = 0
    
    @abstractmethod
    def find_initial_path(self) -> List[int]:
        """Find initial path from start to goal"""
        pass
    
    @abstractmethod
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """Replan path from current position considering new hazards"""
        pass

class DijkstraAlgorithm(PathfindingAlgorithm):
    """
    Dijkstra's algorithm implementation
    - Explores all nodes systematically
    - Guarantees optimal path
    - Replans from scratch when hazards detected
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        super().__init__(graph, start, goal)
        self.algorithm_name = "Dijkstra"
    
    def find_initial_path(self) -> List[int]:
        """Find initial shortest path using Dijkstra's algorithm"""
        start_time = time.time()
        
        # Ensure all edges have weight attribute for NetworkX
        self._ensure_edge_weights()
        
        try:
            path = nx.shortest_path(self.graph, self.start, self.goal, weight='weight')
            self.nodes_explored = len(self.graph.nodes())  # Dijkstra explores all reachable nodes
            self.computation_time += time.time() - start_time
            return path
        except nx.NetworkXNoPath:
            self.computation_time += time.time() - start_time
            return []
    
    def _ensure_edge_weights(self):
        """Ensure all edges have a 'weight' attribute for NetworkX algorithms"""
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data:
                data['weight'] = data.get('length', 100)
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """
        Replan from current position by running Dijkstra again
        Dijkstra replans from scratch, exploring the entire graph
        """
        if current_position >= len(current_path):
            return []
        
        start_time = time.time()
        current_node = current_path[current_position]
        
        # Ensure all edges have weight attribute
        self._ensure_edge_weights()
        
        try:
            # Dijkstra replans completely from current position
            new_path = nx.shortest_path(self.graph, current_node, self.goal, weight='weight')
            
            # Dijkstra explores many nodes during replanning
            reachable_nodes = len(nx.single_source_shortest_path_length(self.graph, current_node))
            self.nodes_explored += reachable_nodes
            
            self.computation_time += time.time() - start_time
            return new_path
        except nx.NetworkXNoPath:
            self.computation_time += time.time() - start_time
            return []

class DStarLiteAlgorithm(PathfindingAlgorithm):
    """
    D* Lite algorithm implementation
    - Incremental pathfinding algorithm
    - Reuses previous computations when replanning
    - More efficient for dynamic environments with frequent changes
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        super().__init__(graph, start, goal)
        self.algorithm_name = "D* Lite"
        
        # D* Lite specific data structures
        self.g = defaultdict(lambda: float('inf'))  # Cost from start
        self.rhs = defaultdict(lambda: float('inf'))  # One-step lookahead cost
        self.open_list = []
        self.key_modifier = 0
        self.last_start = start
        
        # Initialize
        self.rhs[self.goal] = 0
        heapq.heappush(self.open_list, (self._calculate_key(self.goal), self.goal))
    
    def _calculate_key(self, node: int) -> Tuple[float, float]:
        """Calculate priority key for D* Lite"""
        min_g_rhs = min(self.g[node], self.rhs[node])
        h_cost = self._heuristic(node, self.start)
        return (min_g_rhs + h_cost + self.key_modifier, min_g_rhs)
    
    def _heuristic(self, node1: int, node2: int) -> float:
        """Euclidean distance heuristic"""
        if node1 not in self.graph.nodes or node2 not in self.graph.nodes:
            return float('inf')
        
        x1, y1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
        x2, y2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node"""
        return list(self.graph.neighbors(node))
    
    def _get_edge_cost(self, u: int, v: int) -> float:
        """Get cost of edge between two nodes"""
        if self.graph.has_edge(u, v):
            edge_data = self.graph[u][v]
            # Handle both simple graphs and multigraphs
            if isinstance(edge_data, dict):
                return edge_data.get('weight', edge_data.get('length', 100))
            else:
                # For multigraphs, get the first edge's weight
                for key in edge_data:
                    return edge_data[key].get('weight', edge_data[key].get('length', 100))
        return float('inf')
    
    def _update_node(self, node: int):
        """Update node in D* Lite algorithm"""
        if node != self.goal:
            min_rhs = float('inf')
            for neighbor in self._get_neighbors(node):
                cost = self._get_edge_cost(node, neighbor) + self.g[neighbor]
                min_rhs = min(min_rhs, cost)
            self.rhs[node] = min_rhs
        
        # Remove node from open list if present
        self.open_list = [(key, n) for key, n in self.open_list if n != node]
        heapq.heapify(self.open_list)
        
        if self.g[node] != self.rhs[node]:
            heapq.heappush(self.open_list, (self._calculate_key(node), node))
    
    def _compute_shortest_path(self):
        """Main computation loop of D* Lite"""
        start_key = self._calculate_key(self.start)
        
        while (self.open_list and 
               (self.open_list[0][0] < start_key or self.rhs[self.start] != self.g[self.start])):
            
            if not self.open_list:
                break
                
            k_old, current = heapq.heappop(self.open_list)
            k_new = self._calculate_key(current)
            
            self.nodes_explored += 1
            
            if k_old < k_new:
                heapq.heappush(self.open_list, (k_new, current))
            elif self.g[current] > self.rhs[current]:
                self.g[current] = self.rhs[current]
                for neighbor in self._get_neighbors(current):
                    self._update_node(neighbor)
            else:
                self.g[current] = float('inf')
                for neighbor in self._get_neighbors(current) + [current]:
                    self._update_node(neighbor)
            
            start_key = self._calculate_key(self.start)
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using D* Lite"""
        start_time = time.time()
        
        self._compute_shortest_path()
        
        if self.g[self.start] == float('inf'):
            self.computation_time += time.time() - start_time
            return []
        
        # Extract path
        path = []
        current = self.start
        
        while current != self.goal:
            path.append(current)
            
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in self._get_neighbors(current):
                cost = self._get_edge_cost(current, neighbor) + self.g[neighbor]
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                self.computation_time += time.time() - start_time
                return []
            
            current = best_neighbor
        
        path.append(self.goal)
        self.computation_time += time.time() - start_time
        return path
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """
        Replan using D* Lite incremental approach
        Only updates affected areas rather than replanning from scratch
        """
        if current_position >= len(current_path):
            return []
        
        start_time = time.time()
        current_node = current_path[current_position]
        
        # Update start position
        old_start = self.last_start
        self.last_start = current_node
        self.start = current_node
        
        # Update key modifier for new start
        if old_start != current_node:
            self.key_modifier += self._heuristic(old_start, current_node)
        
        # Update affected edges due to hazards
        for hazard in hazards:
            for u, v in hazard.affected_edges:
                if self.graph.has_edge(u, v):
                    # Edge cost changed, update affected nodes
                    self._update_node(u)
                    self._update_node(v)
        
        # Recompute shortest path incrementally
        self._compute_shortest_path()
        
        if self.g[self.start] == float('inf'):
            self.computation_time += time.time() - start_time
            return []
        
        # Extract new path from current position
        path = []
        current = self.start
        
        while current != self.goal:
            path.append(current)
            
            best_neighbor = None
            best_cost = float('inf')
            
            for neighbor in self._get_neighbors(current):
                cost = self._get_edge_cost(current, neighbor) + self.g[neighbor]
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                self.computation_time += time.time() - start_time
                return []
            
            current = best_neighbor
        
        path.append(self.goal)
        self.computation_time += time.time() - start_time
        return path

class SSSPAlgorithm(PathfindingAlgorithm):
    """
    Single-Source Shortest Path (SSSP) algorithm
    - Computes shortest paths from current position to all reachable nodes
    - More comprehensive exploration than Dijkstra for specific scenarios
    - Maintains distance information for efficient replanning
    """
    
    def __init__(self, graph: nx.Graph, start: int, goal: int):
        super().__init__(graph, start, goal)
        self.algorithm_name = "SSSP"
        self.distance_cache = {}
        self.predecessor_cache = {}
    
    def _single_source_shortest_path(self, source: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Compute shortest paths from source to all reachable nodes"""
        distances = defaultdict(lambda: float('inf'))
        predecessors = {}
        distances[source] = 0
        
        # Priority queue: (distance, node)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.nodes_explored += 1
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                # Get edge weight with proper handling for OSMnx graphs
                edge_data = self.graph[current][neighbor]
                if isinstance(edge_data, dict):
                    edge_weight = edge_data.get('weight', edge_data.get('length', 100))
                else:
                    # For multigraphs, get the first edge's weight
                    for key in edge_data:
                        edge_weight = edge_data[key].get('weight', edge_data[key].get('length', 100))
                        break
                
                new_distance = current_dist + edge_weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))
        
        return dict(distances), predecessors
    
    def _reconstruct_path(self, predecessors: Dict[int, int], start: int, goal: int) -> List[int]:
        """Reconstruct path from predecessors"""
        if goal not in predecessors and goal != start:
            return []
        
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
            
            if current == start:
                path.append(start)
                break
        
        return path[::-1] if path and path[-1] == start else []
    
    def find_initial_path(self) -> List[int]:
        """Find initial path using SSSP"""
        start_time = time.time()
        
        distances, predecessors = self._single_source_shortest_path(self.start)
        
        # Cache results for potential reuse
        self.distance_cache[self.start] = distances
        self.predecessor_cache[self.start] = predecessors
        
        path = self._reconstruct_path(predecessors, self.start, self.goal)
        
        self.computation_time += time.time() - start_time
        return path
    
    def replan(self, current_path: List[int], current_position: int, hazards) -> List[int]:
        """
        Replan using SSSP from current position
        SSSP explores extensively to find robust alternative paths
        """
        if current_position >= len(current_path):
            return []
        
        start_time = time.time()
        current_node = current_path[current_position]
        
        # SSSP recomputes from current position, exploring all reachable nodes
        distances, predecessors = self._single_source_shortest_path(current_node)
        
        # Update cache
        self.distance_cache[current_node] = distances
        self.predecessor_cache[current_node] = predecessors
        
        path = self._reconstruct_path(predecessors, current_node, self.goal)
        
        self.computation_time += time.time() - start_time
        return path

# Algorithm factory
def create_algorithm(algorithm_name: str, graph: nx.Graph, start: int, goal: int) -> PathfindingAlgorithm:
    """Factory function to create algorithm instances"""
    algorithms = {
        'Dijkstra': DijkstraAlgorithm,
        'D* Lite': DStarLiteAlgorithm,
        'SSSP': SSSPAlgorithm
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return algorithms[algorithm_name](graph, start, goal)
