"""
Enhanced Evacuation Pathfinding Simulation with Real Map Data
Uses OSMnx to fetch real street networks and compares pathfinding algorithms
with unified hazard injection and detailed performance analysis.
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import folium
import geopandas as gpd
from matplotlib.animation import FuncAnimation
import seaborn as sns
from tqdm import tqdm

# Configuration
START_COORDS = (48.876876, 2.338250)  # Near Rue de la Chaussée-d'Antin
GOAL_COORDS = (48.872617, 2.333496)   # Near Place de la Concorde

@dataclass
class HazardEvent:
    """Represents a hazard event in the simulation with realistic hazard types"""
    event_id: int
    step: int
    affected_nodes: List[int]
    affected_edges: List[Tuple[int, int]]
    hazard_type: str  # 'fire', 'flood', 'traffic_incident', 'building_collapse'
    severity: float  # 0, 2, or 3 based on hazard type
    duration: int
    timestamp: float
    
    @property
    def severity_grade(self) -> float:
        """Get severity grade based on hazard type"""
        hazard_severities = {
            'fire': 3.0,              # Fire class - grade 3
            'flood': 3.0,             # Flooded areas - grade 3
            'traffic_incident': 2.0,  # Traffic incident - grade 2
            'building_collapse': 2.0  # Collapsed building - grade 2
        }
        return hazard_severities.get(self.hazard_type, 0.0)

@dataclass
class ReplanningEvent:
    """Records details of a replanning event"""
    algorithm: str
    hazard_event_id: int
    step: int
    progress_percentage: float
    hazards_injected: int
    replanning_success: bool
    response_time: float
    new_path_length: int
    old_path: List[int]
    new_path: List[int]
    replanning_trigger: str

@dataclass
class AlgorithmResult:
    """Final results for an algorithm"""
    algorithm: str
    success: bool
    failure_reason: str
    path_length: int
    path_cost: float
    execution_time: float
    nodes_explored: int
    replanning_count: int
    hazards_encountered: int
    total_replanning_events: int
    hazard_response_count: int
    final_path: List[int]

class EvacuationEnvironment:
    """Unified simulation environment for all algorithms"""
    
    def __init__(self, start_coords: Tuple[float, float], goal_coords: Tuple[float, float]):
        self.start_coords = start_coords
        self.goal_coords = goal_coords
        self.graph = None
        self.start_node = None
        self.goal_node = None
        self.hazard_events: List[HazardEvent] = []
        self.current_step = 0
        self.active_hazards: Dict[int, HazardEvent] = {}
        
        # Load real map data
        self._load_map_data()
        
    def _load_map_data(self):
        """Load real street network data using OSMnx with a radius around start/goal"""
        print("Loading real map data from OpenStreetMap...")
        
        # Calculate center point between start and goal
        center_lat = (self.start_coords[0] + self.goal_coords[0]) / 2
        center_lon = (self.start_coords[1] + self.goal_coords[1]) / 2
        center_point = (center_lat, center_lon)
        
        # Calculate distance between start and goal
        from math import radians, sin, cos, sqrt, atan2
        
        # Haversine formula to calculate distance
        lat1, lon1 = radians(self.start_coords[0]), radians(self.start_coords[1])
        lat2, lon2 = radians(self.goal_coords[0]), radians(self.goal_coords[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371000 * c  # Earth radius in meters
        
        # Use radius that's 1.5x the distance between points (minimum 2000m, maximum 5000m)
        radius = max(2000, min(5000, int(distance * 1.5)))
        
        print(f"Using center point: {center_point}")
        print(f"Distance between start/goal: {distance:.0f}m")
        print(f"Download radius: {radius}m")
        
        try:
            # Download street network using radius around center point
            self.graph = ox.graph_from_point(center_point, dist=radius, 
                                           network_type='drive', simplify=True)
            
            # Convert to undirected for pathfinding
            self.graph = ox.convert.to_undirected(self.graph)
            
            # Find nearest nodes to start and goal coordinates
            self.start_node = ox.nearest_nodes(self.graph, self.start_coords[1], self.start_coords[0])
            self.goal_node = ox.nearest_nodes(self.graph, self.goal_coords[1], self.goal_coords[0])
            
            print(f"Map loaded successfully!")
            print(f"Network: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            print(f"Start node: {self.start_node}, Goal node: {self.goal_node}")
            
        except Exception as e:
            print(f"Failed to download map data: {e}")
            print("Creating fallback synthetic graph...")
            self._create_fallback_graph()
            return
        
        # Add edge weights based on length with enhanced hazard modeling
        for u, v, data in self.graph.edges(data=True):
            if 'length' not in data:
                data['length'] = 100  # Default length if missing
            
            # Enhanced weight calculation: w(e) = (t_base * (1 + K_hazard * s(e))) / (lanes(e) + 1)
            t_base = data['length'] / 50.0  # Base time: assume 50 km/h average speed
            lanes = data.get('lanes', 2)  # Default 2 lanes if not specified
            
            # Ensure lanes is numeric and within reasonable bounds (1-12)
            try:
                lanes = int(float(lanes)) if lanes else 2
                lanes = max(1, min(12, lanes))
            except (ValueError, TypeError):
                lanes = 2
            
            # Initialize hazard severity to 0 (no hazard)
            severity = 0.0
            K_hazard = 4.0  # Penalty multiplier
            
            # Calculate weight using the enhanced formula
            weight = (t_base * (1 + K_hazard * severity)) / (lanes + 1)
            
            data['weight'] = weight
            data['original_weight'] = weight
            data['base_time'] = t_base
            data['lanes'] = lanes
            data['severity'] = severity
            data['blocked'] = False
    
    def _create_fallback_graph(self):
        """Create a synthetic graph when real map data fails"""
        print("Creating synthetic grid network for simulation...")
        
        # Create a grid graph as fallback
        rows, cols = 20, 20
        self.graph = nx.grid_2d_graph(rows, cols)
        
        # Convert to standard node IDs
        mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
        # Add coordinates to nodes
        for node in self.graph.nodes():
            row, col = divmod(node, cols)
            # Map to approximate coordinates around our target area
            base_lat, base_lon = self.start_coords
            lat = base_lat + (row - rows//2) * 0.001
            lon = base_lon + (col - cols//2) * 0.001
            self.graph.nodes[node]['x'] = lon
            self.graph.nodes[node]['y'] = lat
        
        # Add edge weights with enhanced formula
        for u, v, data in self.graph.edges(data=True):
            data['length'] = 100  # 100m per edge
            
            # Enhanced weight calculation
            t_base = data['length'] / 50.0  # Base time assuming 50 km/h
            lanes = 2  # Default lanes for synthetic graph
            severity = 0.0  # No initial hazards
            K_hazard = 4.0
            
            weight = (t_base * (1 + K_hazard * severity)) / (lanes + 1)
            
            data['weight'] = weight
            data['original_weight'] = weight
            data['base_time'] = t_base
            data['lanes'] = lanes
            data['severity'] = severity
            data['blocked'] = False
        
        # Set start and goal nodes
        self.start_node = 0  # Top-left corner
        self.goal_node = rows * cols - 1  # Bottom-right corner
        
        print(f"Fallback graph created: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
    
    def generate_hazard_timeline(self, num_hazards: int = 4, max_steps: int = 50):
        """Generate a timeline of hazard events that strategically target algorithm paths"""
        self.hazard_events = []
        
        # First, get the initial paths from each algorithm to understand their routes
        print("Analyzing algorithm paths for strategic hazard placement...")
        algorithm_paths = self._get_algorithm_paths()
        
        # Combine all unique edges from all paths
        all_path_edges = set()
        for path in algorithm_paths.values():
            if path:
                for i in range(len(path) - 1):
                    all_path_edges.add((path[i], path[i+1]))
                    all_path_edges.add((path[i+1], path[i]))  # Add reverse for undirected
        
        path_edges_list = list(all_path_edges)
        
        if not path_edges_list:
            print("Warning: No paths found, using random hazard placement")
            self._generate_random_hazards(num_hazards, max_steps)
            return
        
        # Ensure hazards are spread throughout the simulation
        hazard_steps = np.linspace(8, max_steps-15, num_hazards, dtype=int)
        
        for i, step in enumerate(hazard_steps):
            # Select edges from paths (80% chance) or random edges (20% chance)
            if np.random.random() < 0.8 and path_edges_list:
                # Target path edges to ensure replanning
                num_path_edges = min(8, len(path_edges_list))
                affected_edges = list(np.random.choice(len(path_edges_list), 
                                                     size=num_path_edges, replace=False))
                affected_edges = [path_edges_list[idx] for idx in affected_edges]
                
                # Also add some nodes from the paths
                all_path_nodes = set()
                for path in algorithm_paths.values():
                    if path:
                        all_path_nodes.update(path[1:-1])  # Exclude start/goal
                
                path_nodes_list = list(all_path_nodes)
                affected_nodes = []
                if path_nodes_list:
                    num_nodes = min(3, len(path_nodes_list))
                    node_indices = np.random.choice(len(path_nodes_list), size=num_nodes, replace=False)
                    affected_nodes = [path_nodes_list[idx] for idx in node_indices]
            else:
                # Random hazards as fallback
                nodes = list(self.graph.nodes())
                edges = list(self.graph.edges())
                
                affected_nodes = np.random.choice(nodes, size=min(3, len(nodes)), replace=False).tolist()
                affected_edges = [edges[j] for j in np.random.choice(len(edges), size=min(8, len(edges)), replace=False)]
            
            # Randomly select hazard type with realistic distribution
            hazard_types = ['fire', 'flood', 'traffic_incident', 'building_collapse']
            hazard_weights = [0.15, 0.25, 0.40, 0.20]  # Traffic incidents most common
            hazard_type = np.random.choice(hazard_types, p=hazard_weights)
            
            hazard = HazardEvent(
                event_id=i+1,
                step=step,
                affected_nodes=affected_nodes,
                affected_edges=affected_edges,
                hazard_type=hazard_type,
                severity=0.0,  # Will be calculated from hazard_type
                duration=np.random.randint(8, 20),     # Longer duration
                timestamp=time.time()
            )
            self.hazard_events.append(hazard)
            
            print(f"Hazard {i+1}: Step {step}, Type: {hazard_type} (severity {hazard.severity_grade}), "
                  f"{len(affected_edges)} edges, {len(affected_nodes)} nodes")
        
        print(f"Generated {len(self.hazard_events)} strategic hazard events targeting algorithm paths")
    
    def _get_algorithm_paths(self) -> Dict[str, List[int]]:
        """Get initial paths from all algorithms for strategic hazard placement"""
        paths = {}
        
        try:
            # Import algorithms here to avoid circular imports
            from .algorithms import DijkstraAlgorithm, DStarLiteAlgorithm, SSSPAlgorithm
            
            # Get Dijkstra path
            dijkstra = DijkstraAlgorithm(self.graph, self.start_node, self.goal_node)
            paths['Dijkstra'] = dijkstra.find_initial_path()
            
            # Get D* Lite path  
            dstar = DStarLiteAlgorithm(self.graph, self.start_node, self.goal_node)
            paths['DStar_Lite'] = dstar.find_initial_path()
            
            # Get SSSP path
            sssp = SSSPAlgorithm(self.graph, self.start_node, self.goal_node)
            paths['SSSP'] = sssp.find_initial_path()
            
            # Print path information
            for alg_name, path in paths.items():
                if path:
                    print(f"{alg_name} initial path: {len(path)} nodes")
                else:
                    print(f"{alg_name}: No path found")
                    
        except Exception as e:
            print(f"Error getting algorithm paths: {e}")
            paths = {}
        
        return paths
    
    def _generate_random_hazards(self, num_hazards: int, max_steps: int):
        """Fallback method for random hazard generation"""
        hazard_steps = np.linspace(5, max_steps-10, num_hazards, dtype=int)
        
        for i, step in enumerate(hazard_steps):
            nodes = list(self.graph.nodes())
            edges = list(self.graph.edges())
            
            affected_nodes = np.random.choice(nodes, size=min(5, len(nodes)), replace=False).tolist()
            affected_edges = [edges[j] for j in np.random.choice(len(edges), size=min(10, len(edges)), replace=False)]
            
            # Random hazard type for fallback
            hazard_types = ['fire', 'flood', 'traffic_incident', 'building_collapse']
            hazard_type = np.random.choice(hazard_types)
            
            hazard = HazardEvent(
                event_id=i+1,
                step=step,
                affected_nodes=affected_nodes,
                affected_edges=affected_edges,
                hazard_type=hazard_type,
                severity=0.0,  # Will be calculated from hazard_type
                duration=np.random.randint(5, 15),
                timestamp=time.time()
            )
            self.hazard_events.append(hazard)
    
    def apply_hazards(self, step: int) -> List[HazardEvent]:
        """Apply hazards that should be active at this step"""
        new_hazards = []
        
        # Add new hazards
        for hazard in self.hazard_events:
            if hazard.step == step:
                self.active_hazards[hazard.event_id] = hazard
                new_hazards.append(hazard)
                
                # Apply enhanced weight calculation for affected edges
                for u, v in hazard.affected_edges:
                    if self.graph.has_edge(u, v):
                        edge_data = self.graph[u][v]
                        if isinstance(edge_data, dict):
                            self._apply_hazard_to_edge(edge_data, hazard)
                        else:
                            # For multi-graphs, apply to first edge
                            for key in edge_data:
                                self._apply_hazard_to_edge(edge_data[key], hazard)
                                break
        
        # Remove expired hazards
        expired_hazards = []
        for hazard_id, hazard in list(self.active_hazards.items()):
            if step >= hazard.step + hazard.duration:
                expired_hazards.append(hazard_id)
                
                # Restore original weights for affected edges
                for u, v in hazard.affected_edges:
                    if self.graph.has_edge(u, v):
                        edge_data = self.graph[u][v]
                        if isinstance(edge_data, dict):
                            self._remove_hazard_from_edge(edge_data)
                        else:
                            # For multi-graphs, restore first edge
                            for key in edge_data:
                                self._remove_hazard_from_edge(edge_data[key])
                                break
        
        for hazard_id in expired_hazards:
            del self.active_hazards[hazard_id]
        
        return new_hazards
    
    def _apply_hazard_to_edge(self, edge_data: dict, hazard: HazardEvent):
        """Apply hazard to edge using enhanced weight formula"""
        # Get base parameters
        t_base = edge_data.get('base_time', edge_data.get('length', 100) / 50.0)
        lanes = edge_data.get('lanes', 2)
        K_hazard = 4.0
        
        # Get severity from hazard type
        severity = hazard.severity_grade
        
        # Calculate new weight: w(e) = (t_base * (1 + K_hazard * s(e))) / (lanes + 1)
        new_weight = (t_base * (1 + K_hazard * severity)) / (lanes + 1)
        
        # Update edge data
        edge_data['weight'] = new_weight
        edge_data['severity'] = severity
        edge_data['blocked'] = severity > 0
        edge_data['hazard_type'] = hazard.hazard_type
    
    def _remove_hazard_from_edge(self, edge_data: dict):
        """Remove hazard from edge and restore original weight"""
        # Get base parameters
        t_base = edge_data.get('base_time', edge_data.get('length', 100) / 50.0)
        lanes = edge_data.get('lanes', 2)
        K_hazard = 4.0
        severity = 0.0  # No hazard
        
        # Calculate original weight
        original_weight = (t_base * (1 + K_hazard * severity)) / (lanes + 1)
        
        # Restore edge data
        edge_data['weight'] = original_weight
        edge_data['severity'] = severity
        edge_data['blocked'] = False
        edge_data['hazard_type'] = None
    
    def get_current_graph(self):
        """Get current graph state with active hazards"""
        return self.graph
    
    def reset(self):
        """Reset environment for new simulation"""
        self.current_step = 0
        self.active_hazards = {}
        
        # Reset all edge weights to original values using enhanced formula
        for u, v, data in self.graph.edges(data=True):
            if isinstance(data, dict):
                self._remove_hazard_from_edge(data)

class EvacuationSimulator:
    """Main simulator coordinating all algorithms and analysis"""
    
    def __init__(self, environment: EvacuationEnvironment):
        self.env = environment
        self.algorithms = {}
        self.results = {}
        self.replanning_events = []
        self.frame_data = []
        
    def register_algorithm(self, name: str, algorithm_class):
        """Register a pathfinding algorithm"""
        self.algorithms[name] = algorithm_class
    
    def run_simulation(self, max_steps: int = 50, save_frames: bool = True):
        """Run complete simulation with all algorithms"""
        print("Starting evacuation simulation...")
        
        # Generate unified hazard timeline with more hazards for longer routes
        self.env.generate_hazard_timeline(num_hazards=12, max_steps=max_steps)
        
        # Run each algorithm
        for alg_name, alg_class in self.algorithms.items():
            print(f"\n--- Running {alg_name} ---")
            self.env.reset()
            
            algorithm = alg_class(self.env.graph, self.env.start_node, self.env.goal_node)
            
            # Initialize
            start_time = time.time()
            current_path = algorithm.find_initial_path()
            current_position = 0
            nodes_explored = 0
            replanning_count = 0
            
            if not current_path:
                self.results[alg_name] = AlgorithmResult(
                    algorithm=alg_name,
                    success=False,
                    failure_reason="No initial path found",
                    path_length=0,
                    path_cost=0,
                    execution_time=time.time() - start_time,
                    nodes_explored=0,
                    replanning_count=0,
                    hazards_encountered=0,
                    total_replanning_events=0,
                    hazard_response_count=0,
                    final_path=[]
                )
                continue
            
            # Simulation loop
            for step in range(max_steps):
                self.env.current_step = step
                
                # Apply hazards
                new_hazards = self.env.apply_hazards(step)
                
                if new_hazards:
                    print(f"Step {step}: Applied {len(new_hazards)} new hazards")
                
                # Check if replanning is needed
                if new_hazards:
                    remaining_path = current_path[current_position:]
                    path_affected = self._path_affected_by_hazards(remaining_path, new_hazards)
                    
                    if path_affected:
                        print(f"Step {step}: Path affected, triggering replanning for {alg_name}")
                        replan_start = time.time()
                        
                        # Perform replanning
                        new_path = algorithm.replan(current_path, current_position, new_hazards)
                        replan_time = time.time() - replan_start
                        
                        if new_path:
                            # Record replanning event
                            replanning_event = ReplanningEvent(
                                algorithm=alg_name,
                                hazard_event_id=new_hazards[0].event_id if new_hazards else -1,
                                step=step,
                                progress_percentage=(current_position / len(current_path)) * 100 if current_path else 0,
                                hazards_injected=len(self.env.active_hazards),
                                replanning_success=True,
                                response_time=replan_time,
                                new_path_length=len(new_path),
                                old_path=current_path[current_position:],
                                new_path=new_path,
                                replanning_trigger="hazard_detection"
                            )
                            self.replanning_events.append(replanning_event)
                            
                            print(f"  Replanning successful: {len(current_path[current_position:])} -> {len(new_path)} nodes")
                            
                            # Update path
                            current_path = current_path[:current_position] + new_path
                            replanning_count += 1
                            
                            # Save frame if requested
                            if save_frames:
                                self._save_frame(alg_name, step, current_path, current_position, 
                                               new_hazards, replanning_event)
                        else:
                            print(f"  Replanning failed for {alg_name}")
                    else:
                        print(f"Step {step}: Hazards present but path not affected")
                
                # Move along path
                if current_position < len(current_path) - 1:
                    current_position += 1
                    if step % 10 == 0:  # Progress update every 10 steps
                        progress = (current_position / len(current_path)) * 100
                        print(f"Step {step}: {alg_name} progress: {progress:.1f}%")
                
                # Check if goal reached
                if current_position >= len(current_path) - 1:
                    if current_path[-1] == self.env.goal_node:
                        print(f"{alg_name} reached goal at step {step}")
                        break
            
            # Calculate final results
            execution_time = time.time() - start_time
            path_cost = self._calculate_path_cost(current_path)
            
            self.results[alg_name] = AlgorithmResult(
                algorithm=alg_name,
                success=current_path[-1] == self.env.goal_node if current_path else False,
                failure_reason="" if current_path and current_path[-1] == self.env.goal_node else "Did not reach goal",
                path_length=len(current_path),
                path_cost=path_cost,
                execution_time=execution_time,
                nodes_explored=getattr(algorithm, 'nodes_explored', 0),
                replanning_count=replanning_count,
                hazards_encountered=len([h for h in self.env.hazard_events]),
                total_replanning_events=len([e for e in self.replanning_events if e.algorithm == alg_name]),
                hazard_response_count=len([e for e in self.replanning_events if e.algorithm == alg_name]),
                final_path=current_path
            )
        
        print("\nSimulation completed!")
        return self.results, self.replanning_events
    
    def _path_affected_by_hazards(self, path_segment: List[int], hazards: List[HazardEvent]) -> bool:
        """Check if path segment is affected by hazards (more sensitive detection)"""
        if not path_segment or not hazards:
            return False
            
        for hazard in hazards:
            # Check if any path nodes are affected
            for node in path_segment:
                if node in hazard.affected_nodes:
                    print(f"Path affected: Node {node} hit by hazard {hazard.event_id}")
                    return True
            
            # Check if any path edges are affected (more thorough check)
            for i in range(len(path_segment) - 1):
                edge = (path_segment[i], path_segment[i+1])
                reverse_edge = (path_segment[i+1], path_segment[i])
                
                # Check both edge directions
                for affected_edge in hazard.affected_edges:
                    if (edge == affected_edge or reverse_edge == affected_edge or
                        (edge[0] == affected_edge[0] and edge[1] == affected_edge[1]) or
                        (edge[0] == affected_edge[1] and edge[1] == affected_edge[0])):
                        print(f"Path affected: Edge {edge} hit by hazard {hazard.event_id}")
                        return True
        
        return False
    
    def _calculate_path_cost(self, path: List[int]) -> float:
        """Calculate total cost of a path"""
        if not path or len(path) < 2:
            return 0
        
        total_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.env.graph.has_edge(u, v):
                edge_data = self.env.graph[u][v]
                if isinstance(edge_data, dict):
                    total_cost += edge_data.get('weight', 100)
                else:
                    # For multi-graphs, get the first edge weight
                    for key in edge_data:
                        total_cost += edge_data[key].get('weight', 100)
                        break
        
        return total_cost
    
    def _save_frame(self, algorithm: str, step: int, path: List[int], position: int, 
                   hazards: List[HazardEvent], replanning_event: ReplanningEvent):
        """Save visualization frame"""
        frame_dir = f"frames/{algorithm}"
        os.makedirs(frame_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the map
        ox.plot_graph(self.env.graph, ax=ax, show=False, close=False,
                     node_size=10, edge_color='lightgray', edge_linewidth=0.5)
        
        # Plot current path up to position (traveled)
        if len(path) > 1 and position > 0:
            traveled_path = path[:position+1]
            if len(traveled_path) > 1:
                try:
                    ox.plot_graph_route(self.env.graph, traveled_path, ax=ax, route_color='blue', 
                                       route_linewidth=3, show=False, close=False)
                except (AttributeError, KeyError):
                    # Handle edge data issues - plot manually
                    path_edges = [(traveled_path[i], traveled_path[i+1]) for i in range(len(traveled_path)-1)]
                    for u, v in path_edges:
                        if self.env.graph.has_edge(u, v):
                            try:
                                x_coords = [self.env.graph.nodes[u]['x'], self.env.graph.nodes[v]['x']]
                                y_coords = [self.env.graph.nodes[u]['y'], self.env.graph.nodes[v]['y']]
                                ax.plot(x_coords, y_coords, color='blue', linewidth=3, alpha=0.8)
                            except KeyError:
                                pass
        
        # Plot replanned section (if any) in lighter color
        if replanning_event and len(replanning_event.new_path) > 1:
            try:
                ox.plot_graph_route(self.env.graph, replanning_event.new_path, ax=ax, 
                                   route_color='lightblue', route_linewidth=2, show=False, close=False)
            except (AttributeError, KeyError):
                # Handle edge data issues - plot manually
                new_path = replanning_event.new_path
                for i in range(len(new_path) - 1):
                    u, v = new_path[i], new_path[i+1]
                    if self.env.graph.has_edge(u, v):
                        try:
                            x_coords = [self.env.graph.nodes[u]['x'], self.env.graph.nodes[v]['x']]
                            y_coords = [self.env.graph.nodes[u]['y'], self.env.graph.nodes[v]['y']]
                            ax.plot(x_coords, y_coords, color='lightblue', linewidth=2, alpha=0.6)
                        except KeyError:
                            pass
        
        # Plot hazards
        for hazard in hazards:
            hazard_nodes = [n for n in hazard.affected_nodes if n in self.env.graph.nodes()]
            if hazard_nodes:
                node_positions = [(self.env.graph.nodes[n]['x'], self.env.graph.nodes[n]['y']) 
                                for n in hazard_nodes]
                xs, ys = zip(*node_positions)
                ax.scatter(xs, ys, c='red', s=100, alpha=0.7, marker='X', label=f'Hazard {hazard.event_id}')
        
        # Plot current position
        if position < len(path):
            current_node = path[position]
            if current_node in self.env.graph.nodes():
                ax.scatter(self.env.graph.nodes[current_node]['x'], 
                          self.env.graph.nodes[current_node]['y'],
                          c='green', s=200, marker='o', label='Current Position')
        
        # Plot start and goal
        ax.scatter(self.env.graph.nodes[self.env.start_node]['x'], 
                  self.env.graph.nodes[self.env.start_node]['y'],
                  c='blue', s=200, marker='s', label='Start')
        ax.scatter(self.env.graph.nodes[self.env.goal_node]['x'], 
                  self.env.graph.nodes[self.env.goal_node]['y'],
                  c='red', s=200, marker='s', label='Goal')
        
        ax.set_title(f'{algorithm} - Step {step} - Replanning Event')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{frame_dir}/frame_{step:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename: str = "simulation_results"):
        """Save simulation results to files"""
        # Save replanning events
        replanning_df = pd.DataFrame([asdict(event) for event in self.replanning_events])
        replanning_df.to_csv(f"results/{filename}_replanning.csv", index=False)
        
        # Save final results
        results_df = pd.DataFrame([asdict(result) for result in self.results.values()])
        results_df.to_csv(f"results/{filename}_final.csv", index=False)
        
        # Save as JSON for detailed analysis
        with open(f"results/{filename}_detailed.json", 'w') as f:
            json.dump({
                'replanning_events': [asdict(event) for event in self.replanning_events],
                'final_results': [asdict(result) for result in self.results.values()],
                'hazard_timeline': [asdict(hazard) for hazard in self.env.hazard_events]
            }, f, indent=2, default=str)
        
        print(f"Results saved to results/{filename}_*")

if __name__ == "__main__":
    # This will be imported by the main script
    pass
