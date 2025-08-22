"""
Main script to run the enhanced evacuation pathfinding simulation
Coordinates the simulation environment, algorithms, and visualization
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.simulation_environment import EvacuationEnvironment, EvacuationSimulator
from src.algorithms import DijkstraAlgorithm, DStarLiteAlgorithm, SSSPAlgorithm
from src.visualization import EvacuationVisualizer
from src.research_documentation import ResearchDocumentationGenerator

# Configuration - Longer routes for more comprehensive testing
START_COORDS = (48.8566, 2.3522)   # Notre-Dame Cathedral (central Paris)
GOAL_COORDS = (48.8738, 2.2950)    # Arc de Triomphe (western Paris)
MAX_SIMULATION_STEPS = 100          # More steps for longer journey
SAVE_FRAMES = True

def setup_directories():
    """Create necessary directories for results and frames with automatic cleanup"""
    import shutil
    
    directories = ['results', 'frames', 'frames/Dijkstra', 'frames/DStar_Lite', 'frames/SSSP']
    
    # Clean existing directories
    print("🧹 Cleaning previous results...")
    for directory in ['results', 'frames']:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"   Cleaned: {directory}/")
    
    # Create fresh directories
    print("📁 Creating fresh directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directory setup complete!")

def print_simulation_header():
    """Print simulation header information"""
    print("=" * 80)
    print("ENHANCED EVACUATION PATHFINDING SIMULATION")
    print("=" * 80)
    print(f"Start Location: {START_COORDS} (Notre-Dame Cathedral)")
    print(f"Goal Location:  {GOAL_COORDS} (Arc de Triomphe)")
    print(f"Max Steps: {MAX_SIMULATION_STEPS}")
    print(f"Algorithms: Dijkstra, D* Lite (DStar_Lite), SSSP")
    print(f"Real Map Data: OpenStreetMap (Paris)")
    print("=" * 80)

def main():
    """Main simulation execution"""
    print_simulation_header()
    
    # Setup
    setup_directories()
    
    try:
        # Initialize environment with real map data
        print("\n🗺️  Initializing simulation environment...")
        environment = EvacuationEnvironment(START_COORDS, GOAL_COORDS)
        
        # Create simulator
        simulator = EvacuationSimulator(environment)
        
        # Register algorithms
        print("🧮 Registering pathfinding algorithms...")
        simulator.register_algorithm("Dijkstra", DijkstraAlgorithm)
        simulator.register_algorithm("DStar_Lite", DStarLiteAlgorithm)
        simulator.register_algorithm("SSSP", SSSPAlgorithm)
        
        # Run simulation
        print("🚀 Starting simulation...")
        start_time = time.time()
        
        results, replanning_events = simulator.run_simulation(
            max_steps=MAX_SIMULATION_STEPS,
            save_frames=SAVE_FRAMES
        )
        
        simulation_time = time.time() - start_time
        print(f"\n✅ Simulation completed in {simulation_time:.2f} seconds")
        
        # Save results
        print("💾 Saving simulation results...")
        simulator.save_results("enhanced_simulation_results")
        
        # Print summary
        print_results_summary(results, replanning_events)
        
        # Generate visualizations
        print("📊 Generating visualizations and analysis...")
        visualizer = EvacuationVisualizer()
        visualizer.load_results("enhanced_simulation_results")
        visualizer.generate_report("results/enhanced_simulation_report.html")
        
        # Generate comprehensive research documentation
        print("📚 Generating comprehensive research documentation...")
        research_gen = ResearchDocumentationGenerator()
        research_gen.load_simulation_data("enhanced_simulation_results")
        research_gen.generate_comprehensive_report(environment, simulator)
        
        print("\n🎉 Complete simulation and analysis finished!")
        print("📁 Check the 'results' and 'frames' directories for outputs")
        print("🎓 Academic documentation ready for publication!")
        
    except Exception as e:
        print(f"\n❌ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def print_results_summary(results, replanning_events):
    """Print a summary of simulation results"""
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    
    # Algorithm performance summary
    print("\n📊 ALGORITHM PERFORMANCE:")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Success':<8} {'Time(s)':<10} {'Path Len':<10} {'Nodes':<8} {'Replans':<8}")
    print("-" * 60)
    
    for alg_name, result in results.items():
        success_icon = "✅" if result.success else "❌"
        print(f"{alg_name:<12} {success_icon:<8} {result.execution_time:<10.4f} "
              f"{result.path_length:<10} {result.nodes_explored:<8} {result.replanning_count:<8}")
    
    # Replanning events summary
    if replanning_events:
        print(f"\n🔄 REPLANNING EVENTS: {len(replanning_events)} total events")
        print("-" * 60)
        
        from collections import defaultdict
        replan_by_alg = defaultdict(list)
        for event in replanning_events:
            replan_by_alg[event.algorithm].append(event)
        
        for alg_name, events in replan_by_alg.items():
            avg_response = sum(e.response_time for e in events) / len(events)
            success_rate = sum(e.replanning_success for e in events) / len(events)
            print(f"{alg_name:<12} Events: {len(events):<3} Avg Response: {avg_response:<8.4f}s "
                  f"Success Rate: {success_rate:.1%}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 60)
    
    # Find best performers
    fastest_alg = min(results.keys(), key=lambda x: results[x].execution_time)
    shortest_path_alg = min(results.keys(), key=lambda x: results[x].path_length)
    most_efficient_alg = min(results.keys(), key=lambda x: results[x].nodes_explored)
    
    print(f"🏃 Fastest execution: {fastest_alg} ({results[fastest_alg].execution_time:.4f}s)")
    print(f"🎯 Shortest path: {shortest_path_alg} ({results[shortest_path_alg].path_length} nodes)")
    print(f"🧠 Most efficient: {most_efficient_alg} ({results[most_efficient_alg].nodes_explored} nodes explored)")
    
    if replanning_events:
        fastest_replan_alg = min(replan_by_alg.keys(), 
                                key=lambda x: sum(e.response_time for e in replan_by_alg[x]) / len(replan_by_alg[x]))
        avg_fastest_replan = sum(e.response_time for e in replan_by_alg[fastest_replan_alg]) / len(replan_by_alg[fastest_replan_alg])
        print(f"⚡ Fastest replanning: {fastest_replan_alg} ({avg_fastest_replan:.4f}s avg)")

def run_quick_test():
    """Run a quick test with minimal configuration for debugging"""
    print("🧪 Running quick test mode...")
    
    try:
        # Test environment creation
        env = EvacuationEnvironment(START_COORDS, GOAL_COORDS)
        print(f"✅ Environment created with {len(env.graph.nodes)} nodes")
        
        # Test one algorithm
        from src.algorithms import DijkstraAlgorithm
        dijkstra = DijkstraAlgorithm(env.graph, env.start_node, env.goal_node)
        path = dijkstra.find_initial_path()
        
        if path:
            print(f"✅ Dijkstra found path with {len(path)} nodes")
        else:
            print("❌ Dijkstra failed to find path")
        
        print("🎉 Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if quick test mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_quick_test()
    else:
        exit_code = main()
        sys.exit(exit_code)
