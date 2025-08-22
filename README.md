# Evacuation Pathfinding Simulation

Compare three pathfinding algorithms (Dijkstra, D* Lite, and SSSP) on real Paris street maps with dynamic hazards.

## Quick Start

1. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation**:
   ```bash
   python main.py
   ```

3. **Check results**: Look in the `results/` folder for charts and analysis

## What It Does

- Downloads real street data from Paris (Notre-Dame to Arc de Triomphe)
- Tests 3 different pathfinding algorithms
- Adds random hazards (fires, floods, traffic) during evacuation
- Compares how well each algorithm handles changing conditions
- Generates visualizations and performance reports

## What You'll Get

After running the simulation, check the `results/` folder for:

- **Performance charts**: How fast each algorithm ran
- **Path visualizations**: Maps showing the routes taken
- **Replanning analysis**: How algorithms adapted to hazards

## The Algorithms

- **Dijkstra**: Classic algorithm, recalculates everything when hazards appear
- **D* Lite**: Smart algorithm, only updates affected parts of the path
- **SSSP**: Multi-target algorithm with fallback to Dijkstra when needed

## Simulation Details (default usecase, can be edited)

- **Route**: 4.6km from Notre-Dame Cathedral to Arc de Triomphe in Paris
- **Hazards**: Fire, flood, traffic incidents, building collapses
- **Real Data**: Uses actual Paris street network from OpenStreetMap
