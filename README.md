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

## Requirements

```
osmnx>=1.6.0
networkx>=3.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
seaborn>=0.12.0
```

## What You'll Get

After running the simulation, check the `results/` folder for:

- **Performance charts**: How fast each algorithm ran
- **Path visualizations**: Maps showing the routes taken
- **Replanning analysis**: How algorithms adapted to hazards
- **HTML report**: Interactive analysis with detailed metrics
- **Academic tables**: LaTeX-formatted tables for research papers

## The Algorithms

- **Dijkstra**: Classic algorithm, recalculates everything when hazards appear
- **D* Lite**: Smart algorithm, only updates affected parts of the path
- **SSSP**: Multi-target algorithm with fallback to Dijkstra when needed

## Simulation Details

- **Route**: 4.6km from Notre-Dame Cathedral to Arc de Triomphe in Paris
- **Hazards**: Fire, flood, traffic incidents, building collapses
- **Real Data**: Uses actual Paris street network from OpenStreetMap
- **Duration**: Takes about 2-5 minutes to complete

## Troubleshooting

- **Internet Required**: Downloads map data from OpenStreetMap
- **Windows**: Use `python main.py` (not `python3`)
- **Errors**: Make sure all packages are installed with `pip install -r requirements.txt`

## For Researchers

The simulation generates publication-ready outputs including:
- Statistical performance comparisons
- LaTeX-formatted tables
- High-resolution figures
- Comprehensive metadata for methodology sections
