"""
Visualization and Analysis Module for Evacuation Simulation
Creates comprehensive visualizations and performance analysis
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import folium
import os
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class EvacuationVisualizer:
    """Comprehensive visualization suite for evacuation simulation results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.colors = {
            'Dijkstra': '#1f77b4',
            'D* Lite': '#ff7f0e',
            'DStar_Lite': '#ff7f0e',  # Handle both naming conventions
            'SSSP': '#2ca02c'
        }
        
    def load_results(self, filename: str = "simulation_results"):
        """Load simulation results from files"""
        replanning_file = f"{self.results_dir}/{filename}_replanning.csv"
        final_file = f"{self.results_dir}/{filename}_final.csv"
        detailed_file = f"{self.results_dir}/{filename}_detailed.json"
        
        # Load replanning results with error handling
        if os.path.exists(replanning_file) and os.path.getsize(replanning_file) > 0:
            try:
                self.replanning_df = pd.read_csv(replanning_file)
            except pd.errors.EmptyDataError:
                print("Warning: Replanning file is empty, creating empty DataFrame")
                self.replanning_df = pd.DataFrame()
        else:
            print("Warning: No replanning events found or file is empty")
            self.replanning_df = pd.DataFrame()
        
        # Load final results
        self.final_df = pd.read_csv(final_file) if os.path.exists(final_file) else pd.DataFrame()
        
        # Load detailed JSON data
        if os.path.exists(detailed_file):
            with open(detailed_file, 'r') as f:
                self.detailed_data = json.load(f)
        else:
            self.detailed_data = {}
            
        print(f"Loaded results: {len(self.final_df)} algorithms, {len(self.replanning_df)} replanning events")
    
    def create_performance_comparison(self, save_path: str = "results/performance_comparison.png"):
        """Create comprehensive performance comparison visualization"""
        if self.final_df.empty:
            print("No final results data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Execution Time Comparison
        ax1 = axes[0, 0]
        algorithms = self.final_df['algorithm']
        exec_times = self.final_df['execution_time']
        bars1 = ax1.bar(algorithms, exec_times, color=[self.colors[alg] for alg in algorithms])
        ax1.set_title('Execution Time (seconds)')
        ax1.set_ylabel('Time (s)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars1, exec_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.3f}', ha='center', va='bottom')
        
        # 2. Path Length Comparison
        ax2 = axes[0, 1]
        path_lengths = self.final_df['path_length']
        bars2 = ax2.bar(algorithms, path_lengths, color=[self.colors[alg] for alg in algorithms])
        ax2.set_title('Final Path Length')
        ax2.set_ylabel('Number of Nodes')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, length in zip(bars2, path_lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{length}', ha='center', va='bottom')
        
        # 3. Nodes Explored
        ax3 = axes[0, 2]
        nodes_explored = self.final_df['nodes_explored']
        bars3 = ax3.bar(algorithms, nodes_explored, color=[self.colors[alg] for alg in algorithms])
        ax3.set_title('Nodes Explored')
        ax3.set_ylabel('Number of Nodes')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, nodes in zip(bars3, nodes_explored):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{nodes}', ha='center', va='bottom')
        
        # 4. Replanning Response Time
        ax4 = axes[1, 0]
        if not self.replanning_df.empty:
            response_times = self.replanning_df.groupby('algorithm')['response_time'].mean()
            bars4 = ax4.bar(response_times.index, response_times.values, 
                           color=[self.colors[alg] for alg in response_times.index])
            ax4.set_title('Average Replanning Response Time')
            ax4.set_ylabel('Time (s)')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, time in zip(bars4, response_times.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time:.4f}', ha='center', va='bottom')
        
        # 5. Success Rate and Path Cost
        ax5 = axes[1, 1]
        path_costs = self.final_df['path_cost']
        bars5 = ax5.bar(algorithms, path_costs, color=[self.colors[alg] for alg in algorithms])
        ax5.set_title('Total Path Cost')
        ax5.set_ylabel('Cost')
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars5, path_costs):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        # 6. Replanning Events Timeline
        ax6 = axes[1, 2]
        if not self.replanning_df.empty:
            for alg in self.replanning_df['algorithm'].unique():
                alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
                ax6.scatter(alg_data['step'], [alg] * len(alg_data), 
                           c=self.colors[alg], s=100, alpha=0.7, label=alg)
            
            ax6.set_title('Replanning Events Timeline')
            ax6.set_xlabel('Simulation Step')
            ax6.set_ylabel('Algorithm')
            ax6.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Performance comparison saved to {save_path}")
    
    def create_replanning_analysis(self, save_path: str = "results/replanning_analysis.png"):
        """Detailed analysis of replanning behavior"""
        if self.replanning_df.empty:
            print("No replanning data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Replanning Behavior Analysis', fontsize=16, fontweight='bold')
        
        # 1. Response Time Distribution
        ax1 = axes[0, 0]
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            ax1.hist(alg_data['response_time'], alpha=0.7, label=alg, 
                    color=self.colors[alg], bins=10)
        ax1.set_title('Response Time Distribution')
        ax1.set_xlabel('Response Time (s)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Path Length Changes
        ax2 = axes[0, 1]
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            ax2.plot(alg_data['hazard_event_id'], alg_data['new_path_length'], 
                    marker='o', label=alg, color=self.colors[alg])
        ax2.set_title('Path Length Changes During Replanning')
        ax2.set_xlabel('Hazard Event ID')
        ax2.set_ylabel('New Path Length')
        ax2.legend()
        
        # 3. Progress vs Response Time
        ax3 = axes[1, 0]
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            ax3.scatter(alg_data['progress_percentage'], alg_data['response_time'],
                       label=alg, color=self.colors[alg], alpha=0.7, s=100)
        ax3.set_title('Progress vs Response Time')
        ax3.set_xlabel('Progress Percentage')
        ax3.set_ylabel('Response Time (s)')
        ax3.legend()
        
        # 4. Hazards Injected vs Response Time
        ax4 = axes[1, 1]
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            ax4.scatter(alg_data['hazards_injected'], alg_data['response_time'],
                       label=alg, color=self.colors[alg], alpha=0.7, s=100)
        ax4.set_title('Active Hazards vs Response Time')
        ax4.set_xlabel('Number of Active Hazards')
        ax4.set_ylabel('Response Time (s)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Replanning analysis saved to {save_path}")
    
    def create_interactive_timeline(self, save_path: str = "results/interactive_timeline.html"):
        """Create interactive timeline visualization using Plotly"""
        if self.replanning_df.empty:
            print("No replanning data available")
            return
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Response Time Over Time', 'Path Length Changes', 'Active Hazards'),
            vertical_spacing=0.08
        )
        
        # Response Time Timeline
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            fig.add_trace(
                go.Scatter(
                    x=alg_data['step'],
                    y=alg_data['response_time'],
                    mode='lines+markers',
                    name=f'{alg} Response Time',
                    line=dict(color=self.colors[alg]),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Step: %{x}<br>' +
                                 'Response Time: %{y:.4f}s<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Path Length Changes
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            fig.add_trace(
                go.Scatter(
                    x=alg_data['step'],
                    y=alg_data['new_path_length'],
                    mode='lines+markers',
                    name=f'{alg} Path Length',
                    line=dict(color=self.colors[alg]),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Active Hazards
        for alg in self.replanning_df['algorithm'].unique():
            alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
            fig.add_trace(
                go.Scatter(
                    x=alg_data['step'],
                    y=alg_data['hazards_injected'],
                    mode='lines+markers',
                    name=f'{alg} Active Hazards',
                    line=dict(color=self.colors[alg]),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='Interactive Simulation Timeline',
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Simulation Step", row=3, col=1)
        fig.update_yaxes(title_text="Response Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Path Length", row=2, col=1)
        fig.update_yaxes(title_text="Active Hazards", row=3, col=1)
        
        fig.write_html(save_path)
        print(f"Interactive timeline saved to {save_path}")
    
    def create_algorithm_behavior_summary(self, save_path: str = "results/algorithm_behavior.png"):
        """Create detailed behavior comparison of algorithms"""
        if self.final_df.empty:
            print("No final results data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Behavior Summary', fontsize=16, fontweight='bold')
        
        # 1. Efficiency Radar Chart (normalized metrics)
        ax1 = axes[0, 0]
        algorithms = self.final_df['algorithm'].tolist()
        
        # Normalize metrics (lower is better for time, higher is better for success)
        metrics = ['execution_time', 'nodes_explored', 'replanning_count', 'path_length']
        normalized_data = {}
        
        for metric in metrics:
            values = self.final_df[metric].values
            # Invert for time-based metrics (so lower time = higher score)
            if metric in ['execution_time']:
                normalized_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
            normalized_data[metric] = normalized_values
        
        x_pos = np.arange(len(algorithms))
        bar_width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x_pos + i * bar_width, normalized_data[metric], 
                   bar_width, label=metric.replace('_', ' ').title())
        
        ax1.set_title('Normalized Performance Metrics')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Normalized Score (0-1)')
        ax1.set_xticks(x_pos + bar_width * 1.5)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        
        # 2. Success Rate and Robustness
        ax2 = axes[0, 1]
        success_rates = self.final_df['success'].astype(int)
        replanning_success = []
        
        if not self.replanning_df.empty:
            for alg in algorithms:
                alg_replanning = self.replanning_df[self.replanning_df['algorithm'] == alg]
                if len(alg_replanning) > 0:
                    success_rate = alg_replanning['replanning_success'].mean()
                    replanning_success.append(success_rate)
                else:
                    replanning_success.append(1.0)
        else:
            replanning_success = [1.0] * len(algorithms)
        
        x_pos = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, success_rates, width, label='Overall Success',
                       color=[self.colors[alg] for alg in algorithms], alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, replanning_success, width, label='Replanning Success',
                       color=[self.colors[alg] for alg in algorithms], alpha=0.4)
        
        ax2.set_title('Success Rates')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Success Rate')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algorithms)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Computational Efficiency
        ax3 = axes[1, 0]
        exec_times = self.final_df['execution_time']
        nodes_explored = self.final_df['nodes_explored']
        
        # Efficiency = nodes explored per second
        efficiency = nodes_explored / (exec_times + 1e-10)
        
        bars3 = ax3.bar(algorithms, efficiency, color=[self.colors[alg] for alg in algorithms])
        ax3.set_title('Computational Efficiency (Nodes/Second)')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Nodes Explored per Second')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars3, efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01,
                    f'{eff:.0f}', ha='center', va='bottom')
        
        # 4. Adaptability Score
        ax4 = axes[1, 1]
        if not self.replanning_df.empty:
            adaptability_scores = []
            for alg in algorithms:
                alg_data = self.replanning_df[self.replanning_df['algorithm'] == alg]
                if len(alg_data) > 0:
                    # Score based on response time and success rate
                    avg_response = alg_data['response_time'].mean()
                    success_rate = alg_data['replanning_success'].mean()
                    # Lower response time and higher success rate = better adaptability
                    adaptability = success_rate / (avg_response + 1e-10)
                    adaptability_scores.append(adaptability)
                else:
                    adaptability_scores.append(0)
            
            bars4 = ax4.bar(algorithms, adaptability_scores, 
                           color=[self.colors[alg] for alg in algorithms])
            ax4.set_title('Adaptability Score (Success/Response Time)')
            ax4.set_xlabel('Algorithm')
            ax4.set_ylabel('Adaptability Score')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars4, adaptability_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(adaptability_scores)*0.01,
                        f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Algorithm behavior summary saved to {save_path}")
    
    def generate_report(self, output_file: str = "results/simulation_report.html"):
        """Generate comprehensive HTML report"""
        # Create all visualizations
        self.create_performance_comparison()
        self.create_replanning_analysis()
        self.create_interactive_timeline()
        self.create_algorithm_behavior_summary()
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evacuation Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .stats-table {{ border-collapse: collapse; width: 100%; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .stats-table th {{ background-color: #f2f2f2; }}
                .insight {{ background-color: #f0f8ff; padding: 15px; border-left: 4px solid #007acc; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Evacuation Pathfinding Algorithm Comparison Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                {summary_stats}
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                {self._generate_insights()}
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>The following visualizations have been generated:</p>
                <ul>
                    <li><a href="performance_comparison.png">Performance Comparison</a></li>
                    <li><a href="replanning_analysis.png">Replanning Analysis</a></li>
                    <li><a href="algorithm_behavior.png">Algorithm Behavior Summary</a></li>
                    <li><a href="interactive_timeline.html">Interactive Timeline</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive report generated: {output_file}")
    
    def _generate_summary_stats(self) -> str:
        """Generate summary statistics table"""
        if self.final_df.empty:
            return "<p>No data available for summary statistics.</p>"
        
        stats_html = '<table class="stats-table">\n'
        stats_html += '<tr><th>Metric</th>'
        
        for alg in self.final_df['algorithm']:
            stats_html += f'<th>{alg}</th>'
        stats_html += '</tr>\n'
        
        metrics = [
            ('Execution Time (s)', 'execution_time'),
            ('Path Length', 'path_length'),
            ('Path Cost', 'path_cost'),
            ('Nodes Explored', 'nodes_explored'),
            ('Replanning Count', 'replanning_count'),
            ('Success', 'success')
        ]
        
        for metric_name, metric_col in metrics:
            stats_html += f'<tr><td>{metric_name}</td>'
            for _, row in self.final_df.iterrows():
                value = row[metric_col]
                if metric_col in ['execution_time', 'path_cost']:
                    stats_html += f'<td>{value:.4f}</td>'
                elif metric_col == 'success':
                    stats_html += f'<td>{"✓" if value else "✗"}</td>'
                else:
                    stats_html += f'<td>{value}</td>'
            stats_html += '</tr>\n'
        
        stats_html += '</table>'
        return stats_html
    
    def _generate_insights(self) -> str:
        """Generate key insights from the analysis"""
        insights = []
        
        if not self.final_df.empty:
            # Find best performing algorithm for each metric
            fastest_alg = self.final_df.loc[self.final_df['execution_time'].idxmin(), 'algorithm']
            shortest_path_alg = self.final_df.loc[self.final_df['path_length'].idxmin(), 'algorithm']
            most_efficient_alg = self.final_df.loc[self.final_df['nodes_explored'].idxmin(), 'algorithm']
            
            insights.append(f'<div class="insight"><strong>Fastest Algorithm:</strong> {fastest_alg} with {self.final_df["execution_time"].min():.4f}s execution time</div>')
            insights.append(f'<div class="insight"><strong>Shortest Path:</strong> {shortest_path_alg} found path with {self.final_df["path_length"].min()} nodes</div>')
            insights.append(f'<div class="insight"><strong>Most Efficient:</strong> {most_efficient_alg} explored only {self.final_df["nodes_explored"].min()} nodes</div>')
        
        if not self.replanning_df.empty:
            fastest_replan = self.replanning_df.loc[self.replanning_df['response_time'].idxmin(), 'algorithm']
            avg_response = self.replanning_df.groupby('algorithm')['response_time'].mean()
            
            insights.append(f'<div class="insight"><strong>Fastest Replanning:</strong> {fastest_replan} with average response time of {avg_response[fastest_replan]:.4f}s</div>')
        
        return '\n'.join(insights)

if __name__ == "__main__":
    # Example usage
    visualizer = EvacuationVisualizer()
    visualizer.load_results()
    visualizer.generate_report()
