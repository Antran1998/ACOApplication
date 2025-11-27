from aco_shortest_path import AntColonyShortestPath
import numpy as np
import time


def grid_graph(m, weight=1.0, obstacles=None):
    """Create an m x m grid graph adjacency matrix. Nodes numbered row-major.

    obstacles: optional iterable of node indices or (r,c) tuples indicating blocked cells.
    Blocked nodes will have no incoming or outgoing edges.
    """
    n = m * m
    G = np.full((n, n), np.inf)
    for r in range(m):
        for c in range(m):
            i = r * m + c
            # neighbors: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < m:
                    j = rr * m + cc
                    G[i, j] = weight

    # process obstacles
    if obstacles:
        obs_set = set()
        for o in obstacles:
            if isinstance(o, tuple) or isinstance(o, list):
                rr, cc = int(o[0]), int(o[1])
                if 0 <= rr < m and 0 <= cc < m:
                    obs_set.add(rr * m + cc)
            else:
                obs_set.add(int(o))

        for i in obs_set:
            if 0 <= i < n:
                # remove all edges to/from this node
                G[i, :] = np.inf
                G[:, i] = np.inf

    return G


def grid_coords(m, spacing=1.0):
    """Return coordinates (x,y) for each node in an m x m grid, row-major."""
    coords = []
    for r in range(m):
        for c in range(m):
            coords.append((c * spacing, r * spacing))
    return np.array(coords)


def plot_grid_path(m, path, obstacles=None, filename='shortest_path.png', title='Shortest path found by ACO'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    coords = grid_coords(m, spacing=1.0)
    xs = coords[:, 0]
    ys = coords[:, 1]

    plt.figure(figsize=(10, 10))
    # draw grid edges lightly
    for i in range(len(coords)):
        r = i // m
        c = i % m
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < m and 0 <= cc < m:
                j = rr * m + cc
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='lightgray', linewidth=0.8)

    # plot obstacles first
    if obstacles:
        obs_coords = coords[list(obstacles)]
        plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c='0.3', marker='s', s=200, label='obstacle', zorder=2)

    # plot nodes
    plt.scatter(xs, ys, c='white', edgecolors='black', s=50, zorder=3)
    for idx, (x, y) in enumerate(coords):
        if obstacles is None or idx not in obstacles:
            plt.text(x, y, str(idx), fontsize=6, ha='center', va='center', zorder=4)

    # plot path if available
    if path is not None:
        path_coords = coords[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], '-o', color='tab:blue', linewidth=2.5, markersize=8, zorder=5)
        # highlight start and end
        plt.scatter([path_coords[0, 0]], [path_coords[0, 1]], c='green', s=150, marker='*', label='start', zorder=6)
        plt.scatter([path_coords[-1, 0]], [path_coords[-1, 1]], c='red', s=150, marker='*', label='end', zorder=6)
        plt.legend(loc='upper right')

    plt.gca().set_aspect('equal')
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def create_maze_obstacles(m):
    """Create a dense maze-like pattern of obstacles."""
    obstacles = []
    # Create more complex maze with tighter walls
    for r in range(1, m-1):
        for c in range(1, m-1):
            # Create a checkerboard-like pattern with gaps
            if (r % 3 == 0 and c % 2 == 1) or (r % 3 == 2 and c % 2 == 0):
                obstacles.append((r, c))
            # Add extra vertical walls
            if c % 4 == 0 and r % 2 == 0:
                obstacles.append((r, c))
    return obstacles


def create_corridor_obstacles(m):
    """Create multiple narrow corridors forcing complex navigation."""
    obstacles = []
    # Create horizontal barriers with small gaps
    for r in [m // 4, m // 2, 3 * m // 4]:
        for c in range(m):
            # Leave only 2-3 gaps per barrier
            if c not in [m // 4, m // 2, 3 * m // 4]:
                obstacles.append((r, c))
    
    # Add vertical barriers
    for c in [m // 3, 2 * m // 3]:
        for r in range(m):
            # Leave gaps at different positions
            if r not in [m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5]:
                obstacles.append((r, c))
    
    return obstacles


def create_dense_obstacles(m, density=0.3):
    """Create randomly distributed dense obstacles with clusters."""
    np.random.seed(123)
    obstacles = []
    n = m * m
    
    # First pass: random obstacles
    for r in range(m):
        for c in range(m):
            if np.random.random() < density:
                idx = r * m + c
                if idx != 0 and idx != n - 1:  # Don't block start/end
                    obstacles.append((r, c))
    
    # Second pass: create obstacle clusters for extra difficulty
    num_clusters = max(3, m // 4)
    for _ in range(num_clusters):
        cluster_r = np.random.randint(2, m - 2)
        cluster_c = np.random.randint(2, m - 2)
        # Add 3x3 cluster with some gaps
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if np.random.random() < 0.7:  # 70% chance to block
                    new_r, new_c = cluster_r + dr, cluster_c + dc
                    if 0 <= new_r < m and 0 <= new_c < m:
                        idx = new_r * m + new_c
                        if idx != 0 and idx != n - 1:
                            obstacles.append((new_r, new_c))
    
    return obstacles


def create_spiral_obstacles(m):
    """Create a spiral pattern forcing a long winding path."""
    obstacles = []
    # Create walls that force a spiral path
    for r in range(2, m - 2, 3):
        for c in range(2, m - 2):
            obstacles.append((r, c))
    
    for c in range(2, m - 2, 3):
        for r in range(2, m - 2):
            obstacles.append((r, c))
    
    # Add additional blocking to force longer paths
    for r in range(1, m - 1, 4):
        for c in range(1, m - 1, 2):
            obstacles.append((r, c))
    
    return obstacles


def create_extreme_maze(m):
    """Create an extremely challenging maze with minimal paths."""
    obstacles = []
    # Dense wall pattern
    for r in range(m):
        for c in range(m):
            # Create walls with occasional gaps
            if (r % 2 == 1 and c % 3 != 1) or (c % 2 == 1 and r % 3 != 1):
                idx = r * m + c
                if idx != 0 and idx != m * m - 1:
                    obstacles.append((r, c))
    
    return obstacles


"""
    Benchmark a specific ACO configuration over multiple runs.
    
    Parameters:
    -----------
    G : array
        Graph adjacency matrix
    start : int
        Start node
    end : int
        End node
    config_name : str
        Name of the configuration for reporting
    n_runs : int
        Number of times to run the algorithm
    **aco_params : dict
        Parameters to pass to AntColonyShortestPath
    
    Returns:
    --------
    dict : Benchmark results including mean/std of time and cost
"""
def benchmark_aco(G, start, end, config_name, n_runs=10, **aco_params):

    times = []
    costs = []
    paths = []
    
    for run in range(n_runs):
        # Create new instance for each run with different seed
        colony = AntColonyShortestPath(G, seed=42 + run, **aco_params)
        
        start_time = time.perf_counter()
        best_cost, best_path = colony.run(start, end)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if best_path is not None:
            costs.append(best_cost)
            paths.append(best_path)
        else:
            costs.append(float('inf'))
            paths.append(None)
    
    # Calculate statistics
    valid_costs = [c for c in costs if c != float('inf')]
    
    results = {
        'config_name': config_name,
        'n_runs': n_runs,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_cost': np.mean(valid_costs) if valid_costs else float('inf'),
        'std_cost': np.std(valid_costs) if valid_costs else 0,
        'min_cost': np.min(valid_costs) if valid_costs else float('inf'),
        'max_cost': np.max(valid_costs) if valid_costs else float('inf'),
        'success_rate': len(valid_costs) / n_runs * 100,
        'best_path': paths[np.argmin(costs)] if valid_costs else None,
        'all_costs': costs,
        'all_times': times
    }
    
    return results


def print_benchmark_results(results):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*80}")
    print(f"Configuration: {results['config_name']}")
    print(f"{'='*80}")
    print(f"Runs: {results['n_runs']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"\nExecution Time (seconds):")
    print(f"  Mean:  {results['mean_time']:.6f} +/- {results['std_time']:.6f}")
    print(f"  Range: [{results['min_time']:.6f}, {results['max_time']:.6f}]")
    print(f"\nPath Cost:")
    print(f"  Mean:  {results['mean_cost']:.4f} +/- {results['std_cost']:.4f}")
    print(f"  Range: [{results['min_cost']:.4f}, {results['max_cost']:.4f}]")
    if results['best_path'] is not None:
        print(f"  Path Length: {len(results['best_path'])} nodes")
    print(f"{'='*80}")


def compare_results(all_results):
    """Generate a comparison table for all configurations."""
    print(f"\n\n{'='*100}")
    print(f"{'COMPARISON SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Configuration':<30} {'Avg Time (s)':<15} {'Avg Cost':<15} {'Min Cost':<15} {'Success %':<10}")
    print(f"{'-'*100}")
    
    for result in all_results:
        print(f"{result['config_name']:<30} "
              f"{result['mean_time']:<15.6f} "
              f"{result['mean_cost']:<15.4f} "
              f"{result['min_cost']:<15.4f} "
              f"{result['success_rate']:<10.1f}")
    
    print(f"{'='*100}")
    
    # Find best configurations
    valid_results = [r for r in all_results if r['success_rate'] > 0]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x['mean_time'])
        best_quality = min(valid_results, key=lambda x: x['min_cost'])
        most_consistent = min(valid_results, key=lambda x: x['std_cost'])
        
        print(f"\n    Fastest: {fastest['config_name']} ({fastest['mean_time']:.6f}s)")
        print(f"    Best Quality: {best_quality['config_name']} (cost: {best_quality['min_cost']:.4f})")
        print(f"    Most Consistent: {most_consistent['config_name']} (std: {most_consistent['std_cost']:.4f})")


def run_scenario(scenario_name, m, obstacles, start, end, n_runs=10):
    """Run benchmarks for a specific scenario."""
    print(f"\n{'#'*100}")
    print(f"# SCENARIO: {scenario_name}")
    print(f"{'#'*100}")
    
    G = grid_graph(m, obstacles=obstacles)
    
    # Connectivity check
    from collections import deque
    def connected(adj, s, t):
        n = adj.shape[0]
        seen = [False] * n
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            if u == t:
                return True
            for v in range(n):
                if np.isfinite(adj[u, v]) and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return False

    if not connected(G, start, end):
        print('   Obstacles disconnect start and end; skipping this scenario.')
        return None

    # Convert obstacles to node indices set for plotting
    obs_set = set()
    for o in obstacles:
        if isinstance(o, tuple) or isinstance(o, list):
            obs_set.add(o[0] * m + o[1])
        else:
            obs_set.add(int(o))
    obs_set.discard(start)
    obs_set.discard(end)

    print(f'Grid size: {m}x{m} = {m*m} nodes')
    print(f'Start: {start}, End: {end}')
    print(f'Obstacles: {len(obs_set)} nodes blocked ({len(obs_set)/(m*m)*100:.1f}% of grid)')
    
    # Common parameters - increased for harder scenarios
    common_params = {
        'n_ants': 20,
        'n_iterations': 200,
        'decay': 0.3,
        'alpha': 1.0,
        'beta': 2.0
    }
    
    print(f'\nRunning benchmarks with {n_runs} runs per configuration...')
    
    # Configuration 1: No boosting
    print('\n[1/4] Benchmarking: No Boosting...')
    results_no_boost = benchmark_aco(
        G, start, end,
        config_name="No Boosting",
        n_runs=n_runs,
        **common_params
    )
    print_benchmark_results(results_no_boost)
    
    # Configuration 2: Target boost only
    print('\n[2/4] Benchmarking: Target Boost Only...')
    results_target_only = benchmark_aco(
        G, start, end,
        config_name="Target Boost Only",
        n_runs=n_runs,
        target_boost=end,
        boost_factor=10.0,
        boost_neighbors=False,
        **common_params
    )
    print_benchmark_results(results_target_only)
    
    # Configuration 3: Target + neighbor boost (default factor)
    print('\n[3/4] Benchmarking: Target + Neighbor Boost (5.0)...')
    results_neighbor_default = benchmark_aco(
        G, start, end,
        config_name="Target + Neighbor (5.0)",
        n_runs=n_runs,
        target_boost=end,
        boost_factor=10.0,
        boost_neighbors=True,
        **common_params
    )
    print_benchmark_results(results_neighbor_default)
    
    # Configuration 4: Target + neighbor boost (custom factor)
    print('\n[4/4] Benchmarking: Target + Neighbor Boost (3.0)...')
    results_neighbor_custom = benchmark_aco(
        G, start, end,
        config_name="Target + Neighbor (3.0)",
        n_runs=n_runs,
        target_boost=end,
        boost_factor=10.0,
        boost_neighbors=True,
        neighbor_boost_factor=3.0,
        **common_params
    )
    print_benchmark_results(results_neighbor_custom)
    
    # Compare all results
    all_results = [
        results_no_boost,
        results_target_only,
        results_neighbor_default,
        results_neighbor_custom
    ]
    compare_results(all_results)
    
    # Save visualizations for each configuration
    for i, result in enumerate(all_results):
        if result['best_path'] is not None:
            try:
                filename = f"{scenario_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('%', 'pct')}_config_{i+1}.png"
                title = f"{scenario_name}\n{result['config_name']} - Cost: {result['min_cost']:.2f}"
                plot_grid_path(m, result['best_path'], obstacles=obs_set, filename=filename, title=title)
            except Exception as e:
                print(f'Failed to save visualization for {result["config_name"]}: {e}')
    
    print(f"\n   Completed scenario: {scenario_name}")
    return all_results


def main():
    """Run multiple complex scenarios to observe ACO performance."""
    
    print("="*100)
    print("ACO SHORTEST PATH - EXTREME DIFFICULTY BENCHMARK SUITE")
    print("="*100)
    
    n_runs = 15  # Number of runs per configuration
    
    # Scenario 1: Dense maze with many obstacles
    scenario1_results = run_scenario(
        scenario_name="Dense Maze (15x15, Complex Pattern)",
        m=15,
        obstacles=create_maze_obstacles(15),
        start=0,
        end=224,
        n_runs=n_runs
    )
    
    # Scenario 2: Multiple narrow corridors
    scenario2_results = run_scenario(
        scenario_name="Multi-Corridor Maze (15x15)",
        m=15,
        obstacles=create_corridor_obstacles(15),
        start=0,
        end=224,
        n_runs=n_runs
    )
    
    # Scenario 3: Very dense random obstacles with clusters
    scenario3_results = run_scenario(
        scenario_name="Dense Clustered Obstacles (18x18, 40% blocked)",
        m=18,
        obstacles=create_dense_obstacles(18, density=0.40),
        start=0,
        end=323,
        n_runs=n_runs
    )
    
    # Scenario 4: Spiral pattern forcing long paths
    scenario4_results = run_scenario(
        scenario_name="Spiral Maze (20x20)",
        m=20,
        obstacles=create_spiral_obstacles(20),
        start=0,
        end=399,
        n_runs=n_runs
    )
    
    # Scenario 5: Extreme maze - minimal viable paths
    scenario5_results = run_scenario(
        scenario_name="Extreme Maze (16x16, Minimal Paths)",
        m=16,
        obstacles=create_extreme_maze(16),
        start=0,
        end=255,
        n_runs=n_runs
    )
    
    # Scenario 6: Large grid with very dense obstacles
    scenario6_results = run_scenario(
        scenario_name="Large Dense Grid (25x25, 35% blocked)",
        m=25,
        obstacles=create_dense_obstacles(25, density=0.35),
        start=0,
        end=624,
        n_runs=n_runs
    )
    
    # Overall summary across all scenarios
    print(f"\n\n{'='*100}")
    print(f"{'OVERALL SUMMARY ACROSS ALL SCENARIOS':^100}")
    print(f"{'='*100}")
    
    all_scenarios = [
        ("Dense Maze", scenario1_results),
        ("Multi-Corridor Maze", scenario2_results),
        ("Dense Clustered Obstacles", scenario3_results),
        ("Spiral Maze", scenario4_results),
        ("Extreme Maze", scenario5_results),
        ("Large Dense Grid", scenario6_results)
    ]
    
    for scenario_name, results in all_scenarios:
        if results is None:
            print(f"\n {scenario_name}: SKIPPED (disconnected graph)")
            continue
            
        print(f"\n {scenario_name}:")
        valid_results = [r for r in results if r['success_rate'] > 0]
        if valid_results:
            best = min(valid_results, key=lambda x: x['min_cost'])
            fastest = min(valid_results, key=lambda x: x['mean_time'])
            print(f"   Best Solution: {best['config_name']} (cost: {best['min_cost']:.4f})")
            print(f"   Fastest: {fastest['config_name']} (time: {fastest['mean_time']:.6f}s)")
            
            # Calculate improvement percentage
            baseline = [r for r in results if r['config_name'] == 'No Boosting'][0]
            if baseline['min_cost'] != float('inf') and best['min_cost'] < baseline['min_cost']:
                improvement = ((baseline['min_cost'] - best['min_cost']) / baseline['min_cost']) * 100
                print(f"   Improvement over baseline: {improvement:.2f}%")
        else:
            print(f"    No successful paths found")
    
    print(f"\n{'='*100}")
    print("   Benchmark suite completed! Check generated PNG files for visualizations.")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()