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


def plot_grid_path(m, path, filename='shortest_path.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    coords = grid_coords(m, spacing=1.0)
    xs = coords[:, 0]
    ys = coords[:, 1]

    plt.figure(figsize=(6, 6))
    # draw grid edges lightly
    for i in range(len(coords)):
        r = i // m
        c = i % m
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < m and 0 <= cc < m:
                j = rr * m + cc
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='lightgray', linewidth=0.8)

    # plot nodes
    plt.scatter(xs, ys, c='black')
    for idx, (x, y) in enumerate(coords):
        plt.text(x + 0.05, y + 0.05, str(idx), fontsize=8)

    # plot path if available
    if path is not None:
        path_coords = coords[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], '-o', color='tab:blue', linewidth=2, markersize=6)
        # highlight start and end
        plt.scatter([path_coords[0, 0]], [path_coords[0, 1]], c='green', s=80, label='start')
        plt.scatter([path_coords[-1, 0]], [path_coords[-1, 1]], c='red', s=80, label='end')
        plt.legend()

    # if obstacles were provided via global variable, attempt to render them (optional)
    try:
        # look for a module-level variable `current_obstacles` that main may set
        from __main__ import current_obstacles
        if current_obstacles:
            obs_coords = coords[list(current_obstacles)]
            plt.scatter(obs_coords[:, 0], obs_coords[:, 1], c='0.2', marker='s', s=120, label='obstacle')
    except Exception:
        pass

    plt.gca().set_aspect('equal')
    plt.title('Shortest path found by ACO')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

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
        print(f"  Best Path: {results['best_path']}")
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
        
        print(f"\n    Fastest: {fastest['config_name']} ({fastest['mean_time']:.6f}s)")
        print(f"    Best Quality: {best_quality['config_name']} (cost: {best_quality['min_cost']:.4f})")


def main():
    m = 5  # grid size m x m (increased for better benchmarking)
    # example obstacles: list of (row, col) or node indices
    obstacles = [(1, 1), (2, 1), (2, 2), (3, 2)]
    G = grid_graph(m, obstacles=obstacles)
    start = 0
    end = m * m - 1

    # set a module-level variable so the plot function can optionally render obstacles
    try:
        global current_obstacles
        # convert obstacles to node indices set for plotting
        obs_set = set()
        for o in obstacles:
            if isinstance(o, tuple) or isinstance(o, list):
                obs_set.add(o[0] * m + o[1])
            else:
                obs_set.add(int(o))
        # ensure start/end are not obstacles
        obs_set.discard(start)
        obs_set.discard(end)
        current_obstacles = obs_set
    except Exception:
        current_obstacles = set()

    # quick connectivity check: if obstacles disconnect start->end, fall back to no obstacles
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
        print('Obstacles disconnect start and end; ignoring obstacles for this run.')
        obstacles = None
        current_obstacles = set()
        G = grid_graph(m)

    print(f'Grid size: {m}x{m} = {m*m} nodes')
    print(f'Start: {start}, End: {end}')
    print(f'Obstacles: {obstacles}')
    
    # Common parameters
    common_params = {
        'n_ants': 10,
        'n_iterations': 100,
        'decay': 0.3,
        'alpha': 1.0,
        'beta': 2.0
    }
    
    # Number of benchmark runs
    n_runs = 10
    
    print(f'\nRunning benchmarks with {n_runs} runs per configuration...')
    
    # Configuration 1: No boosting
    print('\n[1/3] Benchmarking: No Boosting...')
    results_no_boost = benchmark_aco(
        G, start, end,
        config_name="No Boosting (Baseline)",
        n_runs=n_runs,
        **common_params
    )
    print_benchmark_results(results_no_boost)
    
    # Configuration 2: Target boost only
    print('\n[2/3] Benchmarking: Target Boost Only...')
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
    print('\n[3/3] Benchmarking: Target + Neighbor Boost (default)...')
    results_neighbor_default = benchmark_aco(
        G, start, end,
        config_name="Target + Neighbor Boost (5.0)",
        n_runs=n_runs,
        target_boost=end,
        boost_factor=10.0,
        boost_neighbors=True,
        **common_params
    )
    print_benchmark_results(results_neighbor_default)
    
    # Configuration 4: Target + neighbor boost (custom factor)
    print('\n[4/4] Benchmarking: Target + Neighbor Boost (custom 3.0)...')
    results_neighbor_custom = benchmark_aco(
        G, start, end,
        config_name="Target + Neighbor Boost (3.0)",
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
    
    # Save visualization of best path from best configuration
    best_config = min(all_results, key=lambda x: x['min_cost'])
    if best_config['best_path'] is not None:
        try:
            plot_grid_path(m, best_config['best_path'], filename='best_path.png')
            print(f"\n   Saved visualization of best path to 'best_path.png'")
            print(f"   Configuration: {best_config['config_name']}")
        except Exception as e:
            print(f'Failed to save visualization: {e}')


if __name__ == '__main__':
    main()