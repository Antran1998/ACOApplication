
from aco_shortest_path import AntColonyShortestPath
import numpy as np


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


def main():
    m = 3  # grid size m x m
    # example obstacles: list of (row, col) or node indices
    obstacles = [(1, 0), (1, 1)]
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

    colony = AntColonyShortestPath(G, n_ants=10, n_iterations=200, decay=0.3, alpha=1.0, beta=2.0, seed=42)
    best_cost, best_path = colony.run(start, end)
    print('Grid size:', m, 'nodes:', m*m)
    if best_path is None:
        print('No path found')
    else:
        print('Best cost:', best_cost)
        print('Best path:', best_path)
        # save visualization
        try:
            plot_grid_path(m, best_path, filename='shortest_path.png')
            print('Saved visualization to shortest_path.png')
        except Exception as e:
            print('Failed to save visualization:', e)


if __name__ == '__main__':
    main()
