import numpy as np
import random


class AntColonyShortestPath:
    """ACO adapted for single-source to single-destination shortest path on a weighted graph.

    The graph is provided as a square adjacency matrix `graph` where
    `graph[i, j]` is the positive weight of edge i->j, or 0/np.inf for no edge.

    This implementation prevents revisiting nodes (no cycles) when building paths.
    Pheromone is stored per-edge and updated by ants that reach the target.
    """

    def __init__(self, graph, n_ants=20, n_iterations=200, decay=0.5, alpha=1.0, beta=2.0, seed=None, target_boost=None, boost_factor=10.0):
        self.graph = np.array(graph, dtype=float)
        if self.graph.ndim != 2 or self.graph.shape[0] != self.graph.shape[1]:
            raise ValueError("graph must be a square adjacency matrix")
        self.n_nodes = self.graph.shape[0]

        # mask of allowed moves: True where positive finite weight exists and not a self-loop
        self.allowed = np.isfinite(self.graph) & (self.graph > 0)
        np.fill_diagonal(self.allowed, False)

        # pheromone init (small positive), but zero where no edge
        self.pheromone = np.ones_like(self.graph) * 1e-6
        self.pheromone[~self.allowed] = 0.0
        
        # boost pheromone on edges leading to target node (optional)
        if target_boost is not None:
            for i in range(self.n_nodes):
                if self.allowed[i, target_boost]:
                    self.pheromone[i, target_boost] *= boost_factor

        # heuristic: inverse distance (avoid division by zero)
        eps = np.finfo(float).eps
        self.eta = np.zeros_like(self.graph)
        self.eta[self.allowed] = 1.0 / (self.graph[self.allowed] + eps)

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.random = random.Random(seed)
        self._np_random = np.random.RandomState(seed)

    def _construct_path(self, start, end):
        current = start
        path = [current]
        visited = {current}

        while current != end:
            # possible moves: neighbors j allowed and not visited
            candidates = [j for j in range(self.n_nodes) if self.allowed[current, j] and (j not in visited)]
            if not candidates:
                return None  # dead end

            tau = np.array([self.pheromone[current, j] for j in candidates]) ** self.alpha
            eta = np.array([self.eta[current, j] for j in candidates]) ** self.beta
            probs = tau * eta
            total = probs.sum()
            if total <= 0:
                # fallback: uniform among candidates
                choice = self.random.choice(candidates)
            else:
                probs = probs / total
                choice = int(self._np_random.choice(len(candidates), p=probs))
                choice = candidates[choice]

            path.append(choice)
            visited.add(choice)
            current = choice

            # safety: avoid too long paths
            if len(path) > self.n_nodes:
                return None

        return path

    def _path_cost(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.graph[path[i], path[i + 1]]
        return cost

    def run(self, start, end):
        """Run the ACO search from `start` to `end`. Returns (best_cost, best_path) or (inf, None)."""
        best_cost = float('inf')
        best_path = None

        for _ in range(self.n_iterations):
            all_paths = []
            all_costs = []
            for _ in range(self.n_ants):
                path = self._construct_path(start, end)
                if path is None:
                    continue
                cost = self._path_cost(path)
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path[:]

            # evaporate
            self.pheromone *= (1.0 - self.decay)

            # deposit pheromone for successful ants
            for path, cost in zip(all_paths, all_costs):
                deposit = 1.0 / (cost + 1e-12)
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    self.pheromone[a, b] += deposit

        return best_cost, best_path
