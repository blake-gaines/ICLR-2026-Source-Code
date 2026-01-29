import networkx as nx
from tqdm.auto import tqdm
import sys


def get_diameter_lb(G, k=50, position=None, **kwargs):
    """
    Compute a lower bound on the diameter of the graph

    Args:
        G: NetworkX graph
        k: Number of random trials
        position: Position of the progress bar
        **kwargs: Additional arguments to pass to tqdm

    Returns:
        float: Lower bound on the diameter of the graph
    """
    return max(
        nx.algorithms.approximation.diameter(G, seed=i)
        for i in tqdm(
            range(k),
            desc=f"Worker {position:02d} - Computing LB" if position is not None else "Computing LB",
            leave=False,
            position=position + 1 if position is not None else 0,
            delay=1,
            mininterval=1,
            bar_format="{l_bar}|{bar}| {n_fmt:>4}/{total_fmt} {elapsed}<{remaining:<8}"
            + f" | {G.number_of_nodes():^8} Nodes, {G.number_of_edges():^8} Edges",
            **kwargs,
        )
    )


def get_diameter_ub(G, k=50, position=None, **kwargs):
    """
    Compute an upper bound on the diameter of the graph

    Args:
        G: NetworkX graph
        k: Number of random trials
        position: Position of the progress bar
        **kwargs: Additional arguments to pass to tqdm

    Returns:
        float: Upper bound on the diameter of the graph
    """
    node_degrees = list(G.degree())
    sorted_degrees = sorted(G.nodes, key=lambda i: node_degrees[i], reverse=True)
    depth_limit = None
    min_ub = float("inf")
    for i in tqdm(
        range(k),
        desc=f"Worker {position:02d} - Computing UB" if position is not None else "Computing UB",
        leave=False,
        position=position if position is not None else 0,
        delay=1,
        mininterval=1,
        bar_format="{l_bar}|{bar}| {n_fmt:>4}/{total_fmt} {elapsed}<{remaining:<8}"
        + f" | {G.number_of_nodes():^8} Nodes, {G.number_of_edges():^8} Edges",
        **kwargs,
    ):
        tree = nx.bfs_tree(G, source=sorted_degrees[i % len(sorted_degrees)], depth_limit=depth_limit).to_undirected()
        ub = nx.algorithms.approximation.diameter(tree, seed=1)
        if ub < min_ub:
            min_ub = ub
            depth_limit = ub
    return min_ub


if __name__ == "__main__":
    import sys
    import os
    import multiprocessing as mp
    import pickle

    progress_bar_width = 140

    manager = mp.Manager()
    position_counter = manager.Value("i", 1)
    worker_positions = manager.dict()
    position_lock = manager.Lock()

    def get_worker_position():
        """Assign a unique position to each worker process."""
        process_name = mp.current_process().name
        if process_name not in worker_positions:
            with position_lock:
                if process_name not in worker_positions:
                    pos = position_counter.value
                    position_counter.value += 1
                    worker_positions[process_name] = pos
        return worker_positions[process_name]

    def compute_diameter_bounds(file):
        try:
            with open(file, "rb") as f:
                result = pickle.load(f)
        except Exception as e:
            return {
                "file": file,
                "error": f"Unexpected error loading file: {type(e).__name__}: {e}",
                "skipped": True,
            }
        try:
            if "Diameter LB" in result and "Diameter UB" in result:
                return result

            G = result["Dual Graph"]

            result["Diameter LB"] = get_diameter_lb(G, position=None, ncols=progress_bar_width, disable=True)
            result["Diameter UB"] = get_diameter_ub(G, position=None, ncols=progress_bar_width, disable=True)

            with open(file, "wb") as f:
                pickle.dump(result, f)

            return result
        except Exception as e:
            return {"file": file, "error": f"Unexpected error: {type(e).__name__}: {e}", "skipped": True}

    log_dirs = sys.argv[1:] if len(sys.argv) > 2 else ["results/"]

    all_files = []
    for log_dir in log_dirs:
        shi_counts_dir = os.path.join(log_dir, "shi_counts")
        if not os.path.exists(shi_counts_dir):
            print(f"Warning: Directory does not exist: {shi_counts_dir}")
            continue
        files = os.listdir(shi_counts_dir)
        all_files.extend(
            [
                os.path.join(shi_counts_dir, f)
                for f in files
                if f.endswith(".pkl")  # Only process pickle files
            ]
        )

    nworkers = os.process_cpu_count()

    print(f"Processing {len(all_files)} graphs with {nworkers} workers")
    print()

    with mp.Pool(processes=nworkers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(compute_diameter_bounds, all_files),
                total=len(all_files),
                desc="Computing Diameter Bounds",
                position=0,
                ncols=progress_bar_width,
                bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt}{postfix}",
            )
        )

    print("Done")
