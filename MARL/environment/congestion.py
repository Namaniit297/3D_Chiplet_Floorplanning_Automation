import numpy as np


def compute_congestion(nets, positions, grid_size):
    """
    Estimate routing congestion using a grid-based demand map.
    Returns per-cell congestion map and scalar max congestion.
    """
    demand = np.zeros((grid_size, grid_size), dtype=np.float32)
    for net in nets:
        coords = [positions[i][:2] for i in net if i in positions]
        if len(coords) < 2:
            continue
        xs = [int(c[0]) for c in coords]
        ys = [int(c[1]) for c in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    demand[x, y] += 1.0
    return demand, float(demand.max())


def congestion_penalty(max_cong, threshold=5.0):
    if max_cong <= threshold:
        return 0.0
    return (max_cong - threshold) / threshold
