import numpy as np


def compute_hpwl(nets, positions, chip_size):
    """
    Half-Perimeter Wire Length estimation.
    nets: list of lists of chip indices forming each net
    positions: dict {chiplet_id: (x, y, layer)}
    chip_size: (w, h) per chiplet
    """
    total_hpwl = 0.0
    for net in nets:
        xs = [positions[i][0] for i in net if i in positions]
        ys = [positions[i][1] for i in net if i in positions]
        if len(xs) < 2:
            continue
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total_hpwl += hpwl
    return total_hpwl


def normalize_hpwl(hpwl, canvas_w, canvas_h):
    return hpwl / (canvas_w + canvas_h)
