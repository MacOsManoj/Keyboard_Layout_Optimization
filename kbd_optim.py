#!/usr/bin/env python3
"""
Keyboard Layout Optimization via Simulated Annealing
"""

import argparse
import json
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

Point = Tuple[float, float]
Layout = Dict[str, Point]

def qwerty_coordinates(chars: str) -> Layout:
    """Return QWERTY grid coordinates for the provided character set."""
    row0 = "qwertyuiop"
    row1 = "asdfghjkl"
    row2 = "zxcvbnm"

    coords: Layout = {}
    for i, c in enumerate(row0):
        coords[c] = (float(i), 0.0)
    for i, c in enumerate(row1):
        coords[c] = (0.5 + float(i), 1.0)
    for i, c in enumerate(row2):
        coords[c] = (1.0 + float(i), 2.0)
    coords[" "] = (4.5, 3.0)

    # Backfill for requested chars; unknowns get space position.
    space_xy = coords[" "]
    for ch in chars:
        if ch not in coords:
            coords[ch] = space_xy
    return coords

def initial_layout() -> Layout:
    """Create an initial layout mapping chars to positions."""
    base_keys = "abcdefghijklmnopqrstuvwxyz "
    layout = qwerty_coordinates(base_keys)
    return layout

def preprocess_text(text: str, chars: str) -> str:
    """Lowercase and filter to the allowed character set; map others to space."""
    result = ""
    for c in text.lower():
        if c in chars:
            result += c
        else:
            result += " "
    return result

def path_length_cost(text: str, layout: Layout) -> float:
    """Sum Euclidean distances across consecutive characters in text."""
    if len(text) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i+1]
        x1, y1 = layout[c1]
        x2, y2 = layout[c2]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total += dist
    return total

def create_neighbor(layout: Layout, rng: random.Random) -> Layout:
    """Create neighbor by swapping two random characters."""
    chars = list(layout.keys())
    c1, c2 = rng.sample(chars, 2)
    
    new_layout = layout.copy()
    new_layout[c1], new_layout[c2] = new_layout[c2], new_layout[c1]
    return new_layout

@dataclass
class SAParams:
    iters: int = 50000
    t0: float = 1.0  # Initial temperature setting
    alpha: float = 0.999  # geometric decay per iteration
    epoch: int = 1  # iterations per temperature step

def simulated_annealing(
    text: str,
    layout: Layout,
    params: SAParams,
    rng: random.Random,
) -> Tuple[Layout, float, List[float], List[float]]:
    """Simulated annealing to minimize path-length cost over character swaps."""
    
    current = layout.copy()
    current_cost = path_length_cost(text, current)
    
    best = current.copy()
    best_cost = current_cost
    
    best_trace = []
    current_trace = []
    
    temp = params.t0
    
    for i in range(params.iters):
        # Generate neighbor
        neighbor = create_neighbor(current, rng)
        neighbor_cost = path_length_cost(text, neighbor)
        
        # Accept or reject
        delta = neighbor_cost - current_cost
        if delta < 0 or (temp > 0 and rng.random() < math.exp(-delta / temp)):
            current = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
        
        # Record traces
        best_trace.append(best_cost)
        current_trace.append(current_cost)
        
        # Cool down
        if (i + 1) % params.epoch == 0:
            temp *= params.alpha
    
    return best, best_cost, best_trace, current_trace

def plot_costs(
    layout: Layout, best_trace: List[float], current_trace: List[float]
) -> None:
    """Plot optimization traces and final layout."""
    
    out_dir = "."
    
    if best_trace and current_trace:
        plt.figure(figsize=(6, 3))
        plt.plot(best_trace, lw=1.5, label='Best')
        plt.plot(current_trace, lw=1.5, alpha=0.7, label='Current')
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost vs Iteration")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f"cost_trace.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    # Plot layout scatter
    xs, ys, labels = [], [], []
    for ch, (x, y) in layout.items():
        xs.append(x)
        ys.append(y)
        labels.append(ch)

    plt.figure(figsize=(6, 3))
    plt.scatter(xs, ys, s=250, c="#1f77b4")
    for x, y, ch in zip(xs, ys, labels):
        plt.text(
            x,
            y,
            ch,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="#1f77b4", ec="none", alpha=0.9),
        )
    plt.gca().invert_yaxis()
    plt.title("Optimized Layout")
    plt.axis("equal")
    plt.tight_layout()
    path = os.path.join(out_dir, f"layout.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def save_layout_json(layout: Layout, filename: str = "layout.json"):
    """Save layout to JSON file."""
    with open(filename, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Saved layout: {filename}")

def load_text(filename) -> str:
    """Load text from file or return default."""
    if filename is not None and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback demo text
    return (
        "the quick brown fox jumps over the lazy dog\n"
        "APL is the best course ever\n"
    )

def main(filename: str | None = None, iters: int = 50000, t0: float = 1.0, alpha: float = 0.999) -> None:
    rng = random.Random(42)
    chars = "abcdefghijklmnopqrstuvwxyz "

    # Initial assignment - QWERTY
    layout0 = initial_layout()

    # Prepare text and evaluate baseline
    raw_text = load_text(filename)
    text = preprocess_text(raw_text, chars)

    baseline_cost = path_length_cost(text, layout0)
    print(f"Baseline (QWERTY) cost: {baseline_cost:.4f}")
    print(f"Text length: {len(text)} chars")

    # Annealing parameters
    params = SAParams(iters=iters, t0=t0, alpha=alpha)
    start = time.time()
    best_layout, best_cost, best_trace, current_trace = simulated_annealing(text, layout0, params, rng)
    dur = time.time() - start
    
    improvement = baseline_cost - best_cost
    print(f"Optimized cost: {best_cost:.4f} (improvement: {improvement:.4f})")
    print(f"Runtime: {dur:.2f}s over {iters} iterations")

    # Save results
    save_layout_json(best_layout)
    plot_costs(best_layout, best_trace, current_trace)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keyboard Layout Optimization')
    parser.add_argument('--file', type=str, help='Input text file')
    parser.add_argument('--iters', type=int, default=50000, help='Number of iterations')
    parser.add_argument('--t0', type=float, default=1.0, help='Initial temperature')
    parser.add_argument('--alpha', type=float, default=0.999, help='Cooling rate')
    
    args = parser.parse_args()
    main(args.file, args.iters, args.t0, args.alpha)