# Keyboard Layout Optimization

Optimizes keyboard layouts using **Simulated Annealing** to minimize total finger travel distance for a given text corpus. Starts from the standard QWERTY layout and iteratively swaps key positions, converging on a layout that reduces cumulative Euclidean distance between consecutive keystrokes.

## How It Works

1. **Initialize** with QWERTY key coordinates on a 2D grid (3 rows + spacebar).
2. **Preprocess** input text — lowercase, filter to `a-z` and space.
3. **Optimize** via simulated annealing:
   - **Neighbor generation**: randomly swap two key positions.
   - **Cost function**: sum of Euclidean distances between consecutive characters in the text.
   - **Acceptance**: Metropolis criterion — always accept improvements, probabilistically accept worse solutions (controlled by temperature).
   - **Cooling schedule**: geometric decay (`T ← T × α` every epoch).
4. **Output** the optimized layout as JSON, a layout visualization, and a cost convergence plot.

## Usage

```bash
python kbd_optim.py --file <text_file> --iters 50000 --t0 1.0 --alpha 0.999
```

| Argument  | Default | Description                  |
|-----------|---------|------------------------------|
| `--file`  | None    | Input text file (uses demo text if omitted) |
| `--iters` | 50000   | Number of SA iterations      |
| `--t0`    | 1.0     | Initial temperature          |
| `--alpha` | 0.999   | Geometric cooling rate       |

## Output

| File              | Description                              |
|-------------------|------------------------------------------|
| `layout.json`     | Optimized key → `[x, y]` coordinate map  |
| `layout.png`      | Scatter plot of the final key positions   |
| `cost_trace.png`  | Best & current cost vs. iteration plot    |

## Dependencies

- Python 3.10+
- `matplotlib`

## Project Structure

```
├── kbd_optim.py      # Main optimization script
├── layout.json       # Output: optimized layout coordinates
├── layout.png        # Output: layout visualization
├── cost_trace.png    # Output: convergence plot
└── readme.pdf        # Reference documentation
```

---
