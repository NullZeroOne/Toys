#!/usr/bin/env python3
"""
Backwards Life on a Torus — Growing/Shrinking Sizes, Caching, and JSON Summary
------------------------------------------------------------------------------
Features:
- Searches 1-step predecessors of a target on a toroidal (wrap-around) grid.
- Supports explicit size lists:   --sizes "3x4,4x4,5x4"
- Supports min/max rectangular range: --min-size "RxC" --max-size "RxC"
- Semantics for min size:
    * First predecessor search (gen = -1) starts at the user-specified min.
    * Subsequent generations start at 3x3 (then grow toward the max).
- Attempts and predecessors are logged with wall/CPU times and memory.
- Emoji-table summaries for attempts, predecessors, and run totals.
- In-run cache of solved subproblems keyed by (R,C,canonicalized target under torus translations).
- Pretty JSON summary written to /mnt/data/backwards_life_summary.json (or current dir if sandbox path not present)
  including:
    * last predecessor found (RLE + size),
    * sizes that were conclusively UNSAT for its parent,
    * suggested next command line.

Usage examples:
  python backwards_life_torus_growing.py --seconds 30 --min-size 3x4 --max-size 4x5
  python backwards_life_torus_growing.py --seconds 30 --sizes "3x4,4x4,5x4"
  python backwards_life_torus_growing.py --seconds 30 --start-rle "x = 3, y = 3, rule = B3/S23\nbo$o$3o!" --max-size 6x6
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

# ---------- Life rules & helpers ----------
def life_next_cell(center: int, neighbors_sum: int) -> int:
    # Conway's Life: B3/S23
    if center == 1:
        return 1 if neighbors_sum in (2,3) else 0
    else:
        return 1 if neighbors_sum == 3 else 0

def evolve_torus(grid: List[List[int]]) -> List[List[int]]:
    n, m = len(grid), len(grid[0])
    out = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==0 and dj==0: continue
                    s += grid[(i+di)%n][(j+dj)%m]
            out[i][j] = 1 if (grid[i][j]==1 and s in (2,3)) or (grid[i][j]==0 and s==3) else 0
    return out

# Precompute allowed 3x3 neighborhoods for next-bit 0/1
def all_neighborhoods_for_next(next_bit: int) -> Set[int]:
    allowed = set()
    for mask in range(1<<9):  # 9 bits (3x3), row-major; bit 4 is center
        bits = [(mask>>i)&1 for i in range(9)]
        center = bits[4]
        neighbors_sum = sum(bits) - center
        if life_next_cell(center, neighbors_sum) == next_bit:
            allowed.add(mask)
    return allowed

ALLOWED_FOR_NEXT = {0: all_neighborhoods_for_next(0), 1: all_neighborhoods_for_next(1)}

def wrap(i, n): return i % n

def neighborhood_mask_known(grid, r, c) -> Tuple[int, bool]:
    """Return (mask, known) for 3x3 neighborhood centered at (r,c).
       If any cell is unknown (-1), known=False and unknowns treated as 0 in mask."""
    n, m = len(grid), len(grid[0])
    mask = 0
    known = True
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            rr, cc = wrap(r+dr,n), wrap(c+dc,m)
            v = grid[rr][cc]
            if v == -1:
                known = False
                v = 0
            mask = (mask<<1) | v
    return mask, known

def quick_local_prune(grid, target, r, c) -> bool:
    """After assigning (r,c), validate any *fully known* affected neighborhoods."""
    n, m = len(grid), len(grid[0])
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            rr, cc = wrap(r+dr, n), wrap(c+dc, m)
            mask, known = neighborhood_mask_known(grid, rr, cc)
            if known:
                if mask not in ALLOWED_FOR_NEXT[target[rr][cc]]:
                    return False
    return True

def solve_predecessor_torus(target: List[List[int]], time_limit: float, seed_center_first: bool=True):
    """Search for any 1-step predecessor on same-size torus.
       Returns ('FOUND', grid) or ('UNSAT', None) or ('TIMEOUT', None)."""
    n, m = len(target), len(target[0])
    grid = [[-1]*m for _ in range(n)]
    cells = [(i,j) for i in range(n) for j in range(m)]
    start = time.time()

    # Heuristic: try cells near target's live centroid first
    if seed_center_first:
        live_cells = [(i,j) for i in range(n) for j in range(m) if target[i][j]==1]
        if live_cells:
            ci = sum(i for i,_ in live_cells)/len(live_cells)
            cj = sum(j for _,j in live_cells)/len(live_cells)
            cells.sort(key=lambda rc: (abs(rc[0]-ci)+abs(rc[1]-cj)))

    solution = None
    explored_timeout = False

    def backtrack(k=0) -> bool:
        nonlocal solution, explored_timeout
        if time.time() - start >= time_limit:
            explored_timeout = True
            return False
        if k == len(cells):
            solution = [row[:] for row in grid]
            return True
        i,j = cells[k]
        if grid[i][j] != -1:
            return backtrack(k+1)
        # Try 0 then 1
        for v in (0,1):
            grid[i][j] = v
            if quick_local_prune(grid, target, i, j):
                if backtrack(k+1):
                    return True
        grid[i][j] = -1
        return False

    ok = backtrack(0)
    if ok:
        return ('FOUND', solution)
    elif explored_timeout:
        return ('TIMEOUT', None)
    else:
        return ('UNSAT', None)

# ---------- RLE I/O and pretty printing ----------
def grid_to_rle(grid: List[List[int]], rule="B3/S23") -> str:
    n, m = len(grid), len(grid[0])
    header = f"x = {m}, y = {n}, rule = {rule}"
    rows_rle = []
    for i in range(n):
        row = []
        run_char = None
        run_len = 0
        for j in range(m):
            ch = 'o' if grid[i][j]==1 else 'b'
            if run_char is None:
                run_char, run_len = ch, 1
            elif ch == run_char:
                run_len += 1
            else:
                row.append((run_char, run_len))
                run_char, run_len = ch, 1
        if run_char is not None:
            row.append((run_char, run_len))
        # Trim trailing 'b' group
        if row and row[-1][0] == 'b':
            row.pop()
        rows_rle.append(''.join((str(cnt) if cnt>1 else '') + ch for ch, cnt in row))
    body = '$'.join(rows_rle) + '!'
    return header + "\n" + body

def rle_to_grid(rle: str) -> List[List[int]]:
    import re
    lines = [ln.strip() for ln in rle.strip().splitlines() if ln.strip() and not ln.strip().startswith('#')]
    header = lines[0]
    m = re.search(r'x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)', header)
    if not m: raise ValueError("RLE header missing x,y")
    cols, rows = int(m.group(1)), int(m.group(2))
    body = ''.join(lines[1:])
    if '!' in body:
        body = body[:body.index('!')]
    grid = [[0]*cols for _ in range(rows)]
    r = c = 0
    num = ''
    i = 0
    while i < len(body):
        ch = body[i]
        if ch.isdigit():
            num += ch
        elif ch in ('b','o','$'):
            count = int(num) if num else 1
            num = ''
            if ch == 'b' or ch == 'o':
                for _ in range(count):
                    if c >= cols:
                        r += 1; c = 0
                    if r < rows:
                        grid[r][c] = 1 if ch == 'o' else 0
                    c += 1
            elif ch == '$':
                r += count
                c = 0
        i += 1
    return grid

def print_emoticon_grid(grid: List[List[int]], live='🟩', dead='⬜') -> None:
    for row in grid:
        print(''.join(live if v==1 else dead for v in row))

# ---------- Memory usage helpers ----------
def format_bytes(n: int) -> str:
    if n is None:
        return "-"
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.2f} {unit}" if unit!='B' else f"{n} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def get_memory_usage():
    """Return (rss_bytes, peak_rss_bytes or None) using psutil if available, else resource (Unix), else /proc."""
    rss = None
    peak = None
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        mem = p.memory_info()
        rss = mem.rss
        # On Windows psutil, 'peak_wset' may exist
        peak = getattr(mem, 'peak_wset', None)
    except Exception:
        try:
            import resource  # type: ignore
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss: kilobytes on Linux/macOS
            peak = int(ru.ru_maxrss) * 1024
        except Exception:
            pass
    if rss is None:
        try:
            with open("/proc/self/statm") as f:
                parts = f.read().split()
                pages_rss = int(parts[1])
                import os
                rss = pages_rss * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            pass
    return rss, peak

# ---------- Acorn-in-3x3 helper ----------
def acorn_wrapped_3x3() -> List[List[int]]:
    # Standard Acorn in a 3x7 box:
    rows = [
        ".*.....",
        "...*...",
        "**..***",
    ]
    R, C = 3, 3
    grid = [[0]*C for _ in range(R)]
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == '*':
                grid[i % R][j % C] = 1
    return grid

# ---------- Size helpers ----------
def parse_size(s: str) -> Tuple[int,int]:
    try:
        r, c = s.lower().split('x')
        return int(r), int(c)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}'. Use RxC like 3x4.")

def resize_target_torus_sum(target: List[List[int]], R: int, C: int) -> List[List[int]]:
    """Project target onto an R x C torus by modulo-indexing and OR-summing overlapping cells."""
    r0, c0 = len(target), len(target[0])
    out = [[0]*C for _ in range(R)]
    for i in range(r0):
        for j in range(c0):
            if target[i][j]:
                out[i % R][j % C] = 1
    return out

# ---------- Canonicalization & cache ----------
def torus_translations(R: int, C: int):
    for dr in range(R):
        for dc in range(C):
            yield dr, dc

def translate_grid(grid: List[List[int]], dr: int, dc: int) -> List[List[int]]:
    R, C = len(grid), len(grid[0])
    out = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            out[i][j] = grid[(i+dr)%R][(j+dc)%C]
    return out

def grid_to_bitstring(grid: List[List[int]]) -> str:
    return ''.join('1' if v else '0' for row in grid for v in row)

def canonicalize_torus(grid: List[List[int]]) -> str:
    R, C = len(grid), len(grid[0])
    best = None
    for dr, dc in torus_translations(R, C):
        g2 = translate_grid(grid, dr, dc)
        s = grid_to_bitstring(g2)
        if best is None or s < best:
            best = s
    return best  # canonical key string

# ---------- Logging dataclasses ----------
@dataclass
class FoundInfo:
    gen: int
    rows: int
    cols: int
    wall_seconds: float
    cpu_seconds: float

@dataclass
class AttemptInfo:
    gen: int
    rows: int
    cols: int
    status: str   # 'FOUND' | 'UNSAT' | 'TIMEOUT'
    wall_seconds: float
    cpu_seconds: float
    rss_bytes: Optional[int]
    peak_bytes: Optional[int]

# ---------- Emoticon table formatting ----------
def pad(s, w):
    s = str(s)
    return s + ' ' * max(0, w - len(s))

def emoji_status(status: str) -> str:
    return {'FOUND': '✅', 'UNSAT': '❌', 'TIMEOUT': '⏳', 'CACHED': '🗂️'}.get(status, '❓')

def emoji_bytes(n):
    return format_bytes(n) if n is not None else '-'

def print_emoji_table(title: str, headers, rows):
    # Compute column widths
    cols = len(headers)
    widths = [len(h) for h in headers]
    # Normalize row entries to strings first
    str_rows = [[str(c) for c in r] for r in rows]
    for r in str_rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))
    # Banner
    print(f"== {title} ==")
    # Header
    print(' '.join(pad(headers[i], widths[i]) for i in range(cols)))
    # Underline
    print(' '.join('─'*widths[i] for i in range(cols)))
    # Rows
    for r in str_rows:
        print(' '.join(pad(r[i], widths[i]) for i in range(cols)))
    print()

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Iteratively search for 1-step predecessors on a torus, with caching and summaries.")
    ap.add_argument("--seconds", type=float, required=True, help="Total wall-clock seconds to spend (hard limit).")
    ap.add_argument("--max", type=int, default=None, help="Maximum rows and cols (square cap). Overrides --max-rows/--max-cols if set.")
    ap.add_argument("--max-rows", type=int, default=None, help="Maximum rows; default = start rows if not set.")
    ap.add_argument("--max-cols", type=int, default=None, help="Maximum cols; default = start cols if not set.")
    ap.add_argument("--min-size", type=str, default=None, help="Optional minimum size as RxC (e.g., 3x4).")
    ap.add_argument("--max-size", type=str, default=None, help="Optional maximum size as RxC (e.g., 5x4).")
    ap.add_argument("--sizes", type=str, default=None, help="Comma-separated explicit sizes like 3x4,4x4,5x4 (takes precedence).")
    ap.add_argument("--start-rle", type=str, default=None, help="Optional starting pattern in RLE (overrides default).")
    ap.add_argument("--live", type=str, default="🟩", help="Emoticon for live cell (default 🟩).")
    ap.add_argument("--dead", type=str, default="⬜", help="Emoticon for dead cell (default ⬜).")
    return ap.parse_args()

def main():
    args = parse_args()
    t0 = time.time()
    cpu0 = time.process_time()

    # Build starting grid
    if args.start_rle:
        start_grid = rle_to_grid(args.start_rle)
    else:
        start_grid = [row[:] for row in acorn_wrapped_3x3()]

    # Print program arguments
    print("=== Program arguments ===")
    print(args)
    print()

    current = start_grid
    gen = -1
    found_log: List[FoundInfo] = []
    attempt_log: List[AttemptInfo] = []
    cache = {}  # (R,C,canonical_target) -> {'status': 'UNSAT'|'FOUND', 'pred': [[...]] or None}

    start_rows, start_cols = len(start_grid), len(start_grid[0])

    # Pretty-print the starting target
    print("=== Starting target (generation 0, will search for generation -1) ===")
    print_emoticon_grid(start_grid, live=args.live, dead=args.dead)
    print("\nRLE:\n" + grid_to_rle(start_grid) + "\n")

    last_found_grid = None
    last_found_rle = None

    while True:
        elapsed = time.time() - t0
        if elapsed >= args.seconds:
            print(f"[timeout] Stopping after {elapsed:.2f}s.")
            break

        # Derive list of sizes to try (per generation)
        sizes_to_try = []
        # Explicit list takes precedence
        if args.sizes:
            for item in args.sizes.split(','):
                item = item.strip()
                if not item:
                    continue
                sizes_to_try.append(parse_size(item))
        else:
            # Bounds
            if args.min_size:
                min_rows_cfg, min_cols_cfg = parse_size(args.min_size)
            else:
                min_rows_cfg, min_cols_cfg = start_rows, start_cols

            if args.max_size:
                max_rows_cfg, max_cols_cfg = parse_size(args.max_size)
            else:
                if args.max is not None:
                    max_rows_cfg = max(args.max, start_rows)
                    max_cols_cfg = max(args.max, start_cols)
                else:
                    max_rows_cfg = max(args.max_rows or start_rows, start_rows)
                    max_cols_cfg = max(args.max_cols or start_cols, start_cols)

            # For the first predecessor (gen == -1), use the configured min size; for later gens, use 3x3 as min.
            if gen == -1:
                min_rows = max(1, min_rows_cfg)
                min_cols = max(1, min_cols_cfg)
            else:
                min_rows = 3
                min_cols = 3

            max_rows = max(min_rows, max_rows_cfg)
            max_cols = max(min_cols, max_cols_cfg)

            for R in range(min_rows, max_rows+1):
                for C in range(min_cols, max_cols+1):
                    sizes_to_try.append((R, C))

        # Search across configured sizes (explicit or min/max grid)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        last_target_used = None

        found = None
        # Per-attempt time cap to maintain responsiveness inside total budget
        per_attempt_cap = 5.0

        for (R, C) in sizes_to_try:
            remaining = args.seconds - (time.time() - t0)
            if remaining <= 0:
                print(f"[timeout] Stopping at generation {gen+1} (no more results within time).")
                break
            attempt_budget = max(0.1, min(remaining, per_attempt_cap))
            print(f"[try] size {R}x{C} (budget {attempt_budget:.2f}s)")

            # Project current target to this torus size (allows shrinking or growing)
            target_RC = resize_target_torus_sum(current, R, C)
            key = (R, C, canonicalize_torus(target_RC))
            cached = cache.get(key)

            _wall0 = time.perf_counter(); _cpu0 = time.process_time()
            if cached is not None:
                status = cached['status']
                pred = cached.get('pred')
            else:
                status, pred = solve_predecessor_torus(target_RC, time_limit=attempt_budget, seed_center_first=True)
                # Update cache
                if status == 'FOUND':
                    cache[key] = {'status': 'FOUND', 'pred': [row[:] for row in pred] if pred else None}
                elif status == 'UNSAT':
                    cache[key] = {'status': 'UNSAT', 'pred': None}
            _wall = time.perf_counter() - _wall0; _cpu = time.process_time() - _cpu0
            _rss, _peak = get_memory_usage()

            attempt_log.append(AttemptInfo(gen=gen, rows=R, cols=C, status=status, wall_seconds=_wall, cpu_seconds=_cpu, rss_bytes=_rss, peak_bytes=_peak))

            if status == 'FOUND':
                print(f"[found] predecessor at {R}x{C} in {_wall:.2f}s (CPU {_cpu:.2f}s)")
                found = pred
                last_target_used = target_RC
                break
            elif status == 'UNSAT':
                print(f"[unsat] no predecessor at {R}x{C} (took {_wall:.2f}s, CPU {_cpu:.2f}s)")
            elif status == 'TIMEOUT':
                print(f"[timeout] size {R}x{C} after {_wall:.2f}s (CPU {_cpu:.2f}s)")

        if found is None:
            print(f"[done] No predecessor exists within size limits for generation {gen+1}. Stopping.")
            break

        # Compute timing for this found predecessor
        wall_elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start

        # Print predecessor found
        print(f"=== Found predecessor for generation {gen} (this is generation {gen}) ===")
        print_emoticon_grid(found, live=args.live, dead=args.dead)
        found_rle = grid_to_rle(found)
        print("\nRLE:\n" + found_rle + "\n")

        # Log metadata
        found_log.append(FoundInfo(gen=gen, rows=len(found), cols=len(found[0]), wall_seconds=wall_elapsed, cpu_seconds=cpu_elapsed))
        last_found_grid = [row[:] for row in found]
        last_found_rle = found_rle

        # Sanity check: evolve forward once must match the exact target used at that size
        if last_target_used is not None and evolve_torus(found) != last_target_used:
            print("[warning] Sanity check failed: evolved predecessor didn't match target at that size.")

        # Move back one step
        current = found
        gen -= 1

    # === Summary ===
    # Attempts table
    if attempt_log:
        attempt_rows = []
        for a in attempt_log:
            attempt_rows.append([
                a.gen, f"{a.rows}x{a.cols}", emoji_status(a.status),
                f"{a.wall_seconds:.2f}s", f"{a.cpu_seconds:.2f}s",
                emoji_bytes(a.rss_bytes), emoji_bytes(a.peak_bytes)
            ])
        print_emoji_table("Attempts (per size)", ["gen", "size", "status", "wall", "cpu", "rss", "peak"], attempt_rows)
    else:
        print("== Attempts (per size) ==")
        print("[summary] No attempts logged.")
        print()

    # Predecessors table
    if found_log:
        rows = []
        cum_w = 0.0
        cum_c = 0.0
        for rec in found_log:
            cum_w += rec.wall_seconds
            cum_c += rec.cpu_seconds
            rows.append([rec.gen, f"{rec.rows}x{rec.cols}", f"{rec.wall_seconds:.2f}s", f"{rec.cpu_seconds:.2f}s", f"{cum_w:.2f}s", f"{cum_c:.2f}s"])
        print_emoji_table("Predecessors found", ["gen","size","wall","cpu","cum_wall","cum_cpu"], rows)
    else:
        print("== Predecessors found ==")
        print("[summary] No predecessors found.")
        print()

    # Global totals
    total_wall = time.time() - t0
    total_cpu = time.process_time() - cpu0
    rss, peak = get_memory_usage()
    totals_rows = [[f"{total_wall:.2f}s", f"{total_cpu:.2f}s", emoji_bytes(rss), emoji_bytes(peak)]]
    print_emoji_table("Run totals", ["wall","cpu","rss","peak"], totals_rows)

    # JSON output
    # Build JSON-per-run in a human-readable way.
    summary = {
        "args": {
            "seconds": args.seconds,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "sizes": args.sizes,
        },
        "last_predecessor": {
            "rle": last_found_rle if last_found_rle is not None else None,
            "size": ({"rows": len(last_found_grid), "cols": len(last_found_grid[0])} if last_found_grid is not None else None),
        },
        "exhaustively_unsat_for_last_predecessor": [],
        "next_command": None,
    }

    if last_found_grid is not None and found_log:
        last_gen = found_log[-1].gen
        next_gen = last_gen - 1
        unsat = []
        for a in attempt_log:
            if a.gen == next_gen and a.status == 'UNSAT':
                unsat.append({"rows": a.rows, "cols": a.cols})
        summary["exhaustively_unsat_for_last_predecessor"] = unsat

        # Choose next min as the smallest size >= 3x3 that was not UNSAT for next-gen attempts (within provided max)
        candidate_sizes = set()
        if args.sizes:
            for item in args.sizes.split(','):
                item = item.strip()
                if item:
                    r,c = parse_size(item)
                    candidate_sizes.add((r,c))
        else:
            if args.max_size:
                rmax, cmax = parse_size(args.max_size)
            else:
                # If no --max-size provided, fall back to the last predecessor's size as cap
                rmax, cmax = len(last_found_grid), len(last_found_grid[0])
            for R in range(3, rmax+1):
                for C in range(3, cmax+1):
                    candidate_sizes.add((R,C))
        unsat_set = {(x["rows"], x["cols"]) for x in unsat}
        remaining = sorted(s for s in candidate_sizes if s not in unsat_set)
        next_min = f"{remaining[0][0]}x{remaining[0][1]}" if remaining else "3x3"

        # Build suggested next command
        pieces = ["python", "backwards_life_torus_growing.py", f"--seconds {int(args.seconds)}"]
        pieces.append(f'--start-rle "{last_found_rle.strip()}"')
        pieces.append(f"--min-size {next_min}")
        if args.max_size:
            pieces.append(f"--max-size {args.max_size}")
        elif args.max or args.max_rows or args.max_cols:
            if args.max:
                pieces.append(f"--max {args.max}")
            if args.max_rows:
                pieces.append(f"--max-rows {args.max_rows}")
            if args.max_cols:
                pieces.append(f"--max-cols {args.max_cols}")
        summary["next_command"] = ' '.join(pieces)

    # Choose a path for the JSON (sandbox or local dir)
    json_path = "/mnt/data/backwards_life_summary.json"
    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            import json
            json.dump(summary, jf, indent=2)
        print(f"[json] Wrote summary to {json_path}")
    except Exception:
        json_path = "backwards_life_summary.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            import json
            json.dump(summary, jf, indent=2)
        print(f"[json] Wrote summary to {json_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] Exiting on Ctrl-C.")
