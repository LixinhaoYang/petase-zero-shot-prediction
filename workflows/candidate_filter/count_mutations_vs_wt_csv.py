import argparse
import csv
from pathlib import Path

COMMON_SEQ_COLS = [
    "sequence", "seq", "aa_sequence", "protein_sequence", "protein_seq",
    "aa_seq", "translated", "translation", "mutant_sequence",
    "wt_sequence", "WT", "prot", "protein"
]

def read_csv_rows(path: Path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            raise RuntimeError(f"{path} has no rows")
        return r.fieldnames, rows

def pick_col(fieldnames, preferred=None):
    if preferred and preferred in fieldnames:
        return preferred
    lower_map = {c.lower(): c for c in fieldnames}
    for c in COMMON_SEQ_COLS:
        if c in lower_map:
            return lower_map[c]
    # fallback: longest average string-like column
    best = None
    best_score = -1
    for c in fieldnames:
        lc = c.lower()
        if any(x in lc for x in ["id", "name", "score", "rank", "label", "class", "pred", "ddg", "plddt"]):
            continue
        # heuristic: try a few rows, count alphabetic chars
        score = 0
        for row in sample_rows:
            v = (row.get(c) or "").strip()
            if not v:
                continue
            score += sum(ch.isalpha() for ch in v)
        if score > best_score:
            best_score = score
            best = c
    return best

def sanitize_seq(s: str) -> str:
    s = (s or "").strip().upper()
    # remove whitespace and common separators
    s = "".join(ch for ch in s if ch.isalpha())
    return s

def nw_align(a: str, b: str, match=2, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    S = [[0]*(m+1) for _ in range(n+1)]
    T = [[0]*(m+1) for _ in range(n+1)]  # 0 diag, 1 up, 2 left

    for i in range(1, n+1):
        S[i][0] = S[i-1][0] + gap
        T[i][0] = 1
    for j in range(1, m+1):
        S[0][j] = S[0][j-1] + gap
        T[0][j] = 2

    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            bj = b[j-1]
            diag = S[i-1][j-1] + (match if ai == bj else mismatch)
            up   = S[i-1][j] + gap
            left = S[i][j-1] + gap
            best = diag
            tb = 0
            if up > best:
                best = up
                tb = 1
            if left > best:
                best = left
                tb = 2
            S[i][j] = best
            T[i][j] = tb

    i, j = n, m
    a_aln, b_aln = [], []
    while i > 0 or j > 0:
        tb = T[i][j] if (i > 0 and j > 0) else (1 if j == 0 else 2)
        if tb == 0:
            a_aln.append(a[i-1]); b_aln.append(b[j-1]); i -= 1; j -= 1
        elif tb == 1:
            a_aln.append(a[i-1]); b_aln.append("-"); i -= 1
        else:
            a_aln.append("-"); b_aln.append(b[j-1]); j -= 1

    return "".join(reversed(a_aln)), "".join(reversed(b_aln))

def summarize_edits(wt_aln: str, q_aln: str):
    subs = ins = dels = 0
    for w, q in zip(wt_aln, q_aln):
        if w == "-" and q != "-":
            ins += 1
        elif w != "-" and q == "-":
            dels += 1
        elif w != q:
            subs += 1
    return subs, ins, dels, subs + ins + dels

def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = int(round((p/100.0)*(len(sorted_vals)-1)))
    return sorted_vals[k]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_csv", required=True)
    ap.add_argument("--seq_csv", required=True)
    ap.add_argument("--wt_col", default=None, help="column name in wt_csv containing WT sequence")
    ap.add_argument("--seq_col", default=None, help="column name in seq_csv containing candidate sequences")
    ap.add_argument("--id_col", default=None, help="optional id column in seq_csv (else uses row index)")
    ap.add_argument("--out", default="mutation_counts.csv")
    ap.add_argument("--sort", default="total_edits", choices=["total_edits","subs","ins","dels","length"])
    ap.add_argument("--ascending", action="store_true", help="sort ascending (default descending)")
    args = ap.parse_args()

    wt_fields, wt_rows = read_csv_rows(Path(args.wt_csv))
    seq_fields, seq_rows = read_csv_rows(Path(args.seq_csv))

    # allow heuristic column picking (needs sample rows)
    sample_rows = (seq_rows[:20] if len(seq_rows) >= 20 else seq_rows)

    wt_col = args.wt_col or (wt_fields[0] if len(wt_fields)==1 else None)
    if wt_col is None:
        # pick from wt csv too
        sample_rows = (wt_rows[:20] if len(wt_rows) >= 20 else wt_rows)
        wt_col = pick_col(wt_fields)
    seq_col = args.seq_col or pick_col(seq_fields)

    if wt_col is None or seq_col is None:
        raise RuntimeError(f"Could not auto-detect columns. WT fields={wt_fields}, SEQ fields={seq_fields}")

    wt_seq_raw = wt_rows[0].get(wt_col, "")
    wt_seq = sanitize_seq(wt_seq_raw)
    if not wt_seq:
        raise RuntimeError(f"WT sequence empty after cleaning. wt_col={wt_col}, value preview={wt_seq_raw[:50]!r}")

    # id column
    id_col = args.id_col
    if id_col and id_col not in seq_fields:
        raise RuntimeError(f"--id_col {id_col} not in seq_csv columns")

    results = []
    for i, row in enumerate(seq_rows, start=1):
        q_raw = row.get(seq_col, "")
        q = sanitize_seq(q_raw)
        if not q:
            continue
        wt_aln, q_aln = nw_align(wt_seq, q)
        subs, ins, dels, total = summarize_edits(wt_aln, q_aln)
        rid = row.get(id_col) if id_col else row.get("name") or row.get("id") or f"row_{i:06d}"
        results.append({
            "id": rid,
            "length": len(q),
            "subs": subs,
            "ins": ins,
            "dels": dels,
            "total_edits": total,
        })

    if not results:
        raise RuntimeError(f"No sequences found in seq_csv using seq_col={seq_col}")

    rev = not args.ascending
    results.sort(key=lambda d: d[args.sort], reverse=rev)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","length","subs","ins","dels","total_edits"])
        w.writeheader()
        w.writerows(results)

    totals = sorted([r["total_edits"] for r in results])
    print(f"WT csv: {args.wt_csv}  wt_col={wt_col}  WT_len={len(wt_seq)}")
    print(f"SEQ csv: {args.seq_csv}  seq_col={seq_col}  N={len(results)}")
    print(f"Wrote: {args.out}")
    print(f"total_edits: min={totals[0]}  median={pct(totals,50)}  p90={pct(totals,90)}  max={totals[-1]}")
    print("Top 20:")
    for k, r in enumerate(results[:20], start=1):
        print(f"{k:2d}  {str(r['id'])[:30]:30s}  edits={r['total_edits']:3d}  subs={r['subs']:3d} ins={r['ins']:3d} del={r['dels']:3d} len={r['length']}")

