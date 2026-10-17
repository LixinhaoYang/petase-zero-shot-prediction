from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO
import csv
import time

FASTA = "blast_query.fasta"
OUTCSV = "blast_hits.csv"

MAX_HITS = 5
SLEEP_SEC = 15   # Space out remote requests to respect NCBI service limits.

rows = []

for record in SeqIO.parse(FASTA, "fasta"):
    qid = record.id
    seq = str(record.seq)
    print(f"BLASTing {qid} len={len(seq)}")

    try:
        handle = NCBIWWW.qblast(
            program="blastp",
            database="nr",
            sequence=seq,
            hitlist_size=MAX_HITS,
            expect=1e-5
        )
    except Exception as e:
        print("BLAST failed:", e)
        continue

    blast_record = NCBIXML.read(handle)

    if not blast_record.alignments:
        rows.append([qid, len(seq), None, None, None, None])
    else:
        for aln in blast_record.alignments[:MAX_HITS]:
            hsp = aln.hsps[0]
            identity = hsp.identities / hsp.align_length * 100
            coverage = hsp.align_length / len(seq) * 100
            rows.append([
                qid,
                len(seq),
                aln.accession,
                aln.hit_def[:120],
                round(identity, 2),
                round(coverage, 2),
            ])

    time.sleep(SLEEP_SEC)

with open(OUTCSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "query_id",
        "query_len",
        "hit_accession",
        "hit_description",
        "identity_pct",
        "query_coverage_pct",
    ])
    w.writerows(rows)

print("Wrote:", OUTCSV)

