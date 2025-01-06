import os
import pathlib
import shutil
import tarfile
import tempfile
import urllib
from typing import List, Union

import pandas as pd


def download_diamond() -> str:
    cache_dir = os.path.join(str(pathlib.Path.home()), ".cache")
    if not os.path.exists(os.path.join(cache_dir, "diamond")):
        fname = os.path.join(cache_dir, "diamond.tar.gz")
        urllib.request.urlretrieve(
            "https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz", fname
        )  # nosec
        with tarfile.open(fname, "r:gz") as f:
            f.extractall(cache_dir)  # nosec
        os.remove(fname)
    return os.path.join(cache_dir, "diamond")


def seqs_to_fasta(seqs: List[str], fname: str):
    with open(fname, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">{i}\n{seq}\n\n")


def make_diamond_db(seqs_or_fasta: Union[str, List[str]], diamond_fname: str) -> None:
    fasta_fname = seqs_or_fasta
    os.system(f"{download_diamond()} makedb --in {fasta_fname} --db {diamond_fname} --ignore-warnings")  # nosec

def diamond_alignment(seqs_or_fasta: Union[str, List[str]], ref_db: str) -> pd.DataFrame:
    temp_dir = tempfile.mkdtemp()
    fasta = seqs_or_fasta

    diamond = download_diamond()
    output = os.path.join(temp_dir, "diamond.tsv")
    outcols = "qseqid scovhsp pident sseqid"
    os.system(f"{diamond} blastp -k 1 --query {fasta} --db {ref_db} --out {output} --outfmt 6 {outcols} --quiet")  # nosec
    
    if not os.path.exists(output):
        df = pd.DataFrame(columns=["sequence", "aln_coverage", "max_id", "ref_entry_id"])
    else:
        df = pd.read_csv(output, sep="\t", header=None, index_col=0, names=["aln_coverage", "max_id", "ref_entry_id"])
    
    df["max_id"] *= df["aln_coverage"] / 100
    return df
