"""Basic tests for the hmmer Python bindings."""
import hmmer


def test_alphabet_amino():
    abc = hmmer.Alphabet.amino()
    assert abc.k == 20
    assert "Amino" in repr(abc)


def test_alphabet_dna():
    abc = hmmer.Alphabet.dna()
    assert abc.k == 4
    assert "DNA" in repr(abc)


def test_hmmsearch_globins():
    results = hmmer.hmmsearch("bench_data/globins4.hmm", "bench_data/seqdb_100.fa")
    assert results.nseq == 100
    assert results.nhits == 100
    assert len(results) == 100

    # Top hit should be globins4-sample67
    top = results[0]
    assert top.name == "globins4-sample67"
    assert top.score > 90.0
    assert top.evalue < 1e-25
    assert top.ndom >= 1


def test_hmmsearch_iteration():
    results = hmmer.hmmsearch("bench_data/globins4.hmm", "bench_data/seqdb_100.fa")
    names = [hit.name for hit in results]
    assert len(names) == 100
    assert "globins4-sample67" in names


def test_hmmsearch_no_hits():
    results = hmmer.hmmsearch("bench_data/fn3.hmm", "bench_data/seqdb_100.fa")
    assert results.nhits == 0


def test_hmmsearch_1000():
    results = hmmer.hmmsearch("bench_data/globins4.hmm", "bench_data/seqdb_1000.fa")
    assert results.nseq == 1000
    # With uncalibrated E-values, most sequences pass (no filter)
    # Top hit should still be a globin
    assert results.nhits > 90
    top = results[0]
    assert top.score > 100.0
