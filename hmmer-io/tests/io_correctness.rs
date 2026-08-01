use std::fs;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hmmer_core::alphabet::{Alphabet, AlphabetKind};
use hmmer_io::{FastaReader, read_fasta_digital_sequences, read_hmm};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);

impl TempFile {
    fn create(extension: &str, contents: &[u8]) -> Self {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hmmer-io-test-{}-{id}.{extension}",
            std::process::id()
        ));
        fs::write(&path, contents).expect("write test fixture");
        Self(path)
    }

    fn as_str(&self) -> &str {
        self.0.to_str().expect("temporary path is UTF-8")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        fs::remove_file(&self.0).ok();
    }
}

#[test]
fn bulk_and_streaming_fasta_readers_agree() {
    let fixture = TempFile::create(
        "fasta",
        b">alpha first protein\nACDEFG\nHIK\n>beta\nMNPQRSTVWY\n>empty description\nA-X*\n",
    );
    let alphabet = Alphabet::amino();
    let bulk = read_fasta_digital_sequences(&alphabet, fixture.as_str()).expect("bulk read");

    let mut reader =
        FastaReader::open_digital_fasta(&alphabet, fixture.as_str()).expect("stream open");
    let mut streamed = Vec::new();
    while let Some(sequence) = reader.read().expect("stream read") {
        streamed.push(sequence);
    }

    assert_eq!(bulk.len(), 3);
    assert_eq!(streamed.len(), bulk.len());
    for (left, right) in bulk.iter().zip(&streamed) {
        assert_eq!(left.name, right.name);
        assert_eq!(left.description, right.description);
        assert_eq!(left.residues, right.residues);
        assert_eq!(left.source_length, right.source_length);
    }
    assert_eq!(bulk[0].name, "alpha");
    assert_eq!(bulk[0].description.as_deref(), Some("first protein"));
    assert_eq!(bulk[0].len(), 9);
    assert_eq!(bulk[1].description, None);
    assert_eq!(bulk[2].residues[1], alphabet.digitize(b'-'));
    assert_eq!(bulk[2].residues[2], alphabet.digitize(b'X'));
    assert_eq!(bulk[2].residues[3], alphabet.digitize(b'*'));
}

#[test]
fn parses_hmmer3f_metadata_and_model_rows() {
    let fixture = b"HMMER3/f [test]\n\
NAME  tiny\n\
ACC   TEST0001\n\
DESC  two-node profile\n\
LENG  2\n\
MAXL  77\n\
ALPH  amino\n\
MAP   yes\n\
NSEQ  4\n\
EFFN  3.5\n\
CKSUM 1234\n\
STATS LOCAL MSV -9.0 0.7\n\
STATS LOCAL VITERBI -8.0 0.6\n\
STATS LOCAL FORWARD -3.0 0.5\n\
HMM          A C D E F G H I K L M N P Q R S T V W Y\n\
             m->m m->i m->d i->m i->i d->m d->d\n\
COMPO 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732\n\
2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732\n\
0.693147 1.609438 1.609438 0.356675 1.203973 0.510826 0.916291\n\
1 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 10\n\
2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732\n\
0.693147 1.609438 1.609438 0.356675 1.203973 0.510826 0.916291\n\
2 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 20\n\
2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732\n\
0.693147 1.609438 1.609438 0.356675 1.203973 0.510826 0.916291\n\
//\n";

    let mut reader = BufReader::new(Cursor::new(fixture));
    let (alphabet, hmm) = read_hmm(&mut reader).expect("parse HMM");

    assert_eq!(alphabet.kind, AlphabetKind::Amino);
    assert_eq!(hmm.name, "tiny");
    assert_eq!(hmm.accession.as_deref(), Some("TEST0001"));
    assert_eq!(hmm.description.as_deref(), Some("two-node profile"));
    assert_eq!(hmm.num_nodes, 2);
    assert_eq!(hmm.max_length, Some(77));
    assert_eq!(hmm.num_sequences, Some(4));
    assert_eq!(hmm.effective_num_seq_float, Some(3.5));
    assert_eq!(hmm.checksum, Some(1234));
    assert_eq!(hmm.map.as_ref().expect("map")[1..], [10, 20]);
    assert!(hmm.ev_params.is_some());
    let composition = hmm.model_composition.as_ref().expect("composition");
    assert!((composition.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!((composition[0] - 0.05).abs() < 1e-5);
    assert_eq!(hmm.match_emissions(1).len(), 20);
    assert!((hmm.match_emissions(1)[0] - 0.05).abs() < 1e-5);
    assert!((hmm.transitions(1)[0] - 0.5).abs() < 1e-5);
}
