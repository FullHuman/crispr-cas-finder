use std::fs;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hmmer_core::alphabet::{Alphabet, AlphabetKind};
use hmmer_core::hmm::{EvParams, Hmm};
use hmmer_io::{FastaReader, HmmFile, read_fasta_digital_sequences, read_hmm};

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
1 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 10 - - - -\n\
2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732\n\
0.693147 1.609438 1.609438 0.356675 1.203973 0.510826 0.916291\n\
2 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 2.995732 20 - - - -\n\
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

#[test]
fn hmmer3f_ascii_round_trip_preserves_model() {
    let (alphabet, expected) = example_hmm("round-trip");
    let mut encoded = Vec::new();
    HmmFile::write_ascii(&mut encoded, &expected, &alphabet).expect("write HMM");

    let (parsed_alphabet, parsed) = read_hmm(&mut Cursor::new(encoded)).expect("read written HMM");
    assert_eq!(parsed_alphabet.kind, AlphabetKind::Dna);
    assert_eq!(parsed.name, expected.name);
    assert_eq!(parsed.accession, expected.accession);
    assert_eq!(parsed.description, expected.description);
    assert_eq!(parsed.command_log, expected.command_log);
    assert_eq!(parsed.num_sequences, expected.num_sequences);
    assert_eq!(
        parsed.effective_num_seq_float,
        expected.effective_num_seq_float
    );
    assert_eq!(parsed.max_length, expected.max_length);
    assert_eq!(parsed.checksum, expected.checksum);
    assert_eq!(parsed.cutoffs, expected.cutoffs);
    assert_eq!(parsed.ev_params, expected.ev_params);
    assert_eq!(parsed.map, expected.map);
    assert_eq!(parsed.consensus, expected.consensus);
    assert_eq!(parsed.reference_annotation, expected.reference_annotation);
    assert_eq!(parsed.model_mask, expected.model_mask);
    assert_eq!(parsed.consensus_structure, expected.consensus_structure);
    assert_probabilities_close(
        parsed.model_composition.as_deref().expect("composition"),
        expected.model_composition.as_deref().expect("composition"),
    );
    for node in 0..=expected.num_nodes {
        assert_probabilities_close(
            parsed.insert_emissions(node),
            expected.insert_emissions(node),
        );
        assert_probabilities_close(parsed.transitions(node), expected.transitions(node));
        if node > 0 {
            assert_probabilities_close(
                parsed.match_emissions(node),
                expected.match_emissions(node),
            );
        }
    }
}

#[test]
fn reads_multiple_models_from_one_file() {
    let (alphabet, first) = example_hmm("first");
    let (_, second) = example_hmm("second");
    let mut encoded = Vec::new();
    HmmFile::write_ascii(&mut encoded, &first, &alphabet).expect("write first HMM");
    HmmFile::write_ascii(&mut encoded, &second, &alphabet).expect("write second HMM");
    let fixture = TempFile::create("hmm", &encoded);

    let mut reader = HmmFile::open(fixture.as_str(), None).expect("open models");
    assert_eq!(reader.read().expect("first model").1.name, "first");
    assert_eq!(reader.read().expect("second model").1.name, "second");
    assert!(matches!(
        reader.read(),
        Err(hmmer_core::errors::HmmerError::Eof)
    ));
}

#[test]
fn reads_all_bundled_hmmer3_profiles() {
    let profile_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../crispr-cas-finder-cli/data/CasFinder-2.0.3/CASprofiles-2.0.3");
    let mut profile_count = 0;
    for entry in fs::read_dir(profile_dir).expect("read bundled profile directory") {
        let path = entry.expect("profile directory entry").path();
        if path.extension().and_then(|value| value.to_str()) != Some("hmm") {
            continue;
        }
        let file = fs::File::open(&path).expect("open bundled HMM");
        let (_, hmm) = read_hmm(&mut BufReader::new(file))
            .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
        assert!(!hmm.name.is_empty(), "missing name in {}", path.display());
        assert!(hmm.num_nodes > 0, "empty model in {}", path.display());
        profile_count += 1;
    }
    assert_eq!(profile_count, 121);
}

#[test]
fn malformed_header_value_is_reported() {
    let fixture = b"HMMER3/f\nNAME  broken\nLENG  not-a-number\n";
    let error = read_hmm(&mut Cursor::new(fixture)).expect_err("invalid LENG must fail");
    assert!(error.to_string().contains("LENG"));
    assert!(error.to_string().contains("not-a-number"));
}

#[test]
fn truncated_model_body_is_reported() {
    let (alphabet, hmm) = example_hmm("truncated");
    let mut encoded = Vec::new();
    HmmFile::write_ascii(&mut encoded, &hmm, &alphabet).expect("write HMM");
    encoded.truncate(encoded.len() - 3);

    let error = read_hmm(&mut Cursor::new(encoded)).expect_err("truncation must fail");
    assert!(error.to_string().contains("premature EOF"));
}

fn example_hmm(name: &str) -> (Alphabet, Hmm) {
    let alphabet = Alphabet::dna();
    let mut hmm = Hmm::new(2, &alphabet);
    hmm.name = name.to_string();
    hmm.accession = Some("TEST0002".to_string());
    hmm.description = Some("round-trip model".to_string());
    hmm.command_log = Some("hmmbuild input.sto\nhmmpress model.hmm".to_string());
    hmm.num_sequences = Some(8);
    hmm.effective_num_seq_float = Some(6.25);
    hmm.effective_num_seq = 6.25;
    hmm.max_length = Some(91);
    hmm.map = Some(vec![0, 11, 22]);
    hmm.checksum = Some(42);
    hmm.ev_params = Some(EvParams {
        msv_mu: -9.0,
        msv_lambda: 0.7,
        viterbi_mu: -8.0,
        viterbi_lambda: 0.6,
        forward_tau: -3.0,
        forward_lambda: 0.5,
    });
    hmm.cutoffs.gathering = Some((20.0, 19.0));
    hmm.cutoffs.trusted = Some((25.0, 24.0));
    hmm.cutoffs.noise = Some((15.0, 14.0));
    hmm.model_composition = Some(vec![0.25; 4]);
    hmm.consensus = Some(vec![b' ', b'A', b'C', b' ']);
    hmm.reference_annotation = Some(vec![b' ', b'x', b'y', b' ']);
    hmm.model_mask = Some(vec![b' ', b'm', b'.', b' ']);
    hmm.consensus_structure = Some(vec![b' ', b'<', b'>', b' ']);

    for node in 0..=hmm.num_nodes {
        hmm.insert_emissions_mut(node).fill(0.25);
        hmm.transitions_mut(node).fill(1.0 / 7.0);
        if node > 0 {
            hmm.match_emissions_mut(node).fill(0.25);
        }
    }
    (alphabet, hmm)
}

fn assert_probabilities_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }
}
