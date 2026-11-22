use std::fs;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hmmer_core::alphabet::AlphabetKind;
use hmmer_core::hmm::{Cutoffs, EvParams};
use hmmer_io::{HmmFile, read_hmm};

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

/// A calibrated protein model of two nodes, every probability uniform.
const TINY_AMINO: &[u8] = include_bytes!("../../crispr-cas-finder-core/tests/data/tiny_amino.hmm");

/// A DNA model of two nodes that carries every annotation of the HMMER3/f
/// format: accession, description, command log, cutoffs, and the map,
/// consensus, reference, mask and structure columns.
const ANNOTATED_DNA: &[u8] =
    include_bytes!("../../crispr-cas-finder-core/tests/data/annotated_dna.hmm");

#[test]
fn parses_hmmer3f_metadata_and_model_rows() {
    let mut reader = BufReader::new(Cursor::new(TINY_AMINO));
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
fn parses_every_annotation_of_an_hmmer3f_model() {
    let (alphabet, hmm) = read_hmm(&mut Cursor::new(ANNOTATED_DNA)).expect("parse HMM");

    assert_eq!(alphabet.kind, AlphabetKind::Dna);
    assert_eq!(hmm.name, "round-trip");
    assert_eq!(hmm.accession.as_deref(), Some("TEST0002"));
    assert_eq!(hmm.description.as_deref(), Some("round-trip model"));
    assert_eq!(
        hmm.command_log.as_deref(),
        Some("hmmbuild input.sto\nhmmpress model.hmm")
    );
    assert_eq!(hmm.num_nodes, 2);
    assert_eq!(hmm.num_sequences, Some(8));
    assert_eq!(hmm.effective_num_seq_float, Some(6.25));
    assert_eq!(hmm.max_length, Some(91));
    assert_eq!(hmm.checksum, Some(42));
    assert_eq!(
        hmm.cutoffs,
        Cutoffs {
            gathering: Some((20.0, 19.0)),
            trusted: Some((25.0, 24.0)),
            noise: Some((15.0, 14.0)),
        }
    );
    assert_eq!(
        hmm.ev_params,
        Some(EvParams {
            msv_mu: -9.0,
            msv_lambda: 0.7,
            viterbi_mu: -8.0,
            viterbi_lambda: 0.6,
            forward_tau: -3.0,
            forward_lambda: 0.5,
        })
    );
    assert_eq!(hmm.map.as_deref(), Some(&[0, 11, 22][..]));
    assert_eq!(hmm.consensus.as_deref(), Some(&b" AC "[..]));
    assert_eq!(hmm.reference_annotation.as_deref(), Some(&b" xy "[..]));
    assert_eq!(hmm.model_mask.as_deref(), Some(&b" m. "[..]));
    assert_eq!(hmm.consensus_structure.as_deref(), Some(&b" <> "[..]));
    assert_probabilities_close(
        hmm.model_composition.as_deref().expect("composition"),
        &[0.25; 4],
    );
    for node in 0..=hmm.num_nodes {
        assert_probabilities_close(hmm.insert_emissions(node), &[0.25; 4]);
        assert_probabilities_close(hmm.transitions(node), &[1.0 / 7.0; 7]);
        if node > 0 {
            assert_probabilities_close(hmm.match_emissions(node), &[0.25; 4]);
        }
    }
}

#[test]
fn reads_multiple_models_from_one_file() {
    let text = std::str::from_utf8(ANNOTATED_DNA).expect("fixture is UTF-8");
    let models = [
        text.replace("NAME  round-trip", "NAME  first"),
        text.replace("NAME  round-trip", "NAME  second"),
    ]
    .concat();
    let fixture = TempFile::create("hmm", models.as_bytes());

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
    // Cut the model before the `//` that ends it. Counting bytes from the end
    // would depend on the line endings of the checkout.
    let terminator = ANNOTATED_DNA
        .windows(2)
        .rposition(|pair| pair == b"//")
        .expect("the fixture ends with //");
    let truncated = &ANNOTATED_DNA[..terminator];

    let error = read_hmm(&mut Cursor::new(truncated)).expect_err("truncation must fail");
    assert!(error.to_string().contains("premature EOF"));
}

fn assert_probabilities_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }
}
