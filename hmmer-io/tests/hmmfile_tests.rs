use hmmer_io::hmmfile::{V3A_MAGIC, V3E_MAGIC, V3F_MAGIC};

#[test]
fn test_hmmfile_magic_numbers() {
    assert_eq!(V3F_MAGIC, 0xe8ededba);
    assert_eq!(V3A_MAGIC, 0xe8ededb5);
    assert_eq!(V3E_MAGIC, 0xe8ededb9);
}
