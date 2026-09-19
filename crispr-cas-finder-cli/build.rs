use std::{env, fs, io::Write, path::Path};

fn main() {
    let root = Path::new("data/CasFinder-2.0.3");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut files = Vec::new();
    for directory in fs::read_dir(root).expect("bundled Cas model directories") {
        let directory = directory.unwrap().path();
        if directory.is_dir() {
            for entry in fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                if matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("xml" | "hmm")
                ) {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    assert!(
        !files.is_empty(),
        "bundled Cas data must be present at build time"
    );
    let out = Path::new(&env::var_os("OUT_DIR").unwrap()).join("cas_data.rs");
    let mut generated = fs::File::create(out).unwrap();
    writeln!(generated, "const FILES: &[(&str, &[u8])] = &[").unwrap();
    for path in files {
        let name = path
            .strip_prefix(root)
            .unwrap()
            .to_str()
            .unwrap()
            .replace('\\', "/");
        writeln!(
            generated,
            "({name:?}, include_bytes!({:?})),",
            fs::canonicalize(path).unwrap()
        )
        .unwrap();
    }
    writeln!(generated, "];").unwrap();
}
