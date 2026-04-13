use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const LOGSUM_SCALE: f64 = 1000.0;
const LOGSUM_TABLE_SIZE: usize = 16000;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("flogsum_table.rs");
    let mut f = File::create(dest_path).unwrap();

    writeln!(f, "#[allow(clippy::approx_constant)]").unwrap();
    writeln!(f, "static FLOGSUM_LOOKUP: [f32; {}] = [", LOGSUM_TABLE_SIZE).unwrap();
    for i in 0..LOGSUM_TABLE_SIZE {
        let val = (1.0_f64 + (-(i as f64) / LOGSUM_SCALE).exp()).ln() as f32;
        writeln!(f, "    {val:?},").unwrap();
    }
    writeln!(f, "];").unwrap();
}
