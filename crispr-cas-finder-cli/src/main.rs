use anyhow::Result;
use clap::Parser;
use crispr_cas_finder_cli::{Cli, run};

fn main() -> Result<()> {
    env_logger::init();
    run(Cli::parse())
}
