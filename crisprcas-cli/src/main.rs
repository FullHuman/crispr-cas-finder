use anyhow::Result;
use clap::Parser;
use crisprcas_cli::{Cli, run};

fn main() -> Result<()> {
    env_logger::init();
    run(Cli::parse())
}
