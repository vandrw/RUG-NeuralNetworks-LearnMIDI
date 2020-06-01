use std::fs::{File, OpenOptions};
use std::io::Write;

use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use std::path::PathBuf;
use structopt::StructOpt;

pub mod midi;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "data-prepping",
    about = "A tool to convert MIDI files to use as training data."
)]
struct Opt {
    /// Path to an ignore file. Like gitignore.
    #[structopt(long)]
    ignore: Option<PathBuf>,

    /// Path to an ignore file. Appends path to all files that have been
    /// processed. Can be the same as the --ignore file.
    #[structopt(long)]
    processed: Option<PathBuf>,

    /// Max depth while recursing directories. Default is no maximum.
    #[structopt(long)]
    max_depth: Option<usize>,

    /// Number of files to process.
    #[structopt(short = "n", long)]
    count: Option<usize>,

    /// The folder or file to search through.
    input: PathBuf,

    /// The folder or file to put processed data.
    output: PathBuf,
}

fn main() {
    pretty_env_logger::init();

    let opt = Opt::from_args();

    let mut types_builder = TypesBuilder::new();
    types_builder.add("midi", "*.{mid,smf}").unwrap();
    types_builder.select("midi");

    let mut walk_builder = WalkBuilder::new(opt.input.canonicalize().unwrap());
    walk_builder.types(types_builder.build().unwrap());
    walk_builder.max_depth(opt.max_depth);
    if let Some(ignore) = opt.ignore {
        walk_builder.add_ignore(ignore);
    }

    let mut processed_file = opt.processed.map(|processed| {
        let processed_path: PathBuf = processed.canonicalize().unwrap();
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(&processed_path)
            .unwrap()
    });

    let walk = walk_builder
        .build()
        .filter_map(|r| match r {
            Ok(entry) if !entry.path().is_dir() => Some(entry.path().to_owned()),
            _ => None,
        })
        .take(opt.count.unwrap_or(usize::MAX));

    for path in walk {
        midi::test_read_midi(&path);
        if let Some(processed_file) = &mut processed_file {
            writeln!(processed_file, "{}", path.to_str().unwrap()).unwrap();
        }
    }
}
