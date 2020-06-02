use std::fs::OpenOptions;
use std::io::Write;

use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use log::{warn, info, trace};
use std::path::PathBuf;
use structopt::StructOpt;

pub mod midi;

use midi::AbortError;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "data-prepping",
    about = "A tool to convert MIDI files to use as training data."
)]
struct Opt {
    /// Path to an ignore file. Like gitignore. Can be specified multiple times.
    #[structopt(short, long)]
    ignore: Vec<PathBuf>,

    /// Path to a file. All paths to the processed midi files will be appended
    /// to this file. This file can also be used as an ignore file (See: --pi
    #[structopt(short, long)]
    processed: Option<PathBuf>,

    /// This specifies both --processed and --ignore at the same time. Can not
    /// be used together with --processed.
    #[structopt(long = "pi", conflicts_with = "processed")]
    processed_ignore: Option<PathBuf>,

    /// Max depth while recursing directories. Default is no maximum.
    #[structopt(short = "d", long)]
    max_depth: Option<usize>,

    /// Number of files to process. Infinite if unspecified.
    #[structopt(short = "n", long)]
    count: Option<usize>,

    /// The folder or file to search through.
    input: PathBuf,

    /// The folder or file to put processed data.
    output: PathBuf,
}

fn main() {
    pretty_env_logger::init();

    let mut opt = Opt::from_args();

    if let Some(pi) = opt.processed_ignore {
        opt.processed = Some(pi.clone());
        opt.ignore.push(pi);
    }

    let mut types_builder = TypesBuilder::new();
    types_builder.add("midi", "*.{mid,smf}").unwrap();
    types_builder.select("midi");

    let mut walk_builder = WalkBuilder::new(opt.input.canonicalize().unwrap());
    walk_builder.types(types_builder.build().unwrap());
    walk_builder.max_depth(opt.max_depth);
    for ignore in opt.ignore {
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
        midi::test_read_midi(&path, |track| match track {
            Ok((name, track)) => {
                info!("track_name: {}", name);
                for n in track {
                    info!("track: {:?}", n);
                }
            }
            Err(AbortError::NameMismatch) => trace!("Midi track name did not match the regex"),
            Err(AbortError::EmptyTrack) => trace!("Midi track was basically empty"),
            Err(err) => warn!("Error while processing midi: {:?}", err),
        });
        if let Some(processed_file) = &mut processed_file {
            writeln!(processed_file, "{}", path.to_str().unwrap()).unwrap();
        }
    }
}
