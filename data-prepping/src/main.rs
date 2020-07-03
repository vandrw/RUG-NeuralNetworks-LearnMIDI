use std::fs::OpenOptions;
use std::io::Write;

use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use log::{trace, warn};
use std::path::PathBuf;
use structopt::StructOpt;

pub mod midi;
pub mod output;

use midi::AbortError;

use output::{Output, OutputFormat};

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

    /// Specify a regex filter which is matched against the midi track-name and
    /// midi instrument-name tags. If it matches this track will be processed.
    #[structopt(
        short = "f",
        long = "filter",
        default_value = "piano|guitar|string|harmon"
    )]
    name_filter: String,

    /// Specify an output format. Valid options are 'bits-hex' or 'chars'.
    #[structopt(short, long, default_value = "bits-hex")]
    output_format: OutputFormat,

    /// Specifies the maximum amount of consecutive output notes that can be
    /// all zero. This is useful for training RNNs to avoid vanishing gradients.
    #[structopt(short, long, default_value = "65535")]
    max_off_entries: usize,

    /// Specifies the maximum ratio of time steps without any note to all time
    /// steps. If the ratio is higher than the given the track will be ignored.
    #[structopt(long, default_value = "0.25")]
    max_off_ratio: f32,

    /// The folder or file to search through.
    input: PathBuf,

    /// The file to append processed data.
    output: PathBuf,
}

fn create_file_walk(opt: &Opt) -> impl Iterator<Item = PathBuf> {
    let mut types_builder = TypesBuilder::new();
    types_builder.add("midi", "*.{mid,smf}").unwrap();
    types_builder.select("midi");

    let mut walk_builder = WalkBuilder::new(opt.input.canonicalize().unwrap());
    walk_builder.types(types_builder.build().unwrap());
    walk_builder.max_depth(opt.max_depth);
    for ignore in &opt.ignore {
        walk_builder.add_ignore(ignore);
    }

    walk_builder
        .build()
        .filter_map(|r| match r {
            Ok(entry) if !entry.path().is_dir() => Some(entry.path().to_owned()),
            _ => None,
        })
        .take(opt.count.unwrap_or(usize::MAX))
}

fn main() {
    pretty_env_logger::init();

    let mut opt = Opt::from_args();

    if let Some(pi) = opt.processed_ignore.take() {
        opt.processed = Some(pi.clone());
        opt.ignore.push(pi);
    }

    let mut processed_file = opt.processed.as_ref().map(|processed| {
        let processed_path: PathBuf = processed.canonicalize().unwrap();
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(&processed_path)
            .unwrap()
    });

    let mut output = Output::new(&opt.output, opt.output_format, opt.max_off_entries).unwrap();

    for path in create_file_walk(&opt) {
        midi::process_midi_file(&path, &opt.name_filter, |track| match track {
            Ok((name, track)) => {
                let off_count = track.iter().filter(|n| n.is_empty()).count();
                let off_ratio = off_count as f32 / track.len() as f32;
                if off_ratio < opt.max_off_ratio {
                    output.write(&name, track).unwrap()
                }
            }
            Err(AbortError::NameMismatch) => trace!("Midi track name did not match the filter"),
            Err(AbortError::EmptyTrack) => trace!("Midi track was basically empty"),
            Err(err) => warn!("Error while processing midi: {:?}", err),
        });

        if let Some(processed_file) = processed_file.as_mut() {
            writeln!(processed_file, "{}", path.to_str().unwrap()).unwrap();
        }
    }
}
