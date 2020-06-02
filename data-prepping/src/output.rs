use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Result as IOResult, Write};
use std::path::Path;
use std::str::FromStr;

use crate::midi::Notes;

pub struct Output {
    writer: BufWriter<File>,
    format: OutputFormat,
}

impl Output {
    pub fn new(path: impl AsRef<Path>, format: OutputFormat) -> IOResult<Output> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Output {
            format,
            writer: BufWriter::new(file),
        })
    }

    pub fn write(&mut self, name: &str, notes: &[Notes]) -> IOResult<()> {
        writeln!(self.writer, "# {}", name)?;

        match self.format {
            OutputFormat::BitsHex => {
                for note in notes {
                    writeln!(self.writer, "{}", BitsHexNotes(note))?;
                }
            }
            OutputFormat::Chars => {
                for note in notes {
                    writeln!(self.writer, "{}", CharsNotes(note))?;
                }
            }
        }

        self.writer.flush()?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    BitsHex,
    Chars,
}

impl FromStr for OutputFormat {
    type Err = OutputFormatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bits-hex" => Ok(OutputFormat::BitsHex),
            "chars" => Ok(OutputFormat::Chars),
            _ => Err(OutputFormatError(s.to_owned())),
        }
    }
}

#[derive(Debug)]
pub struct OutputFormatError(String);

impl std::fmt::Display for OutputFormatError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            fmt,
            "Invalid option '{}'. \
             Valid options are 'bits-hex' or 'chars'.",
            self.0
        )
    }
}

struct BitsHexNotes<'a>(&'a Notes);

impl<'a> std::fmt::Display for BitsHexNotes<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 .0 {
            fmt.write_fmt(format_args!("{:02X}", byte))?;
        }
        Ok(())
    }
}

struct CharsNotes<'a>(&'a Notes);

impl<'a> std::fmt::Display for CharsNotes<'a> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        static CHARS: [char; 84] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '_', '/', '\\', '=',
            '?', '*', '~', '!', '\'', '"', '$', '%', '&', '.', ':', ',', ';', '<', '|', '>',
        ];

        for (i, pressed) in self.0.bits().iter().enumerate().take(CHARS.len()) {
            if *pressed {
                fmt.write_fmt(format_args!("{}", CHARS[i]))?;
            }
        }
        Ok(())
    }
}
