use std::{mem::replace, path::Path};

use ghakuf::messages::{MetaEvent, MidiEvent, SysExEvent};
use ghakuf::reader::{Handler, HandlerStatus, Reader};
use log::{debug, info};

use regex::bytes::Regex;

pub fn test_read_midi(path: impl AsRef<Path>) {
    let mut handler =
        ExtractorHandler::new(Regex::new("(?-u)(piano|guitar|string|harmon)").unwrap());
    let mut reader = Reader::new(&mut handler, path.as_ref()).unwrap();
    if reader.read().is_err() {
        //println!("Error_path: {:?}", path.as_ref());
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Notes([u8; 16]);

struct ExtractorHandler {
    instrument_track_matcher: Regex,
    ticks_per_quarter_note: u32,
    current_time_ticks: u32,
    current_is_match: bool,
    current_notes: Notes,
    current_track_notes: Vec<Notes>,
    abort: Option<AbortError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AbortError {
    InvalidTimeBase,
}

impl Handler for ExtractorHandler {
    fn header(&mut self, _smf_format: u16, _track_count: u16, time_base: u16) {
        if self.ticks_per_quarter_note != 0 {
            panic!("header called twice!");
        }

        if time_base & 0x8000 == 0x0 && time_base != 0 {
            self.ticks_per_quarter_note = time_base as u32;
        } else {
            self.abort = Some(AbortError::InvalidTimeBase);
        };
    }
    fn meta_event(&mut self, delta_time: u32, event: &MetaEvent, data: &Vec<u8>) {
        debug!("meta_event: {:?}", (delta_time, event, data));
        match event {
            MetaEvent::SequenceOrTrackName | MetaEvent::InstrumentName => {}
            MetaEvent::SequenceNumber => (),
            MetaEvent::TextEvent => (),
            MetaEvent::CopyrightNotice => (),
            MetaEvent::Lyric => (),
            MetaEvent::Marker => (),
            MetaEvent::CuePoint => (),
            MetaEvent::MIDIChannelPrefix => (),
            MetaEvent::EndOfTrack => (),
            MetaEvent::SetTempo => (),
            MetaEvent::SMTPEOffset => (),
            MetaEvent::TimeSignature => (),
            MetaEvent::KeySignature => (),
            MetaEvent::SequencerSpecificMetaEvent => (),
            MetaEvent::Unknown { .. } => (),
        }
        match std::str::from_utf8(data) {
            Ok(s) => debug!("meta_event_text: {}", s),
            _ => (),
        }
        self.advance_time(delta_time);
    }
    fn midi_event(&mut self, delta_time: u32, event: &MidiEvent) {
        debug!("midi_event: {:?}", (delta_time, event));
        self.advance_time(delta_time);
        match event {
            MidiEvent::NoteOff { ch, note, velocity } => (),
            MidiEvent::NoteOn { ch, note, velocity } => (),
            MidiEvent::PolyphonicKeyPressure { ch, note, velocity } => (),
            MidiEvent::ControlChange { ch, control, data } => (),
            MidiEvent::ProgramChange { ch, program } => (),
            MidiEvent::ChannelPressure { ch, pressure } => (),
            MidiEvent::PitchBendChange { ch, data } => (),
            MidiEvent::Unknown { ch } => (),
        }
    }
    fn sys_ex_event(&mut self, delta_time: u32, event: &SysExEvent, data: &Vec<u8>) {
        debug!("sys_ex_event: {:?}", (delta_time, event, data));
        self.advance_time(delta_time);
    }
    fn track_change(&mut self) {
        if self.abort.is_none() {
            info!("track: {:?}", self.current_notes);
            // TODO: Save track
        }
        self.current_time_ticks = 0;
    }
    fn status(&mut self) -> HandlerStatus {
        match self.abort {
            None => HandlerStatus::Continue,
            Some(AbortError::InvalidTimeBase) => HandlerStatus::SkipAll,
        }
    }
}

impl ExtractorHandler {
    fn new(instrument_track_matcher: Regex) -> ExtractorHandler {
        ExtractorHandler {
            instrument_track_matcher,
            ticks_per_quarter_note: 0,
            current_time_ticks: 0,
            current_is_match: false,
            current_track_notes: Vec::new(),
            current_notes: Notes::empty(),
            abort: None,
        }
    }

    /// Advances time and returns true if successful.
    fn advance_time(&mut self, delta_time: u32) -> bool {
        if self.abort.is_some() {
            return false;
        }

        self.current_time_ticks += delta_time;

        // Multiply by two to get eights of a note.
        let note_index = self.current_time_ticks * 2 / self.ticks_per_quarter_note;

        while note_index > self.current_track_notes.len() as u32 {
            self.current_track_notes
                .push(replace(&mut self.current_notes, Notes::empty()));
        }

        true
    }
}

impl Notes {
    pub fn empty() -> Notes {
        Notes([0x00; 16])
    }
}
