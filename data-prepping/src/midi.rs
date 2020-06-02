use std::path::Path;

use bitvec::order::Msb0;
use bitvec::slice::{AsBits, BitSlice};
use ghakuf::messages::{MetaEvent, MidiEvent, SysExEvent};
use ghakuf::reader::{Handler, HandlerStatus, Reader};
use log::{error, trace};
use regex::bytes::Regex;

pub fn process_midi_file(
    path: impl AsRef<Path>,
    instrument_regex: &str,
    callback: impl FnMut(Result<(String, &[Notes]), AbortError>),
) {
    let mut handler = ExtractorHandler::new(
        Regex::new(&format!("(?-u)({})", instrument_regex)).unwrap(),
        callback,
    );

    let mut reader = Reader::new(&mut handler, path.as_ref()).unwrap();

    if let Err(err) = reader.read() {
        error!("Midi parse error ({:?}): {:?}", path.as_ref(), err);
    }

    handler.change_track();

    if let Some(err) = handler.abort {
        error!("Midi processing error ({:?}): {:?}", path.as_ref(), err);
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Notes(pub [u8; 16]);

struct ExtractorHandler<Cb> {
    instrument_track_matcher: Regex,
    ticks_per_quarter_note: u32,
    current_time_ticks: u32,
    current_name_match: Option<String>,
    current_notes: Notes,
    current_track_notes: Vec<Notes>,
    current_channel: Option<u8>,
    callback: Cb,
    abort: Option<AbortError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortError {
    InvalidTimeBase,
    TooManyMidiChannels,
    NameMismatch,
    MultipleDifferentNames,
    EmptyTrack,
}

impl<Cb> Handler for ExtractorHandler<Cb>
where
    Cb: FnMut(Result<(String, &[Notes]), AbortError>),
{
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
        trace!("meta_event: {:?}", (delta_time, event, data));
        if !self.advance_time(delta_time) {
            return;
        }

        match event {
            MetaEvent::SequenceOrTrackName | MetaEvent::InstrumentName => {
                if let Some(m) = self.instrument_track_matcher.find(&data[..]) {
                    let mut m = String::from_utf8_lossy(m.as_bytes()).into_owned();
                    m.make_ascii_lowercase();
                    if let Some(cm) = &self.current_name_match {
                        if cm != &m {
                            self.abort = Some(AbortError::MultipleDifferentNames);
                            return;
                        }
                    } else {
                        self.current_name_match = Some(m);
                    }
                }
            }
            _ => (),
        }
    }
    fn midi_event(&mut self, delta_time: u32, event: &MidiEvent) {
        trace!("midi_event: {:?}", (delta_time, event));
        if !self.advance_time(delta_time) {
            return;
        }
        match event {
            MidiEvent::NoteOff { ch, note, .. } => {
                if !self.check_midi_channel(*ch) {
                    return;
                }
                self.current_notes.set_off(*note);
            }
            MidiEvent::NoteOn { ch, note, velocity } => {
                if !self.check_midi_channel(*ch) {
                    return;
                }
                if *velocity > 3 {
                    self.current_notes.set_on(*note);
                } else {
                    self.current_notes.set_off(*note);
                }
            }
            _ => (),
        }
    }
    fn sys_ex_event(&mut self, delta_time: u32, event: &SysExEvent, data: &Vec<u8>) {
        trace!("sys_ex_event: {:?}", (delta_time, event, data));
        if !self.advance_time(delta_time) {
            return;
        }
    }
    fn track_change(&mut self) {
        self.change_track();
    }
    fn status(&mut self) -> HandlerStatus {
        match self.abort {
            None => HandlerStatus::Continue,
            Some(AbortError::InvalidTimeBase) => HandlerStatus::SkipAll,
            Some(AbortError::TooManyMidiChannels)
            | Some(AbortError::NameMismatch)
            | Some(AbortError::EmptyTrack)
            | Some(AbortError::MultipleDifferentNames) => HandlerStatus::SkipTrack,
        }
    }
}

impl<Cb> ExtractorHandler<Cb>
where
    Cb: FnMut(Result<(String, &[Notes]), AbortError>),
{
    fn new(instrument_track_matcher: Regex, callback: Cb) -> ExtractorHandler<Cb> {
        ExtractorHandler {
            instrument_track_matcher,
            callback,
            ticks_per_quarter_note: 0,
            current_time_ticks: 0,
            current_name_match: None,
            current_track_notes: Vec::new(),
            current_notes: Notes::empty(),
            current_channel: None,
            abort: None,
        }
    }

    fn change_track(&mut self) {
        (self.callback)(match (self.abort.take(), self.current_name_match.take()) {
            (None, Some(name)) => Ok((name, &self.current_track_notes)),
            (None, None) => Err(AbortError::EmptyTrack),
            (Some(AbortError::InvalidTimeBase), _) => return,
            (Some(err), _) => Err(err),
        });
        self.current_time_ticks = 0;
        self.current_channel = None;
        self.current_time_ticks = 0;
        self.current_notes.clear();
        self.current_track_notes.clear();
    }

    /// Advances time and returns true if successful.
    #[must_use]
    fn advance_time(&mut self, delta_time: u32) -> bool {
        if self.abort.is_some() {
            return false;
        }

        self.current_time_ticks += delta_time;

        if self.current_time_ticks > 0 && self.current_name_match.is_none() {
            self.abort = Some(AbortError::NameMismatch);
            return false;
        }

        // Multiply by two to get eights of a note.
        let note_index = self.current_time_ticks * 2 / self.ticks_per_quarter_note;

        while note_index > self.current_track_notes.len() as u32 {
            self.current_track_notes.push(self.current_notes.clone());
        }

        true
    }

    /// This checks that there only is one midi channel in use. If there are
    /// multiple it aborts and returns false.
    #[must_use]
    fn check_midi_channel(&mut self, channel: u8) -> bool {
        if let Some(cc) = self.current_channel {
            if cc == channel {
                true
            } else {
                self.abort = Some(AbortError::TooManyMidiChannels);
                false
            }
        } else {
            self.current_channel = Some(channel);
            true
        }
    }
}

impl Notes {
    pub fn empty() -> Notes {
        Notes([0x00; 16])
    }

    pub fn bits(&self) -> &BitSlice<Msb0, u8> {
        self.0.bits::<Msb0>()
    }
    pub fn bits_mut(&mut self) -> &mut BitSlice<Msb0, u8> {
        self.0.bits_mut::<Msb0>()
    }

    pub fn set_on(&mut self, note: u8) {
        self.bits_mut().set(note as usize, true);
    }

    pub fn set_off(&mut self, note: u8) {
        self.bits_mut().set(note as usize, false);
    }

    pub fn clear(&mut self) {
        self.bits_mut().set_all(false);
    }
}

impl std::fmt::Debug for Notes {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str("Notes(")?;
        for bit in self.bits() {
            fmt.write_str(if *bit { "1" } else { "0" })?;
        }
        fmt.write_str(")")?;
        Ok(())
    }
}