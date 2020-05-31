use ghakuf::messages::{MetaEvent, MidiEvent, SysExEvent};
use ghakuf::reader::{Handler, Reader};
use log::debug;
use std::path::Path;

pub fn test_read_midi(path: impl AsRef<Path>) {
    let mut handler = LogHandler {};
    let mut reader = Reader::new(&mut handler, path.as_ref()).unwrap();
    reader.read().unwrap();
}

struct LogHandler;

impl Handler for LogHandler {
    fn header(&mut self, format: u16, track: u16, time_base: u16) {
        debug!("header: {}, {}, {}", format, track, time_base);
    }
    fn meta_event(&mut self, delta_time: u32, event: &MetaEvent, data: &Vec<u8>) {
        debug!("meta_event: {:?}", (delta_time, event, data));
        match std::str::from_utf8(data) {
            Ok(s) => debug!("meta_event_text: {}", s),
            _ => (),
        }
    }
    fn midi_event(&mut self, delta_time: u32, event: &MidiEvent) {
        debug!("midi_event: {:?}", (delta_time, event));
    }
    fn sys_ex_event(&mut self, delta_time: u32, event: &SysExEvent, data: &Vec<u8>) {
        debug!("sys_ex_event: {:?}", (delta_time, event, data));
    }
    fn track_change(&mut self) {
        debug!("track_change!");
    }
    fn status(&mut self) -> ghakuf::reader::HandlerStatus {
        ghakuf::reader::HandlerStatus::Continue
    }
}
