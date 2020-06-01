use std::collections::HashSet;
use std::path::Path;

use ghakuf::messages::{MetaEvent, MidiEvent, SysExEvent};
use ghakuf::reader::{Handler, HandlerStatus, Reader};
use log::{debug, info};

pub fn test_read_midi(path: impl AsRef<Path>, names: &mut HashSet<String>) {
    let mut handler = MyHandler { names };
    let mut reader = Reader::new(&mut handler, path.as_ref()).unwrap();
    if reader.read().is_err() {
        //println!("Error_path: {:?}", path.as_ref());
    }
}

struct MyHandler<'a> {
    names: &'a mut HashSet<String>,
}

impl<'a> Handler for MyHandler<'a> {
    fn header(&mut self, smf_format: u16, track_count: u16, time_base: u16) {
        if time_base & 0x8000 == 0x8000 {
            let smpte_format = ((time_base & 0x7FFF) >> 8) as i8;
            let tps = time_base & 0xFF;
            info!(
                "header: f={}, tc={}, smpte_f={}, tps={}",
                smf_format, track_count, smpte_format, tps
            );
        } else {
            let tpqn = time_base;
            debug!(
                "header: f={}, tc={}, tpqn={}",
                smf_format, track_count, tpqn
            );
        }
        //self.0 = true;
    }
    fn meta_event(&mut self, delta_time: u32, event: &MetaEvent, data: &Vec<u8>) {
        debug!("meta_event: {:?}", (delta_time, event, data));
        match event {
            MetaEvent::SequenceOrTrackName | MetaEvent::InstrumentName => {
                let mut name = String::from_utf8_lossy(&data[..]).into_owned();
                name.make_ascii_lowercase();
                self.names.insert(name);
            }
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
            MetaEvent::Unknown { event_type } => (),
        }
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
    fn status(&mut self) -> HandlerStatus {
        HandlerStatus::Continue
    }
}
