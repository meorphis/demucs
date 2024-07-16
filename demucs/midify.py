
import tempfile
from io import BytesIO
from typing import Optional
import time
import pkg_resources
import os
from scipy.io import wavfile
import requests
import json

import torch
from cog import BasePredictor, Input, Path
import numpy as np

from demucs.audio import save_audio

from basic_pitch.inference import predict, predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

import fluidsynth

def midify_audio(name, numpy_filename, midify, midify_params, save_audio_kwargs, fluidsynth_sample_rate, output_format):
    numpy_data = np.load(numpy_filename)
    os.remove(numpy_filename)
    torch_data = torch.tensor(numpy_data)
    midify_params = json.loads(midify_params) if midify_params else {}

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        save_audio(torch_data, f.name, **save_audio_kwargs)

        if midify:
            start_time = time.time()
            print("Midifying", name)

            instrument_params = midify_params.get(name, {})
            minimum_note_length = instrument_params.get("minimum_note_length", 63)
            onset_threshold = instrument_params.get("onset_threshold", 0.5)
            frame_threshold = instrument_params.get("frame_threshold", 0.3)

            model_output, midi_data, note_events = predict(
                f.name,
                Model(ICASSP_2022_MODEL_PATH),
                minimum_note_length=minimum_note_length,
                multiple_pitch_bends=True,
                minimum_frequency=50,
                maximum_frequency=30000,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )
            print("Done midifying after", time.time() - start_time, "seconds")

            start_time = time.time()
            print("Synthesizing", name)

            program = instrument_params.get("program", 42)
            should_fill = instrument_params.get("should_fill", False)

            samples = synthesize_midi(
                midi_data,
                numpy_data.reshape(-1, 1)[:, 0],
                sf2_path=pkg_resources.resource_filename('demucs', 'sound_fonts/gameboy.sf2'),
                program=program,
                should_fill=should_fill,
                fs=fluidsynth_sample_rate,
            )
            with tempfile.NamedTemporaryFile(suffix=f".wav") as f_inner:
                wavfile.write(f_inner.name, fluidsynth_sample_rate, samples)
                with open(f_inner.name, "rb") as file:
                    audio = BytesIO(file.read())

            print("Done synthesizing after", time.time() - start_time, "seconds")

        else:
            audio = BytesIO(open(f.name, "rb").read())
        
    return name, audio

def synthesize_midi(
    midi,
    audio_data,
    sf2_path=None,
    program=10,
    is_drum=False,
    fs=48000,
    gain=0.2,
    should_fill=False
):
    """Converts a PrettyMidi object to a synthesized waveform using fluidsynth.
    
    Forked from the PrettyMidi library
    
    """
    # If there are no instruments, or all instruments have no notes, return
    # an empty array
    if len(midi.instruments) == 0 or all(len(i.notes) == 0
                                            for i in midi.instruments):
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = [synthesize_instrument(
        i, sf2_path, audio_data, program=program, is_drum=is_drum, fs=fs, gain=gain, should_fill=should_fill
    ) for i in midi.instruments]
    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform
    # Normalize
    synthesized /= np.abs(synthesized).max()
    return synthesized

def synthesize_instrument(
    instrument,
    sf2_path,
    audio_data,
    program=10,
    is_drum=False,
    fs=48000,
    gain=0.2,
    should_fill=False
):
    """Converts a PrettyMidi instrument to a synthesized waveform using fluidsynth.
    
    Forked from the PrettyMidi library
    
    """
    # Create fluidsynth instance
    fl = fluidsynth.Synth(samplerate=fs, gain=gain)
    # Load in the soundfont
    sfid = fl.sfload(sf2_path)
    # If this is a drum instrument, use channel 9 and bank 128
    if instrument.is_drum:
        channel = 9
        # Try to use the supplied program number
        res = fl.program_select(channel, sfid, 128, program)
        # If the result is -1, there's no preset with this program number
        if res == -1:
            # So use preset 0
            fl.program_select(channel, sfid, 128, 0)
    # Otherwise just use channel 0
    else:
        channel = 0
        fl.program_select(channel, sfid, 0, program)
    # Collect all notes in one list
    event_list = []
    for note in transform_notes(instrument.notes) if should_fill else instrument.notes:
        nearby_audio_data = audio_data[int(fs * note.start):int(fs * note.end)]
        # check if every entry is very quiet 
        if np.all(nearby_audio_data < 0.01):
            continue

        event_list += [[note.start, 'note on', note.pitch, note.velocity]]
        event_list += [[note.end, 'note off', note.pitch]]
    for bend in instrument.pitch_bends:
        event_list += [[bend.time, 'pitch bend', bend.pitch]]
    for control_change in instrument.control_changes:
        event_list += [[control_change.time, 'control change',
                        control_change.number, control_change.value]]
    # Sort the event list by time, and secondarily by whether the event
    # is a note off
    event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
    # Add some silence at the beginning according to the time of the first
    # event
    current_time = event_list[0][0]
    # Convert absolute seconds to relative samples
    next_event_times = [e[0] for e in event_list[1:]]
    for event, end in zip(event_list[:-1], next_event_times):
        event[0] = end - event[0]
    # Include 1 second of silence at the end
    event_list[-1][0] = 1.
    # Pre-allocate output array
    total_time = current_time + np.sum([e[0] for e in event_list])
    synthesized = np.zeros(int(np.ceil(fs*total_time)))
    # Iterate over all events
    for event in event_list:
        # Process events based on type
        if event[1] == 'note on':
            fl.noteon(channel, event[2], event[3])
        elif event[1] == 'note off':
            fl.noteoff(channel, event[2])
        elif event[1] == 'pitch bend':
            fl.pitch_bend(channel, event[2])
        elif event[1] == 'control change':
            fl.cc(channel, event[2], event[3])
        # Add in these samples
        current_sample = int(fs*current_time)
        end = int(fs*(current_time + event[0]))
        samples = fl.get_samples(end - current_sample)[::2]
        synthesized[current_sample:end] += samples
        # Increment the current sample
        current_time += event[0]
    # Close fluidsynth
    fl.delete()

    return synthesized

def transform_notes(notes):
    threshold = 0.5

    sorted_notes = sorted(notes, key=lambda x: x.start)
    next_notes = sorted_notes[1:] + [None]
    for note, next_note in zip(sorted_notes, next_notes):
        print (note.start, next_note.start - note.end if next_note is not None else None)
        if next_note is not None and 0 < next_note.start - note.end < threshold:
            note.end = next_note.start
            # scale velocity
            note.velocity = min([int(note.velocity * (next_note.start - note.start) / (note.end - note.start)), 127])

    print(sorted_notes)
    return sorted_notes
