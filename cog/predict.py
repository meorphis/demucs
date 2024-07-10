import tempfile
from io import BytesIO
from typing import Optional
import concurrent.futures

import torch
from cog import BasePredictor, Input, Path
from torch.cuda import is_available as is_cuda_available

from demucs.api import Separator
from demucs.apply import BagOfModels
from demucs.audio import save_audio
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.repo import AnyModel

from basic_pitch.inference import predict, predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

import fluidsynth

# The demucs API does have a method to get all models but it
# returns models we don't want so it's easier to manually curate
DEMUCS_MODELS = [
    # Demucs v4
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    # Demucs v3
    "hdemucs_mmi",
    # Demucs v2
    # I'm not including the non-quantized versions because
    # according to the author, there is no quality degradation
    # so this should just help speed up boot times
    "mdx_q",
    "mdx_extra_q",
]


class PreloadedSeparator(Separator):
    """
    For efficiency, this cog keeps the models in memory
    so that they don't need to be loaded for every single request.

    The Separator API only supports loading models by name, so
    we have to subclass it and load the model manually.
    """

    def __init__(
        self,
        model: BagOfModels,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
    ):
        self._model = model
        self._audio_channels = model.audio_channels
        self._samplerate = model.samplerate

        self.update_parameter(
            device="cuda" if is_cuda_available() else "cpu",
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
            progress=True,
            callback=None,
            callback_arg=None,
        )


class Predictor(BasePredictor):
    """
    This cog implements the Cog API to inference Demucs models.
    """

    def setup(self):
        """
        Loading the models into memory will provide faster prediction
        when multiple requests are made in succession.
        """
        self.models = {model: get_model(model) for model in DEMUCS_MODELS}
        self.basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

    def predict(
        self,
        audio: Path = Input(description="Upload the file to be processed here."),
        model: str = Input(
            default="htdemucs",
            description="Choose the demucs audio that proccesses your audio. The readme has more information on what to choose.",
            choices=DEMUCS_MODELS,
        ),
        # list of strings
        stems: str = Input(
            default="drums,bass,other,vocals,guitar,piano",
            description="Choose the stems you would like to extract from the audio (comma separated).",
        ),
        # Audio Options
        output_format: str = Input(
            default="mp3",
            description="Choose the audio format you would like the result to be returned in.",
            choices=["mp3", "flac", "wav"],
        ),
        mp3_bitrate: int = Input(
            default=320,
            description="Choose the bitrate for the MP3 output. Higher is better quality but larger file size. If MP3 is not selected as the output type, this has no effect.",
        ),
        mp3_preset: int = Input(
            default=2,
            choices=range(2, 8),
            description="Choose the preset for the MP3 output. Higher is faster but worse wuality. If MP3 is not selected as the output type, this has no effect.",
        ),
        wav_format: str = Input(
            default="int24",
            choices=["int16", "int24", "float32"],
            description="Choose format for the WAV output. If WAV is not selected as the output type, this has no effect.",
        ),
        clip_mode: str = Input(
            default="rescale",
            choices=["rescale", "clamp", "none"],
            description="Choose the strategy for avoiding clipping. Rescale will rescale entire signal if necessary or clamp will allow hard clipping.",
        ),
        # Separator Options
        shifts: int = Input(
            default=1,
            description="Choose the amount random shifts for equivariant stabilization. This performs multiple predictions with random shifts of the input and averages them, which makes it x times slower.",
        ),
        overlap: float = Input(
            default=0.25,
            description="Choose the amount of overlap between prediction windows.",
        ),
        split: bool = Input(
            default=True,
            description="Choose whether or not the audio should be split into chunks.",
        ),
        segment: int = Input(
            default=None,
            description="Choose the segment length to use for separation.",
        ),
        jobs: int = Input(
            default=0,
            description="Choose the number of parallel jobs to use for separation.",
        ),
        midify: bool = Input(
            default=False,
            description="Choose whether or not to midify the separated stems.",
        ),
        input_sample_rate: int = Input(
            default=44100,
            description="Choose the sample rate of the input audio.",
        ),
        input_num_channels: int = Input(
            default=2,
            description="Choose the number of channels in the input audio.",
        ),
        input_duration: float = Input(
            default=5.0,
            description="Choose the duration of the input audio.",
        ),
        input_raw_format: str = Input(
            default="f32le",
            description="Choose the format of the input audio.",
        ),
    ) -> dict:
        model_object: AnyModel = get_model(model)
        max_allowed_segment = float("inf")
        if isinstance(model_object, HTDemucs):
            max_allowed_segment = float(model_object.segment)
        elif isinstance(model_object, BagOfModels):
            max_allowed_segment = model_object.max_allowed_segment
        if segment is not None and segment > max_allowed_segment:
            raise Exception(
                "Cannot use a Transformer model with a longer segment than it was trained for."
            )

        separator = PreloadedSeparator(
            model=model_object,
            shifts=shifts,
            overlap=overlap,
            segment=segment,
            split=split,
            jobs=jobs,
        )

        _, outputs = separator.separate_audio_file(
            audio,
            num_channels=input_num_channels,
            sample_rate=input_sample_rate,
            duration=input_duration,
            raw_format=input_raw_format,
        )

        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": mp3_bitrate,
            "preset": mp3_preset,
            "clip": clip_mode,
            "as_float": wav_format == "float32",
            "bits_per_sample": 24 if wav_format == "int24" else 16,
        }

        output_stems = {}

        input_stems = stems.split(",")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_source, name, source, midify, kwargs, self.basic_pitch_model): name
                for name, source in outputs.items() if name in input_stems
            }
            
            for future in concurrent.futures.as_completed(futures):
                name, audio = future.result()
                if audio:
                    output_stems[name] = audio
        
        return output_stems

def process_source(name, source, midify, kwargs, basic_pitch_model):
    torch_data = source.cpu()

    if midify:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            save_audio(torch_data, f.name, **kwargs)

            model_output, midi_data, note_events = predict(
                f.name,
                basic_pitch_model,
                minimum_note_length=63,
                multiple_pitch_bends=True,
                minimum_frequency=50,
                maximum_frequency=30000
            )

        samples = synthesize_midi(
            midi_data,
            torch_data.numpy(),
            sf2_path="sound_fonts/gameboy.sf2",
            program=(6 if name == "guitar" else 42),
            should_fill=False,
            fs=48000,
        )

        torch_data = torch.from_numpy(samples).float()

    with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:                
        save_audio(torch_data, f.name, **kwargs)
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
        nearby_audio_data = audio_data[int(fs * 2 * note.start):int(fs * 2 * note.end)]
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
