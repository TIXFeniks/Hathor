
from music21 import converter, stream, note, chord, duration, pitch
from scipy.sparse import csc_matrix

import numpy as np

instruments = ['Piano']

def midiToMatrix(filename):
    parsed = converter.parse(filename)

    party = []

    for part in parsed:
        for voice in part.getElementsByClass(stream.Voice):
            if voice.getInstrument().instrumentName in instruments:
                for thisNote in [n for n in voice if (isinstance(n,note.Note) or isinstance(n,chord.Chord))]:
                    cur_chord = np.zeros(129)
                    for _pitch in thisNote.pitches:
                        cur_chord[_pitch.midi] = 1
                        #text += pitch.name+str(pitch.octave)
                    dur = thisNote.duration.quarterLength
                    cur_chord[-1] = dur#dur.numerator / float(dur.denominator)

                    party.append(cur_chord)
                    #text += dur_to_text(thisNote.duration.type)+'z'

    return csc_matrix(party)

from music21 import midi

def save_mat2_mid(sparsed, fname='output/test.mid'):
    music_stream = stream.Stream()

    for dense_line in np.array(sparsed.todense()):
        (notes,) = np.where(dense_line[:-1]>0.5)

        pitches = []
        for n in notes:
            pitches.append(pitch.Pitch(midi=n))

        crd = chord.Chord(notes= pitches, quarterLength=dense_line[-1])
        music_stream.append(crd)
        
    md = midi.translate.streamToMidiFile(music_stream)
    md.open(fname, 'wb')
    md.write()
    md.close()