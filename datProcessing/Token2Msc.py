import numpy as np
from music21.stream import Stream
import music21 as m21

class Token2Msc:
    def __init__(self,
                 mscTensor,
                 sChord,
                 type='tensor'):
        assert type in ['tensor', 'token']
        self.type = type
        self.x = np.array(mscTensor)
        if isinstance(sChord, str):
            self.sChord = np.load(sChord, allow_pickle=True)
        else:
            self.sChord = sChord

    def decode(self, timeseq, sChord:list):
        def getScore(seq):
            stream = Stream()
            the_chord = None
            is_rest = False
            if seq[0][2] != 1:
                seq[0][2] = 1
            dura = 0
            for ts in seq:
                if ts[2] == 1:  # new_note
                    if dura != 0: # save
                        if is_rest:
                            stream.append(m21.note.Rest(duration=m21.duration.Duration(dura/4)))
                            is_rest = False
                        else:
                            the_chord.duration.quarterLength = dura / 4
                            stream.append(the_chord)
                        dura = 0
                    the_chord = m21.chord.Chord()
                    if (pitch := ts[0]) != -1:  # not Rest
                        if (type := ts[1]) == 0:  # a note
                            # n_tmp = m21.note.Note(pitch)
                            # n_tmp.volume.velocity = 60
                            the_chord.add(m21.note.Note(pitch))
                        else:  # a chord
                            nC = sChord[type]
                            for p in np.add(pitch, list(nC)):
                                # n_tmp = m21.note.Note(p)
                                # n_tmp.volume.velocity = 60
                                the_chord.add(m21.note.Note(p))

                    else:
                        is_rest = True
                    dura += 1
                else:
                    dura += 1

            if len(the_chord.notes) == 0:
                stream.append(m21.note.Rest(duration=m21.duration.Duration(dura/4)))
            else:
                the_chord.duration.quarterLength = dura / 4
                stream.append(the_chord)

            return stream
        # time seq = (2, ts, 3)
        # 2 -> left, right ;;; 3 -> (pitch, type, state) where state=1 is onset and continue otherwise
        # [[64,0,1],[65,0,1]]
        left, right = timeseq
        stream = Stream([getScore(left),getScore(right)])

        return stream

    def recreate(self, x):
        # convert tensor to musical tokens
        # [lr, seq, 3]
        lp = np.argmax(x[..., 0:61], axis=-1)[..., np.newaxis]
        lt = np.argmax(x[..., 61:137], axis=-1)[..., np.newaxis]
        rp = np.argmax(x[..., 137:198], axis=-1)[..., np.newaxis]
        rt = np.argmax(x[..., 198:274], axis=-1)[..., np.newaxis]
        state = np.argmax(x[..., 274:], axis=-1)[..., np.newaxis]

        npstate = state
        l_s = np.where(state == (0 or 3), 0, 1)
        r_s = np.where(state == (0 or 2), 0, 1)

        lp = np.add(lp, 35)
        lp = np.where(lp == 35, -1, lp)
        rp = np.add(rp, 35)
        rp = np.where(rp == 35, -1, rp)

        l = np.reshape(np.concatenate([lp, lt, l_s], axis=-1), [1, -1, 3])
        r = np.reshape(np.concatenate([rp, rt, r_s], axis=-1), [1, -1, 3])

        # shape = (2, ts, note)
        # (ts,note) => ([pitch, chord_type, state],[...])
        return np.concatenate([l, r], axis=0)

    def mucGen(self, fname='merged', fmt='mid'):

        x = self.recreate(self.x) if self.type == 'tensor' else self.x
        # gen1 0.36; gen2 0.59  x125*20
        self.decode(x, self.sChord).write(fp='.'.join([fname,fmt]), fmt=fmt)

