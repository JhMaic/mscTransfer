from music21 import converter
import numpy as np


class XmlDatParse:
    '''
    parse musical xml to tokens
    '''
    def __init__(self,
                 lpath,
                 sChord,
                 readonly=False,
                 pitchRange=None, # accepted pitch range
                 end='.mxl'):
        self.l_state_seq = [] # state token
        self.l_note_seq = []  # left note token
        self.r_state_seq = []
        self.r_note_seq = []
        self.sChord = sChord
        self.pitchRange = pitchRange if pitchRange else (36, 96)
        self.lpath = lpath
        self.readonly = readonly # do not modify sChord

    def convert(self, combine=False):
        if not isinstance(self.lpath, list): self.lpath = [self.lpath]
        for path in self.lpath:
            try:
                self.xmlDataParse(path)
            except ConnectionError(): pass

        if combine:
            l_state_seq = np.array(self.l_state_seq)[np.newaxis, ..., np.newaxis]
            l_note_seq = np.array(self.l_note_seq)[np.newaxis,]
            r_state_seq = np.array(self.r_state_seq)[np.newaxis, ..., np.newaxis]
            r_note_seq = np.array(self.r_note_seq)[np.newaxis,]
            l_note_state = np.concatenate([l_note_seq, l_state_seq], axis=-1)
            r_note_state = np.concatenate([r_note_seq, r_state_seq], axis=-1)
            return np.concatenate([l_note_state, r_note_state], 0)

        else:
            return self.l_state_seq, self.l_note_seq, self.r_state_seq, self.r_note_seq, self.sChord

    def xmlDataParse(self,path, sChord=None, pitchRange=None):

        try:
            m = converter.parse(path)
            parts = m.getElementsByClass(['PartStaff'])
            assert len(parts) == 2
            left, right = parts
            left = left.chordify()
            right = right.chordify()
            nBars = len(left.getElementsByClass('Measure'))
            assert nBars == len(right.getElementsByClass('Measure'))

            left = left.getElementsByClass('Measure')
            right = right.getElementsByClass('Measure')
            assert len(left) == len(right)

        except:
            raise ConnectionError()

        # left.show('t')
        # print('#######################################')
        # right.show('t')
        #####
        # enumerate respectively
        for lr in zip(left, right):
            left, right = lr
            left = left.flat.getElementsByClass(['Chord', 'Rest'])
            right = right.flat.getElementsByClass(['Chord', 'Rest'])
            ##
            lstates, lnotes = self.get_note(left, left=True, sChord=sChord, pitchRange=pitchRange)
            rstates, rnotes = self.get_note(right, left=False, sChord=sChord, pitchRange=pitchRange)
            if len(lstates) != len(rstates):
                maxl = max(len(lstates), len(rstates))
                if len(lstates) == maxl:
                    rstates = [1] + [0] * (maxl - 1)
                    rnotes = [[-1, 0]] * maxl
                else:
                    lstates = [1] + [0] * (maxl - 1)
                    lnotes = [[-1, 0]] * maxl
            if len(lstates) != len(lnotes):
                raise ValueError('...')

            self.l_state_seq += lstates
            self.l_note_seq += lnotes
            self.r_state_seq += rstates
            self.r_note_seq += rnotes
            ##

            assert len(lstates) == len(rstates)

    def get_note(self,measures,left=True,sChord=None, pitchRange=None ):
        def odd_note_transform(lp: list, ld: list):
            if sum(ld) % 0.25 != 0:
                return None
            else:
                cover = int(sum(ld) / 0.25)
                if cover == 1:  # sum as 16th note
                    return [[lp[-1]], [1]]  # treat it as a 16th note
                else:
                    while (aux := round(cover / len(lp))) == 0 or aux * len(lp) - cover > aux:
                        del lp[int(len(lp) / 2)]
                    ft = aux * len(lp)

                    if (x := ft - cover) > 0:
                        pits = lp[:-2] + [lp[-1]]
                        duras = [aux] * (len(lp) - 1)
                        rem = cover - (len(lp) - 1) * aux
                        if rem != 0:
                            pits += [-1]
                            duras += [rem]
                    elif x < 0:
                        pits = lp + [-1]
                        duras = [aux] * len(lp) + [-x]
                    else:
                        pits = lp
                        duras = [aux] * len(lp)
                    return [pits, duras]

        def noteExtra(note, sChord=None, pitchRange=None, dura=None):
            d = dura if dura else int(note.duration.quarterLength * 4)  # 16th note -> 1

            # X consists of pitch & chord type, where 'type=0' means single note and chord type otherwise
            # X = [pitch, type], where RIST note is considered as a pitch
            X = []
            pitchRange = self.pitchRange if not pitchRange else pitchRange
            sChord = self.sChord if not sChord else sChord

            if 'Rest' in note.classes:
                X = [-1, 0]
            elif 'Chord' in note.classes:
                lpitch = note.pitches
                if len(lpitch) == 1:  # a note
                    lpitch = lpitch[0].midi
                    while lpitch >= pitchRange[1]:
                        lpitch -= 12
                    while lpitch < pitchRange[0]:
                        lpitch += 12
                    X += [lpitch]
                    X += [0]
                else:
                    lpitch_idx = [x.midi for x in lpitch]
                    lpitch_idx = sorted(list(set(lpitch_idx)), reverse=True)
                    while lpitch_idx[0] >= pitchRange[1]:
                        lpitch_idx[0] -= 12
                        lpitch_idx = sorted(list(set(lpitch_idx)), reverse=True)
                    while lpitch_idx[-1] < pitchRange[0]:
                        lpitch_idx[-1] += 12
                        lpitch_idx = sorted(list(set(lpitch_idx)), reverse=True)

                    lpitch_idx = lpitch_idx[:3] if left else lpitch_idx[-3:]

                    while max(lpitch_idx) - min(lpitch_idx) > 12:
                        if left:
                            del lpitch_idx[-1]
                        else:
                            del lpitch_idx[0]

                    if len(lpitch_idx) == 1 or self.readonly:
                        X += [lpitch_idx[0]]
                        X += [0]
                    else:
                        top = max(lpitch_idx)
                        X += [top]
                        arge = tuple(np.add(lpitch_idx, -1 * top))
                        if arge not in sChord:
                            # idx:=0 is single note
                            np.append(sChord,arge)
                        X += [sChord.index(arge)]
            return X, d

        states = []
        notes = []
        cache_note = []
        cache_note_dura = []
        is_odd = False
        for idx, note in enumerate(measures):
            txx = note.duration.quarterLength
            if (note.duration.quarterLength % 0.25) != 0.0 or is_odd is True:
                cache_note += [note]
                cache_note_dura += [note.duration.quarterLength]
                ret = odd_note_transform(cache_note, cache_note_dura)
                if ret is None:
                    is_odd = True
                    continue
                else:
                    _notes, _duras = ret
                    for x in zip(_notes, _duras):
                        _note, _dura = x
                        if _note == -1:
                            X = [-1, 0]
                            d = _dura
                        else:
                            X, d = noteExtra(_note, _dura)
                        states += [1] + [0] * (d - 1)
                        notes += [X] * d

                        assert len(states) == len(notes)
                    is_odd = False
                    cache_note = []
                    cache_note_dura = []
                    continue

            X, d = noteExtra(note, sChord=sChord, pitchRange=pitchRange)

            states += [1] + [0] * (d - 1)
            notes += [X] * d
            assert len(states) == len(notes)

        return states, notes


