import os
from datProcessing.XmlDatParse import *

class TransferDataset:
    def __init__(self):
        self.savepath = 'styletransfer\\'
        self.pitchRange = (36,96) # C2 -> B6 [36, 96)


    @staticmethod
    def searchFile(path, end):
        namelist = []
        nfile = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(end):
                    namelist.append(os.path.join(root, file))
                    nfile += 1
        print('searched file for {}'.format(nfile))

        return namelist, nfile

    def loadnpy(self, filenamelist):
        if not isinstance(filenamelist, list):
            raise ValueError('input should be a list')
        PATH = self.savepath
        ret = []
        for name in filenamelist:
            _, n = self.searchFile(PATH, name)
            if n == 0:
                ret.append(None)
            else:
                with open('{}/{}'.format(PATH, name), 'rb') as f:
                    ret.append(np.load(f,allow_pickle=True))
        return ret

    def savenpy(self, list, name):
        savePATH = self.savepath
        os.makedirs(savePATH, exist_ok=True)
        np.save('{}/{}'.format(savePATH, name), np.array(list))

    def parseData(self):
        path = 'style_transfer'
        path1 = path + '\\classical'
        path2 = path + '\\vocaloid'

        ###
        # process first data style
        sChord = [(0)] # distinct set of chords
        n = 0
        f1, _ = TransferDataset.searchFile(path1, '.mxl')

        l_state_seq, l_note_seq, r_state_seq, r_note_seq, sChord = XmlDatParse(f1, sChord, self.pitchRange).convert()

        l_state_seq = np.array(l_state_seq)[np.newaxis,...,np.newaxis]
        l_note_seq = np.array(l_note_seq)[np.newaxis,]
        r_state_seq = np.array(r_state_seq)[np.newaxis,...,np.newaxis]
        r_note_seq = np.array(r_note_seq)[np.newaxis,]
        l_note_state = np.concatenate([l_note_seq, l_state_seq], axis=-1)
        r_note_state = np.concatenate([r_note_seq, r_state_seq], axis=-1)
        self.savenpy(np.concatenate([l_note_state, r_note_state], 0), 'classical.npy')

        ##########################3

        f2, _ = TransferDataset.searchFile(path2, '.mxl')
        l_state_seq, l_note_seq, r_state_seq, r_note_seq, sChord = XmlDatParse(f2, sChord, self.pitchRange).convert()

        l_state_seq = np.array(l_state_seq)[np.newaxis,...,np.newaxis]
        l_note_seq = np.array(l_note_seq)[np.newaxis,]
        r_state_seq = np.array(r_state_seq)[np.newaxis,...,np.newaxis]
        r_note_seq = np.array(r_note_seq)[np.newaxis,]
        l_note_state = np.concatenate([l_note_seq, l_state_seq], axis=-1)
        r_note_state = np.concatenate([r_note_seq, r_state_seq], axis=-1)
        self.savenpy(np.concatenate([l_note_state, r_note_state], 0), 'vocaloid.npy')
        self.savenpy(sChord, 'chord_dict.npy')
