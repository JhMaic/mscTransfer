from system.MidiTransfer import MidiTransfer
from datProcessing.Token2Msc import Token2Msc
import os
import sys

if __name__ == '__main__':
    args = sys.argv

    if len(args) == 3:
        a_path = args[1]
        b_path = args[2]
        nMeasure = 25
    elif len(args) == 4:
        a_path = args[1]
        b_path = args[2]
        nMeasure = args[3]

    sChordPath = os.getcwd()+ '\\saved/mscDat/chord_dict.npy'
    modelSavePaht =os.getcwd()+ '\\saved\\model_weights'


    model = MidiTransfer(train=False,load=True, chord_path=sChordPath,modelSavepath=modelSavePaht,nMeasure=nMeasure)
    ab, ba = model.predict(a_path,b_path,'mscPath')
    Token2Msc(ab, sChordPath).mucGen('merge1')
    Token2Msc(ba, sChordPath).mucGen('merge2')
