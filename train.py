from system.MidiTransfer import MidiTransfer


if __name__ == '__main__':
    savePath = 'saved/model_weights'
    sChord = 'saved/mscDat/chord_dict.npy'

    MidiTransfer(train=True,load=True,modelSavepath=savePath,chord_path=sChord).train()