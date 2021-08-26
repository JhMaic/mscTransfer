from system.model.cores.nn import *

'''
pre-train model for Embedding model
'''
def EmbedderPreTrainingModel(embedder, odim, pdim, tdim, sdim, nNote):
    notesIn = Input([nNote, odim])

    e_notes = embedder(notesIn)

    x = Conv1D(120, 1)(e_notes)
    x = Linear(120, kernel_initializer='random_normal',
               activation=LeakyReLU(0.2))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    # x = Linear(120*nNote, kernel_initializer='random_normal',
    #            activation=LeakyReLU(0.2))(x)
    # x = LayerNormalization()(x)

    x = Linear(odim,
               kernel_initializer='random_normal',
               activation=LeakyReLU(0.2))(x)
    x = Dropout(0.5)(x)

    state = Linear(sdim, name='note_extractor', activation='softmax')(x)

    lp = Linear(pdim, name='lp_exc', activation='softmax')(x)
    lt = Linear(tdim, name='lt_exc', activation='softmax')(x)

    rp = Linear(pdim, name='rp_exc', activation='softmax')(x)
    rt = Linear(tdim, name='rt_exc', activation='softmax')(x)

    note = Concatenate(-1)([lp, lt, rp, rt, state])

    return Model([notesIn], note)