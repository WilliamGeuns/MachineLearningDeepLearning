import numpy as np
import mne
import pandas as pd

mne.set_log_level('WARNING')

'''
df format

epoch 1                 epoch ...               epoch 193
[Chanel    time step  ] [Chanel    time step  ] [Chanel    time step  ]
[   1      0 ... 3000 ] [   1      0 ... 3000 ] [   1      0 ... 3000 ]
[   .  .              ] [   .  .              ] [   .  .              ]
[   .     .           ] [   .     .           ] [   .     .           ]
[   .        .        ] [   .        .        ] [   .        .        ]
[   31          .     ] [   31          .     ] [   31          .     ]
'''

df = mne.io.read_epochs_eeglab('G:\\a Data\\fam2animalsonly\\sph\\Preprocessed Data\\distractor\\sphdistractor.set')
i=0

for epoch in df:
    i=i+1
    dfNew1 = pd.DataFrame(epoch)
    np.savetxt("sph d " + format(i, '03d') + ".csv",dfNew1,delimiter=',')
