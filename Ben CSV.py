# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:39:27 2018

@author: William
"""

import mne
import pandas as pd
import numpy as np

# ['T','H','K','N','L']
Names = ['L']
Numbers = [1,2,3,4,5,6]

TRIGGER = 10

for name in Names:
    for number in Numbers:
        try:            
            df = mne.io.read_epochs_eeglab('D:\\Datasets\\Data\\2\\10\\{} Test{} T10.set'.format((name),(number)))
        
            i = 0
            for epoch in df:
                i=i+1
                dfNew1 = pd.DataFrame(epoch)
                np.savetxt(str(name)+" Test"+str(number)+" T10 "+format(i,'02d')+".csv",dfNew1,delimiter=',')
                
        except IOError:
            print("hos")
print("DONEEEE")  
        
            
    
#for name in Names:
# for number in Numbers:
#     try:
#      df = mne.io.read_epochs_eeglab('D:\\Datasets\\Data\\'+str(CLASS)+'\\'+str(TRIGGER)+'\\{} Test{} T'+str(TRIGGER)+'.set'.format((name),(number)))
      #df = mne.io.read_epochs_eeglab('D:\\Datasets\\Data\\7\\9\\{} Test{} T09.set'.format((name),(number)))
#      if IOError == False:
#         for ep in df:
#             dfn = pd.DataFrame(ep)
#             namez = name + ' Test{} T'+str(TRIGGER)+'.csv'.format((number)) 
#             dfn.to_csv(namez, sep=',', header=False,index=False)
#     except IOError:
#         print(str(i)+' file D:\\Datasets\\Data\\2\\10\\{} Test{} T10.set doesnt exist, continuing'.format((name),(number)))
     
#print("DONE, CHANGE THE PATH AND TRIGGERS EVERYWHERE")