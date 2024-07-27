
import pandas as pd
import sys
from M3 import M3

DF_COLS = ['M3ID','FILE_FOUND',
           'LON', 'LON_MIN', 'LON_MAX',  
           'LAT', 'LAT_MIN', 'LAT_MAX',
           'INC', 'AZM']

if __name__ == '__main__':
    m3ids = open(sys.argv[1], 'r').read().split('\n')
    data = pd.DataFrame(columns=DF_COLS)
    fnout = f'{sys.argv[2]}.csv'
    print(m3ids)
    for id in m3ids:
        print(id)
        infodict = {}
        for k in DF_COLS: infodict[k] = None
        infodict["M3ID"]=id
        try:
            M3_OBJ = M3(id,
                        m3dir='Data_M3',
                        fn_postfix='')
            infodict['FILE_FOUND'] = True
        except FileNotFoundError as e:
            infodict['FILE_FOUND'] = False

        if infodict['FILE_FOUND']:
            infodict['LAT']         =   M3_OBJ.clat
            infodict['LAT_MIN']     =   M3_OBJ.minlat
            infodict['LAT_MAX']     =   M3_OBJ.maxlat
            infodict['LON']         =   M3_OBJ.clon
            infodict['LON_MIN']     =   M3_OBJ.minlon
            infodict['LON_MAX']     =   M3_OBJ.maxlon
            infodict['INC']         =   M3_OBJ.inc
            infodict['AZM']         =   M3_OBJ.azm
        data = pd.concat([data, pd.DataFrame(infodict, index=[0])])
        data.to_csv(fnout, index=False)

