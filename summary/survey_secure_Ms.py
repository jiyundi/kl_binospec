import joblib


def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    return data_info


def find_secure_Ms():
    slitnums  = []
    secures = []
    
    for slit_num in range(1, 143):
        try:
            data_info = another_load_mock(
                pkl_folder='../scripts/binospec_pkl/', 
                slit_num=slit_num)
            
        except FileNotFoundError:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no PKL found.')
            continue
        
        z = data_info['galaxy']['redshift']
        M = data_info['galaxy']['log10_Mstar']
        
        if (z is not None) & (M is not None):
            accept = True
        else:
            accept = False
        
        slitnums.append(slit_num)
        secures.append(bool(accept))
                
    return slitnums, secures


if __name__ == '__main__':
    slitnums, secures = find_secure_Ms()