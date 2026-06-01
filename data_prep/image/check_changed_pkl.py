import joblib
import pickle
import hashlib  # Compare by fingerprint


def check_changed_dict(dic_old, dic_new):
    fp_old = hashlib.md5(pickle.dumps(dic_old)).hexdigest()
    fp_new = hashlib.md5(pickle.dumps(dic_new)).hexdigest()
    if fp_old == fp_new:
        print('Dictionary is not changed.')
        return False
    else:
        print(f'DICTIONARY CHANGED: \nold = {fp_old}, \nnew = {fp_new}.')
        return True


if __name__ == '__main__':
    with open('pkl', "rb") as f:
        data_info_raw = joblib.load(f)
    with open('pkl', "rb") as f:
        data_info_new = joblib.load(f)
    assert check_changed_dict(data_info_new,
                              data_info_raw) == False, \
        "\033[43m"+'WARNING:'+"\033[0m"+' PKL file changed!'