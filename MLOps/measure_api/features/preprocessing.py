import pickle
import numpy as np
import pandas as pd


extra_feats = {
     'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  
     'S_el': lambda x: x['i_s']*x['u_s'],                  
     'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  
     'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
     'S_x_w': lambda x: x['S_el']*x['motor_speed'],
}


def preprocess(df):
    df = pd.DataFrame(data = df, index=[0])
    df = df.assign(**extra_feats)
    return df



