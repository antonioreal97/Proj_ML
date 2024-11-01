from features.preprocessing import preprocess
from model.models import XGB
import pandas as pd

model_dir = 'model/artifacts/model.pkl'
scaler_dir = 'model/artifacts/scaler.pkl'

entry = {'u_q': -0.4506815075874328,
        'coolant': 18.80517196655273,
        'stator_winding': 19.086669921875,
        'u_d': -0.3500545918941498,
        'stator_tooth': 18.2932186126709,
        'motor_speed': 0.0028655678033828,
        'i_d': 0.0044191367924213,
        'i_q': 0.0003281021781731,
        'pm': 24.554214477539062,
        'stator_yoke': 18.316547393798828,
        'ambient': 19.850690841674805}



df = preprocess(entry)

boost = XGB(model_dir,scaler_dir).realiza_previsao(df)
print(boost)
# print(df)
