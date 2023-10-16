import pandas as pd
import os
from astropy.table import Table
import fitsio
from datetime import datetime


path_data = "/Users/P.Gebhardt/Development/PhD/data"
path_survey = f"{path_data}/sct2"

df_survey = pd.DataFrame()
for idx, file in enumerate(os.listdir(path_survey)):
    if "fits" in file:
        survey_data = Table(fitsio.read(f"{path_survey}/{file}"))
        survey_cols = survey_data.colnames
        df_survey_tmp = pd.DataFrame()
        for i, col in enumerate(survey_cols):
            print(i, col)
            df_survey_tmp[col] = survey_data[col]
        if idx == 0:
            df_survey = df_survey_tmp
        else:
            df_survey = pd.concat([df_survey, df_survey_tmp], ignore_index=True)
        print(df_survey_tmp.shape)

datetime_now = datetime.now().strftime("%Y-%m-%d")
df_survey.to_pickle(f"{path_data}/survey_conditions_{datetime_now}.pkl")
