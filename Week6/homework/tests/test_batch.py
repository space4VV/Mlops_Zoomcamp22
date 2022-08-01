
import pandas as pd
from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def test_preprocess(capsys):
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    test_df = prepare_data(df, ['PUlocationID', 'DOlocationID'])
    
    print(test_df)
    captured = capsys.readouterr()
    assert captured.out == test_df
    #assert test_df.shape[0] == 2

    
