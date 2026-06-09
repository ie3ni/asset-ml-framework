import numpy as np

def always_up(y_test):

    return np.ones(
        len(y_test)
    )

def persistence(
        df,
        split_index):

    return (
        df['today_direction']
        .iloc[split_index:]
    )