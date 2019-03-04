import pytest
from wing.preprocess import * 

def test_add_datepart(): 
    df = pd.DataFrame({'A': ['2000-01-01', '2200-12-12', '2010-07-01', '1999-12-31']})
    add_datepart(df, 'A')
    assert df.A_Year[0] == 2000
    assert df.A_Is_month_end[0] == False
    assert df.A_Is_month_start[2] == True
    assert df.A_Is_quarter_start[2] == True
    assert df.A_Day[2] == 1
    assert df.A_Day[3] == 31
    assert df.A_Month[2] == 7
    assert df.A_Quarter[1] == 4

    df1 = pd.DataFrame({'A': ['2000-01-01 00:23:21', '2200-12-12 12:41:20', '2010-07-01 14:01:00', '1999-12-31 06:12:30']})
    add_datepart(df1, 'A', time=True)
    assert df1.A_Month[2] == 7
    assert df1.A_Minute[2] == 1
    assert df1.A_Second[3] == 30
    assert df1.A_Hour[0] == 0

    df2 = pd.DataFrame({'A': ['2000-01-01 00:23:21', '2200-12-12 12:41:20', '2010-07-01 14:01:00', '1999-12-31 06:12:30']})
    add_datepart(df2, 'A', drop=False)
    assert 'A' in df2.columns