import pandas as pd
import datetime as dt


def assert_eq(a, b):
    assert a == b, "{} != {}".format(a, b)


def test_model_range(model_start_date=None, ho_start_date=None, MODEL_WIN=None):
    """
    if model input period is 14 days, make sure all 14 days are present
    in the range; Basically, ensure inclusive range.
    """
    mod_range = pd.date_range(start=model_start_date, periods=MODEL_WIN, freq="D")
    assert_eq(len(mod_range), MODEL_WIN)
    assert_eq(mod_range[0].to_pydatetime().date(), model_start_date)
    assert_eq(
        mod_range[-1].to_pydatetime().date() + dt.timedelta(days=1), ho_start_date
    )


def test_ho_range(ho_start_date, ho_last_date, HO_WIN):
    ho_range = pd.date_range(start=ho_start_date, periods=HO_WIN, freq="D")
    assert_eq(len(ho_range), HO_WIN)
    assert_eq(ho_range[0].to_pydatetime().date(), ho_start_date)
    assert_eq(ho_range[-1].to_pydatetime().date(), ho_last_date)


def test_rft(df, duration=None):
    """
    df: has columns `Frequency, Recency, T, Max_day, Min_day`
    """
    df = df["Frequency Recency T Max_day Min_day".split()]
    freq_mp = duration - 1
    f_mp_df = df.query("Frequency == {}".format(freq_mp))
    assert (f_mp_df[["Recency", "Frequency", "T"]] == freq_mp).all().all()
    assert df.eval("Recency <= T").all()
    assert df.eval("Recency >= Frequency").all()
    assert (df.Frequency.eq(0) == df.Recency.eq(0)).all()

    max_dur = df.eval("Max_day - Min_day").astype("timedelta64[D]").astype(int).max()
    assert_eq(max_dur, duration - 1)
    # return f_mp_df
