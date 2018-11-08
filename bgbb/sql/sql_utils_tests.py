
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
    assert max_dur == duration - 1
    # return f_mp_df
