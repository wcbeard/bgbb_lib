import datetime as dt

import pandas as pd
from pytest import raises, fixture

from bgbb.sql.sql_utils import to_sql_list, mk_time_params, to_samp_ids

mod_win1 = 90
mod_win2 = 120

ho_win1 = 14
ho_win2 = 21


@fixture
def r1():
    return mk_time_params(
        ho_win=ho_win1, model_win=mod_win1, ho_start="2018-08-01"
    )


@fixture
def r2():
    return mk_time_params(
        ho_win=ho_win2, model_win=mod_win2, ho_start="2001-11-01"
    )


def check_model_range(
    model_start_date=None, ho_start_date=None, model_win=None
):
    """
    if model input period is 14 days, make sure all 14 days are present
    in the range; Basically, ensure inclusive range.
    """
    mod_range = pd.date_range(
        start=model_start_date, periods=model_win, freq="D"
    )
    assert len(mod_range) == model_win
    assert mod_range[0].to_pydatetime().date() == model_start_date
    assert (
        mod_range[-1].to_pydatetime().date() + dt.timedelta(days=1)
    ) == ho_start_date


def check_ho_range(ho_start_date, ho_last_date, ho_win):
    ho_range = pd.date_range(start=ho_start_date, periods=ho_win, freq="D")
    assert len(ho_range) == ho_win
    assert ho_range[0].to_pydatetime().date() == ho_start_date
    assert ho_range[-1].to_pydatetime().date() == ho_last_date


def test_to_sql_list():
    assert to_sql_list(["GB"]) == "'GB'"
    assert to_sql_list(["GB", "US", "IN"]) == "'GB', 'US', 'IN'"
    assert to_sql_list([1, 2, 3]) == "1, 2, 3"


def test_mk_time_param_model_range(r1, r2):
    check_model_range(
        model_start_date=r1.model_start_date,
        ho_start_date=r1.ho_start_date,
        model_win=mod_win1,
    )

    check_model_range(
        model_start_date=r2.model_start_date,
        ho_start_date=r2.ho_start_date,
        model_win=mod_win2,
    )

    # Check this test is actually doing something with wrong
    # `model_win` arg
    raises(
        AssertionError,
        check_model_range,
        model_start_date=r2.model_start_date,
        ho_start_date=r2.ho_start_date,
        model_win=mod_win1,
    )


def test_ho_range(r1, r2):
    check_ho_range(r1.ho_start_date, r1.ho_last_date, ho_win1)
    check_ho_range(r2.ho_start_date, r2.ho_last_date, ho_win2)
    raises(
        AssertionError,
        check_ho_range,
        r2.ho_start_date,
        r2.ho_last_date,
        ho_win1,
    )


def test_to_samp_ids():
    assert to_samp_ids([42]) == "'42'"
    assert to_samp_ids([]) == ''
    assert to_samp_ids(["42"]) == "'42'"
    assert to_samp_ids(["42", 1]) == "'42', '1'"
