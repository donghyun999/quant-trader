from quant.strategies.overlay import parse_spec, split_members
from quant.strategies import get_strategy


def test_parse_spec_separators():
    assert parse_spec("momentum_ensemble(skip_recent=21;top_n=3)") == ("momentum_ensemble", {"skip_recent": 21, "top_n": 3})
    assert parse_spec("trend_invvol(trend_window=100,max_weight=0.5)") == ("trend_invvol", {"trend_window": 100, "max_weight": 0.5})
    name, params = parse_spec("ensemble(members=tranched(base=trend_invvol)|buy_hold)")
    assert name == "ensemble" and params == {"members": "tranched(base=trend_invvol)|buy_hold"}
    assert split_members("tranched(base=a(x=1;y=2)),b|c") == ["tranched(base=a(x=1;y=2))", "b", "c"]


def test_nested_member_spec_builds():
    s = get_strategy("ensemble", members="tranched(base=trend_invvol),tranched(base=momentum_ensemble(skip_recent=21;top_n=3)),buy_hold")
    assert len(s._members()) == 3
