from quant.report import monthly_table, orders_by_month, format_report_korean, build_report
from quant.paper import PaperAccount


def test_monthly_table_and_orders(tmp_path):
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    acc.buy("AAA", 100, 1000, 0.001, date="2026-01-05")
    acc.snapshot({"AAA": 1000}, "2026-01-05")
    acc.snapshot({"AAA": 1100}, "2026-01-30")
    acc.snapshot({"AAA": 990}, "2026-02-10")
    acc.sell("AAA", 100, 1050, 0.001, date="2026-02-27")
    acc.snapshot({"AAA": 1050}, "2026-02-27")
    m = monthly_table(acc.history)
    assert list(m["month"]) == ["2026-01", "2026-02"]
    assert m["return"].iloc[0] > 0 and m["max_drawdown"].iloc[1] < 0
    o = orders_by_month(acc.orders)
    assert o["orders"].sum() == 2 and o.loc[o.month == "2026-02", "sell_value"].iloc[0] == 105_000
    txt = format_report_korean(build_report(acc))
    assert "2026-01" in txt and "주문 수" in txt


def test_empty_report(tmp_path):
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    assert "기록된 자산 이력이 없습니다" in format_report_korean(build_report(acc))
