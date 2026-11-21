"""scripts/readme_figures.py's M2b charts (docs/m2b-store-and-size.md, S15):
both draw in both themes from m2b_results.chart_data's numbers, their alt
texts hold those numbers, and nothing is drawn until the records are in."""
import pytest

pytest.importorskip("matplotlib")


def spread(mean, low, high, n=None):
    return {"mean": mean, "low": low, "high": high, **({"n": n} if n else {})}


@pytest.fixture
def figures(load_script):
    return load_script("readme_figures")


def chart_data(rf) -> dict:
    """chart_data's shape, with M2b S14's v2 numbers for two tags."""
    m1, m2, m2b = rf.MILESTONES
    return {"tags": {"every expiry": {m1: spread(25, 20, 30, 68), m2: spread(48.67, 46, 51, 68),
                                      m2b: spread(64, 62, 65, 68)},
                     "sideways": {m1: spread(0, 0, 0, 10), m2: spread(6.33, 5, 8, 10), m2b: spread(9.33, 9, 10, 10)}},
            "listed": {"curved": {m1: "0/1", m2: "0/1", m2b: "1/1"}},
            "points": {name: {"found": spread(40 + k, 38 + k, 42 + k, 68), "precision": spread(0.9 - k / 50, 0.85, 0.95)}
                       for k, name in enumerate(rf.POINT_STYLE)}}


def test_both_m2b_charts_draw_in_both_themes_and_say_their_numbers(figures, tmp_path):
    import matplotlib.pyplot as plt
    data = chart_data(figures)
    for theme in figures.THEMES:
        for chart in (figures.by_tag_chart, figures.found_precision_chart):
            fig = chart(data, theme)
            fig.savefig(tmp_path / f"{chart.__name__}_{theme}.png")
            assert fig.get_facecolor()[:3] == pytest.approx(plt.matplotlib.colors.to_rgb(figures.THEMES[theme]["surface"]))
            plt.close(fig)
    assert len(list(tmp_path.glob("*.png"))) == 4
    texts = figures.alt_texts(data)
    assert "every expiry M1 25/68, M2 48.7/68, M2b 64/68; sideways M1 0/10, M2 6.3/10, M2b 9.3/10." in \
        texts["m2b_v2_by_tag"]
    assert "The curved group has one or two expiries and isn't charted (M1 0/1, M2 0/1, M2b 1/1)." in \
        texts["m2b_v2_by_tag"]
    assert "n1280-ft, the M2b detector's arm 43/68 at 0.840" in texts["m2b_v2_found_precision"]


def test_the_m2b_charts_wait_for_the_records(figures, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    monkeypatch.setattr(figures.m2b_results, "chart_data", lambda: None)
    figures.m2b_charts()
    assert "not drawn" in capsys.readouterr().out and not list(tmp_path.iterdir())


def test_the_ordinal_ramp_runs_by_milestone_and_flips_in_dark(figures):
    light, dark = figures.ORDINAL["light"], figures.ORDINAL["dark"]
    assert light == tuple(reversed(dark)) and len(light) == len(figures.MILESTONES) == 3
