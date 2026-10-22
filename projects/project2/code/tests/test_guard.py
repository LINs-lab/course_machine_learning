
from rsi.selfedit.guard import EDITABLE_SURFACES, check_candidate

CANONICAL = '''"""doc"""
# <<<SELF-IMPROVE-MUTABLE
PRIORITY = ["MOMENTUM", "LR", "CLIP"]
EXPLORE_WIDTH = 3
# SELF-IMPROVE-MUTABLE>>>
'''


def _candidate(mutable):
    return ('"""doc"""\n# <<<SELF-IMPROVE-MUTABLE\n'
            + mutable + "# SELF-IMPROVE-MUTABLE>>>\n")


def test_value_edit_passes():
    v = check_candidate(CANONICAL, _candidate(
        'PRIORITY = ["LR", "MOMENTUM", "CLIP"]\nEXPLORE_WIDTH = 3\n'))
    assert v.ok


def test_immutable_change_rejected():
    v = check_candidate(CANONICAL,
                        '"""other doc"""\n' + _candidate(
                            'PRIORITY = ["LR"]\nEXPLORE_WIDTH = 3\n'))
    assert not v.ok and v.layer == "byte-identity"


def test_syntax_error_rejected():
    v = check_candidate(CANONICAL, _candidate("PRIORITY = [\n"))
    assert not v.ok and v.layer == "ast"


def test_statement_shape_change_rejected():
    # a smuggled new assignment inside the region: same bytes outside, but
    # the top-level statement structure changed
    v = check_candidate(CANONICAL, _candidate(
        'PRIORITY = ["LR", "MOMENTUM", "CLIP"]\nEXPLORE_WIDTH = 3\n'
        '_hijack = lambda: None\n'))
    assert not v.ok and v.layer == "mutable-shape"


def test_missing_sentinels_rejected():
    v = check_candidate(CANONICAL, "PRIORITY = []\n")
    assert not v.ok and v.layer == "sentinel-integrity"


# THE trust boundary test: the improver may never touch the scorer.
def test_scorer_surface_is_rejected():
    v = check_candidate(CANONICAL, _candidate(
        'PRIORITY = []\nEXPLORE_WIDTH = 3\n'),
        surface="src/rsi/evaluation/gates.py")
    assert not v.ok and v.layer == "surface-allowlist"
    assert "state/strategy.py" in EDITABLE_SURFACES


def test_layer4_judge_seam_default_reject():
    mut = 'PRIORITY = ["LR"]\nEXPLORE_WIDTH = 3\n'
    v = check_candidate(CANONICAL, _candidate(mut),
                        judge=lambda text: "override: yes — bad")
    assert not v.ok and v.layer == "override-scan"
    # unparseable judge answer is also a reject (default-reject)
    v2 = check_candidate(CANONICAL, _candidate(mut),
                         judge=lambda text: "no idea")
    assert not v2.ok and v2.layer == "override-scan"
    # judge exception → reject
    def _boom(text):
        raise RuntimeError("judge down")
    v3 = check_candidate(CANONICAL, _candidate(mut), judge=_boom)
    assert not v3.ok and v3.layer == "override-scan"
    # clean pass
    v4 = check_candidate(CANONICAL, _candidate(mut),
                         judge=lambda text: "override: no")
    assert v4.ok
