import pytest

from rsi.selfedit import sentinels


def _src(body="X = 1\n"):
    return f"before\n# <<<SELF-IMPROVE-MUTABLE\n{body}# SELF-IMPROVE-MUTABLE>>>\nafter\n"


def test_split_ok():
    before, mutable, after = sentinels.split_regions(_src("A = 1\nB = 2\n"))
    assert before == "before\n"
    assert mutable == "A = 1\nB = 2\n"
    assert after.startswith("# SELF-IMPROVE-MUTABLE>>>")


def test_two_open_tokens_rejected():
    src = ("# <<<SELF-IMPROVE-MUTABLE\nA = 1\n# <<<SELF-IMPROVE-MUTABLE\n"
           "# SELF-IMPROVE-MUTABLE>>>")
    with pytest.raises(sentinels.SentinelError):
        sentinels.split_regions(src)


def test_wrong_order_rejected():
    src = "# SELF-IMPROVE-MUTABLE>>>\n# <<<SELF-IMPROVE-MUTABLE\n"
    with pytest.raises(sentinels.SentinelError):
        sentinels.split_regions(src)


def test_smuggled_close_token_in_body_rejected():
    # the body contains an EXTRA close token — a boundary cannot be smuggled
    src = ("# <<<SELF-IMPROVE-MUTABLE\nX = 1\n# SELF-IMPROVE-MUTABLE>>>\n"
           "tail\n# SELF-IMPROVE-MUTABLE>>>")
    with pytest.raises(sentinels.SentinelError):
        sentinels.split_regions(src)


def test_unterminated_open_rejected():
    with pytest.raises(sentinels.SentinelError):
        sentinels.split_regions("# <<<SELF-IMPROVE-MUTABLE")


def test_replace_mutable_roundtrip():
    src = _src("OLD = 1\n")
    out = sentinels.replace_mutable(src, "NEW = 2\n")
    _, mutable, _ = sentinels.split_regions(out)
    assert mutable == "NEW = 2\n"
