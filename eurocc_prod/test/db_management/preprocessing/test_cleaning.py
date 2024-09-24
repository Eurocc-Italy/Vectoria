import pytest
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.preprocessing.cleaning import Cleaning
from vectoria_lib.db_management.preprocessing.regex import (
    remove_header,
    remove_footer,
    replace_ligatures,
    remove_bullets,
    remove_empty_lines,
    remove_multiple_spaces  
)

@pytest.fixture
def cleaning(request):
    c = Cleaning()
    for cleaning_step in request.param:
        c.add_cleaning_step(cleaning_step)
    return c


@pytest.mark.parametrize('cleaning', [[replace_ligatures]], indirect=True)
def test_clean_ligatures(cleaning):
    with open(TEST_DIR / "data/raw/ligatures.txt", "r") as f:
        text = f.read()
    cleaned_text = cleaning.clean_text(text)
    for l in cleaned_text.splitlines():
        assert len(l) > 1

@pytest.mark.parametrize('cleaning', [[remove_empty_lines]], indirect=True)
def test_clean_empty_lines(cleaning):
    with open(TEST_DIR / "data/raw/empty_lines.txt", "r") as f:
        text = f.read()
    cleaned_text = cleaning.clean_text(text)
    assert len(cleaned_text.splitlines()) == 5