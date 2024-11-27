import pytest
from vectoria_lib.rag.preprocessing.utils import seek_and_replace

from langchain.docstore.document import Document

@pytest.mark.parametrize("name, regex, replace_with, input_text, expected_text", [
    # Test for multiple spaces
    ("remove_multiple_spaces", r'[ \t]{2,}', ' ', "This  is   a   test.", "This is a test."),
    # Test for bullets
    ("remove_bullets", r'^\s*[\u2022\u25AA\u27A2]\s*', '', "• Bullet point text", "Bullet point text"),
    # Test for ligature "ﬆ" to "st"
    ("remove_ligature_st", r'ﬆ', 'st', "This is a teﬆ.", "This is a test."),
    # Test for ligature "ﬁ" to "fi"
    ("replace_fi", r'ﬁ', 'fi', "This is a ﬁle.", "This is a file."),
    # Test for ligature "ﬂ" to "fl"
    ("replace_fl", r'ﬂ', 'fl', "A ﬂower is blooming.", "A flower is blooming."),
    # Test for ligature "ﬃ" to "ffi"
    ("replace_ffi", r'ﬃ', 'ffi', "An eﬃcient process.", "An efficient process."),
    # Test for ligature "ﬄ" to "ffl"
    ("replace_ffl", r'ﬄ', 'ffl', "An aﬄuent neighborhood.", "An affluent neighborhood."),
    # Test for ligature "ﬅ" to "ft"
    ("replace_ft", r'ﬅ', 'ft', "A draﬅ of the document.", "A draft of the document."),
    # Test for ligature "Ꜳ" to "AA"
    ("replace_AA", r'Ꜳ', 'AA', "An Ꜳbstract concept.", "An AAbstract concept."),
    # Test for ligature "Æ" to "AE"
    ("replace_AE", r'Æ', 'AE', "An Æsthetic choice.", "An AEsthetic choice."),
    # Test for ligature "ꜳ" to "aa"
    ("replace_aa", r'ꜳ', 'aa', "A draꜳft version.", "A draaaft version."),
])
def test_seek_and_replace(config, name, regex, replace_with, input_text, expected_text):
    doc = Document(input_text)
    updated_doc = seek_and_replace(doc, [{ "name": name, "pattern": regex, "replace_with": replace_with }])
    assert updated_doc.page_content == expected_text
