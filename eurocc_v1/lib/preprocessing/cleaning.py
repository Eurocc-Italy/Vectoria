import logging
from eurocc_v1.lib.preprocessing.regex import remove_header, remove_footer, replace_ligatures, remove_bullets, remove_empty_lines, remove_multiple_spaces

def clean_text(text: str) -> str:
    logger = logging.getLogger('ecclogger')

    # just temporary test, better to encapsulate in a class
    cleanings = [remove_header, remove_footer, replace_ligatures, remove_bullets, remove_empty_lines, remove_multiple_spaces]

    pages_str = ""
    for cleaning_step in cleanings:
        logger.debug("Performing cleaning step: %s", cleaning_step.__name__)
        pages_str = cleaning_step(text)
    
    return pages_str