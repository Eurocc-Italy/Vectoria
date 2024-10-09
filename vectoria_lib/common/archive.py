#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

# Archive of regex (hopefullty with explanations... :) )

# --------------------------------------------------------------------------------------------------
# HEADER REGEX
# MATCH: "PROCEDURA PRO002 -P-IT REV. 01 PRO0 02-P-IT REV. 01 SELEZIONE, AUTORIZZAZIONE E QUALIFICA DEI FORNITORI"
header = 'PROCEDURA\s+PRO\d+\s+-P-IT\s+REV\.\s+\d+\s+PRO\d\s+\d+-P-IT\s+REV\.\s+\d+\s+SELEZIONE,\s+AUTORIZZAZIONE\s+E\s+QUALIFICA\s+DEI\s+FORNITORI\s+'
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# FOOTER REGEX
# MATCH: "Template: QUA049 -T-CO it rev00       © Copyright Selex ES S.p.A. 201 4 – Tutti i diritti riservati  Pag. 3 di 42"
footer = 'Template:\s+QUA\d+\s+-T-CO\s+it\s+rev\d+\s+©\s+Copyright\s+Selex\s+ES\s+S\.p\.A\.\s+\d+\s+\d+\s+–\s+Tutti\s+i\s+diritti\s+riservati\s+Pag\.\s+\d+\s+di\s+\d+'
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# EMPTY LINES REGEX
# MATCH: empty lines (you don't say...) (Anyhow: watch out for "invisible characters")
# empty_lines = TODO: see regex.py (same comment: when do we delete them? How?)
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# MULTIPLE SPACES REGEX
# MATCH: at least 2 consecutive "space characters"
multiple_spaces = '[ \t]{2,}'
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# LIGATURES REGEX
# MATCH: every most common ligatures in pdf files
ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        "Ꜳ": "AA",
        "Æ": "AE",
        "ꜳ": "aa",
    }
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# BULLET REGEX
# MATCH: • (\u2022), ▪ (\u25AA), ➢ (\u27A2)
bullets = '^\s*[\u2022\u25AA\u27A2]\s*'
# --------------------------------------------------------------------------------------------------

# TODO: do we want to remove "(par. 5.3.1 )" in line "Attività propedeutiche alla Pre -qualifica (par. 5.3.1 )" ???
# TODO: do we want to remove each '\n' as they do in Savia? Maybe we want to exploit the newline in certain sections before removing...
# TODO: "... essere riprodott e, utilizzat e o divulgat e in tutto o in parte..." fix "riprodott e"
# TODO: other cleanings to discuss
