#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import re
from vectoria_lib.common.config import Config

def remove_header(text):    
    RE_HEADER = re.compile(Config().get("header_regex"), re.IGNORECASE) 
    return RE_HEADER.sub("", text).strip() 

def remove_footer(text):
    RE_FOOTER = re.compile(f'{Config().get("footer_regex")}', re.IGNORECASE) 
    return RE_FOOTER.sub("", text).strip()

def remove_empty_lines(text): # TODO: add regex instead of building a temp list
    
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != '']
    result = '\n'.join(non_empty_lines)
    
    return result

def remove_multiple_spaces(text):
    
    _RE_COMBINE_WHITESPACE = re.compile(r"[ \t]{2,}")
    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
    
    return text

def replace_ligatures(text: str):
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
    for search, replace in ligatures.items():
        text = text.replace(search, replace)
        
    return text

def remove_bullets(text):  # • (\u2022), ▪ (\u25AA), ➢ (\u27A2)
    
    _RE_COMBINE_WHITESPACE = re.compile(r"^\s*[\u2022\u25AA\u27A2]\s*", flags=re.MULTILINE)
    text = _RE_COMBINE_WHITESPACE.sub("", text).strip() # Important to add multiline flag
    
    return text

# TODO: do we want to remove "(par. 5.3.1 )" in line "Attività propedeutiche alla Pre -qualifica (par. 5.3.1 )" ???
# TODO: do we want to remove each '\n' as they do in Savia? Maybe we want to exploit the newline in certain sections before removing...
# TODO: "... essere riprodott e, utilizzat e o divulgat e in tutto o in parte..." fix "riprodott e"
# TODO: other cleanings to discuss

# --------------------------------------------------------------------------------------------

# def main():

#     text = """SOMMARIO : 
# Il presente documento descrive le attività di selezione, autorizzazione     e qualifica dei fornitori utilizzabili in
  
# Selex ES S.p.A  e la gestione degli stessi nell’Elenco Fornitori Qualificati (EFQ)  e nell’Anagrafica Fornitori SAP . 

#  PROCEDURA  
# PRO002 -P-IT REV. 01 PRO0 02-P-IT REV. 01 
# SELEZIONE, AUTORIZZAZIONE E QUALIFICA DEI FORNITORI   
 
# TEST LIGATURES (double ff): aﬀiliato
 
# Template: QUA049 -T-CO it rev00       © Copyright Selex ES S.p.A. 201 4 – Tutti i diritti riservati  Pag. 2 di 42

# TEST BULLETS
# creato danni materiali o di immagine Valutazioni di Procurement sulla 
# necessità del rinnovo di qualifica  
# • Ripetuti rilievi di bassi livelli di prestazioni  
# • Violazione dei requisiti riportati nella direttiva n. 21 Finmeccanica del 
# 30.03.2015.  
# Descrizione:  Attivazione del processo  (par. 5.2) 
# La qualifica e/o l’autorizzazione di un fornitore si può attivare sia per 
# autocandidatura del fornitore sia per richieste interne che portano a 
# coinvolgere un  determinato fornitore.  
# Fornitori soggetti a Qualifica (fornitori diretti)  (par.5.3) 
# ➢ Attività propedeutiche alla Pre -qualifica  (par. 5.3.1 ) 
# ➢ Sviluppo e conclusione della Pre -qualifica  (par. 5.3.2 ) 
# SBC attua una serie di verifiche coinvolgendo SHE, ed eventualmente 

# """

#     print(f"\n------------- START ORIGINAL TEXT -------------")
#     print(text)
#     print(f"------------- END ORIGINAL TEXT -------------\n")

#     text = remove_header(text)
#     text = remove_footer(text)
#     text = replace_ligatures(text)
#     text = remove_bullets(text)
#     text = remove_multiple_spaces(text)
#     text = remove_empty_lines(text)
    
#     print(f"\n------------- START CLEANED TEXT -------------")
#     print(text)
#     print(f"------------- END CLEANED TEXT -------------\n")


# if __name__ == "__main__":

#     print("START TEST CLEANING")
#     main()
#     print("END TEST CLEANING")
 
