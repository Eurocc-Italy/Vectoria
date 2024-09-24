import re

def extract_metadata(text):

    # TODO: add checks for string matching or set to defalut value
        
    # Regular expressions to extract the needed values
    id = re.search(r"IDENTIFICATIVO\s*:\s*(.*)", text).group(1).strip()
    date = re.search(r"DATA\s*:\s*(.*)", text).group(1).strip()
    doc_type = re.search(r"TIPO DOCUMENTO\s*:\s*(.*)", text).group(1).strip()
    app = re.search(r"APPLICAZIONE\s*:\s*(.*)", text).group(1).strip()
    summary = re.search(r"SOMMARIO\s*:\s*(.*(?:\n.*)*)", text).group(1).strip()

    return {"id": id,
            "date": date,
            "doc_type": doc_type,
            "app": app,
            "summary": summary}

def print_metadata(metadata):

    for key, value in metadata.items():
        print(f"{key}: {value}")

def main():

    text = """
    Le informazioni contenute nel presente documento sono di proprietà di Selex ES S.p.A.  e non possono, al pari di tale documento, essere 
    riprodott e, utilizzat e o divulgat e in tutto o in parte a terzi senza preventiva autorizzazione scritta di Selex ES S.p.A.  
    Il documento è disponibile nell’Intranet Aziendale/BMS di Selex ES S.p.A. Le copie, sia in formato elettronico che cartaceo dovranno 
    essere verificate, prima dell’utilizzo, con la versione vigente disponibile su Intranet.  
    © Copyright Selex ES S.p.A. 2014 - Tutti i diritti riservati  

    IDENTIFICATIVO : PRO0 02-P-IT Rev. 01 
    DATA: 30/12/2015  
    TIPO DOCUMENTO : PROCEDURA  
    APPLICAZIONE : Selex ES S.p.A.  

    Selezione, Autorizzazione e Qualifica dei 
    Fornitori   

    SOMMARIO : 
    Il presente documento descrive le attività di selezione, autorizzazione e qualifica dei fornitori utilizzabili in 
    Selex ES S.p.A  e la gestione degli stessi nell’Elenco Fornitori Qualificati (EFQ)  e nell’Anagrafica Fornitori SAP . 
    """

    metadata = extract_metadata(text=text)

    print_metadata(metadata)

if __name__ == "__main__":
    main()