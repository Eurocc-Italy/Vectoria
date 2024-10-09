#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from pathlib import Path

import docx
from langchain.docstore.document import Document

from vectoria_lib.db_management.preprocessing.document_data import  DocumentData

logger = logging.getLogger('db_management')
    
def extract_text_from_docx_file(file_path: Path, filter_paragraphs: list, log_in_folder: Path = None) -> list[Document]:

    logger.debug("Extracting text from %s", file_path.stem)

    document = docx.Document(file_path)

    document_flat_structure = _extract_flat_structure(document)

    if log_in_folder is not None:
        Path(log_in_folder).mkdir(parents=True, exist_ok=True)  
        _log_document_structure_on_file(document_flat_structure, Path(log_in_folder) / f"{file_path.stem}_structure.txt")    
    
    # TODO: what to do with unstructured_data?
    structured_data, unstructured_data = _extract_text_structure(document_flat_structure) 

    structured_data = _get_flat_docs_list(structured_data)

    logger.debug("Extracted %d documents from %s", len(structured_data), file_path.stem)



    return structured_data

def _log_document_structure_on_file(structure, file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        # Print the document structure, including where tables are located
        for element in structure:
            if element[0].startswith("Heading"):
                print(f"{element[0]}: {element[1]}", file=f)
            elif element[0] == "Paragraph":
                print(f"   Paragraph: {element[1]}", file=f)
            elif element[0] == "Table":
                print(f"   Table under Heading: {element[2]}", file=f)
                for row in element[1]:
                    print(f"      {row}", file=f)

def _extract_flat_structure(doc):

    current_heading = None  # Track the current heading
    structure = []  # Store content structure

    for element in doc.element.body:
        if element.tag.endswith('p'):  # It's a paragraph
            paragraph = docx.text.paragraph.Paragraph(element, doc)
            heading_level = _get_heading_level(paragraph)
            if heading_level:
                current_heading = paragraph.text  # Update current heading
                structure.append((f"Heading {heading_level}", paragraph.text))
            else:
                if paragraph.text:
                    structure.append(("Paragraph", paragraph.text))
        
        elif element.tag.endswith('tbl'):  # It's a table
            table_data = _extract_table_data(element, doc)
            if not _check_table_empty(table_data):
                structure.append(("Table", table_data, current_heading))

    return structure


def _get_heading_level(paragraph):
    # A helper function to determine the level of a heading based on style
    if paragraph.style.name.startswith("Heading"):
        return int(paragraph.style.name.split()[-1])  # Extract heading level number
    return None  # Not a heading


def _check_table_empty(table):
    # Check if a table is empty
    for row in table:
        for col in row:
            if col:
                return False
    return True


def _extract_table_data(table, doc):
    table = docx.table.Table(table, doc)
    table_data = []
    for row in table.rows:
        row_data = [cell.text for cell in row.cells]
        table_data.append(row_data)
    return table_data
    



def _extract_text_structure(document_flat_structure):
    """
    ######## From:

    document_flat_structure = [
        ('Paragraph', 'Questa è benzina'),
        ('Paragraph', 'non strutturata.'),
        ('Heading 1', 'INTRODUZIONE'), # capitolo 1
        ('Heading 2', 'Scopo'), # capitolo 1.1
        ('Paragraph', 'Lo scopo del presente documento è la descrizione delle regole per la gestione, l’elaborazione ed emissione dei documenti del sistema normativo della Divisione Elettronica.'),
        ('Paragraph', 'Il BMS è lo strumento per la pubblicazione, la diffusione e l’archiviazione dei documenti di origine interna, i modelli/moduli ad essi associati, la documentazione emessa da Corporate e ogni altro documento esterno con particolare riferimento a leggi internazionali e standard, che si ritenga opportuno diffondere, purché non coperti da copyright di terze parti.'),
        ('Heading 2', 'Applicabilità'), # capitolo 1.2
        ('Paragraph', 'Il presente documento si applica alla Divisione Elettronica (perimetro Italia).'),
        ('Heading 1', 'RIFERIMENTI'), # capitolo 2
        ('Heading 2', 'Documenti'), # capitolo 2.1
        ('Paragraph', 'I documenti sotto riportati sono da intendere nell’ultima versione correntemente emessa, se non diversamente riportato.'),
        ('Table', [['Codice', 'Titolo'], ['AQAP-2110', 'NATO Quality Assurance Requirement for Design Development and Production'], ['AQAP-2210', 'NATO Supplementary Software Quality Assurance Requirement to AQAP2110 or AQAP2310']], 'Documenti'),
        ('Paragraph', 'NOTA: I documenti indicati con (*) sono in fase di emissione alla data di pubblicazione del presente documento. Nel transitorio si applicano i corrispondenti documenti Legacy.')
    ]

    ######### to:
    
    [{'childs': [{'childs': [],
                'name': 'Scopo',
                'docs': [Document('Lo scopo del presente documento è la descrizione delle '
                        'regole per la gestione, l’elaborazione ed emissione '
                        'dei documenti del sistema normativo della Divisione '
                        'Elettronica.'),
                        Document('Il BMS è lo strumento per la pubblicazione, la '
                        'diffusione e l’archiviazione dei documenti di origine '
                        'interna, i modelli/moduli ad essi associati, la '
                        'documentazione emessa da Corporate e ogni altro '
                        'documento esterno con particolare riferimento a leggi '
                        'internazionali e standard, che si ritenga opportuno '
                        'diffondere, purché non coperti da copyright di terze '
                        'parti.')]},
                {'childs': [],
                'name': 'Applicabilità',
                'docs': [Document('Il presente documento si applica alla Divisione '
                        'Elettronica (perimetro Italia).')]}],
    'name': 'INTRODUZIONE',
    'docs': []},
    {'childs': [{'childs': [],
                'name': 'Documenti',
                'docs': [Document('I documenti sotto riportati sono da intendere '
                        'nell’ultima versione correntemente emessa, se non '
                        'diversamente riportato.'),
                        Document("[['Codice', 'Titolo'], ['AQAP-2110', 'NATO Quality "
                        'Assurance Requirement for Design Development and '
                        "Production'], ['AQAP-2210', 'NATO Supplementary "
                        'Software Quality Assurance Requirement to AQAP2110 or '
                        "AQAP2310']]"),
                        Document('NOTA: I documenti indicati con (*) sono in fase di '
                        'emissione alla data di pubblicazione del presente '
                        'documento. Nel transitorio si applicano i '
                        'corrispondenti documenti Legacy.')]}],
    'name': 'RIFERIMENTI',
    'docs': []}]
    """

        
    structured_data = []
    unstructured_data = Document(page_content="", metadata={"type": "unstructured"})
    heading_stack = []  # This stack will keep track of current headings at different levels

    # Find for unstructured data first
    for item in document_flat_structure:
        if not item[0].startswith('Heading'):
            unstructured_data.page_content += str(item[1])
        else:
            break
        
    current_name = None
    current_level = None
    current_id = 0
    
    for item in document_flat_structure:

        # Check if the item is a heading (e.g., 'Heading 1', 'Heading 2', etc.)
        if item[0].startswith('Heading'):
            level = int(item[0].split()[1])  # Extract heading level (e.g., 1, 2, 3, etc.)
            current_name = item[1]
            current_level = level
            current_id = 0
            heading_dict = {
                "name": item[1],
                "doc": None,
                "childs": []
            }
            
            # Adjust the stack based on heading level
            while len(heading_stack) >= level:

                heading_stack.pop()  # Go up to the correct parent level by popping

            if heading_stack:
                
                # Add the new heading as a child of the current top of the stack
                heading_stack[-1]["childs"].append(heading_dict)
            else:
                # If the stack is empty, it means we're at the top level
                structured_data.append(heading_dict)
                
            # Push the new heading onto the stack
            heading_stack.append(heading_dict)
        
        # If the item is a paragraph or table, add it to the current section
        elif item[0] == 'Paragraph' or item[0] == 'Table':
            text = str(item[1])
            if heading_stack:
                if heading_stack[-1]["doc"] is None:
                    heading_stack[-1]["doc"] = Document(page_content=text, metadata={"name": current_name, "level": current_level, "id": current_id})
                    current_id += 1
                else:
                    heading_stack[-1]["doc"].page_content += text
                            

    return structured_data, unstructured_data


def _get_flat_docs_list(structured_data) -> list[Document]:
    # recursively extract all the Document objects from the children of structured_data
    docs = []
    for item in structured_data:
        if item["doc"] is not None:
            docs.append(
                item["doc"]
            )
        docs.extend(_get_flat_docs_list(item["childs"]))
    return docs