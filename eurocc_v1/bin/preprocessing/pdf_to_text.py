import os
import logging
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from eurocc_v1.paths import DATA_DIR, OUT_DIR
from tqdm import tqdm

# ref: /leonardo_work/PhDLR_prod/SAVIA/assemblea-legislativa/scrapers/servizi_regione_ER_GR/4_insert_in_mongo.py

def parse_single_doc(file_path):
    text = []
    reader = PdfReader(file_path)

    for page in tqdm(reader.pages[:]):

        # Extract text from the page
        curr_text = page.extract_text()
        if curr_text is None:
            raise ValueError("No text could be extracted from the PDF page.")
        
        text.append(curr_text)
        text.append("\n\n====================================END PAGE====================================n\n")
    
    return "".join(text)


def main():
    logger = logging.getLogger('ecclogger')
    num_errors = 0

    src_folder = DATA_DIR / "raw"
    res_folder = OUT_DIR / "pdf_to_text"
    res_folder.mkdir(exist_ok=True)

    filenames = sorted(os.listdir(src_folder))

    logger.info("num documents: %d", len(filenames))

    for ind, filename in enumerate(filenames[:]):

        logger.info("Parsing pdf #%d: (%s)", ind, filename)

        try:
            # Construct the full file path
            file_path = os.path.join(src_folder, filename)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
    
            # CALL
            extracted_text = parse_single_doc(file_path)
            
            # Write the extracted text to a new file (new filename is just the procedure index)
            output_file_path = os.path.join(res_folder, f"RES_{filename[:12]}.txt") 

            with open(output_file_path, "w") as out_file:
                out_file.write(extracted_text)
            
            logger.info("Extraction and writing completed successfully!")

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
        except PdfReadError as pdf_error:
            logger.error("Error reading PDF file: %s", pdf_error)
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)

    logger.info("Tot errors: %d", num_errors)


if __name__ == "__main__":

    print("Start PDF parsing")
    main()
    print("End PDF parsing")
 