import os
import logging
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from eurocc_v1.paths import DATA_DIR, OUT_DIR

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
    
            # Read the PDF file
            reader = PdfReader(file_path)

            # Write the extracted text to a new file (new filename is just the procedure index)
            output_file_path = os.path.join(res_folder, f"STRUCTURE_{filename[:12]}.txt")
            out_file = open(output_file_path, "w")

            # Extracting possibly relevant metadata
            out_file.write(f"Attachments: {reader.attachments}\n")
            out_file.write(f"Is encrypted? {reader.is_encrypted}\n")
            out_file.write(f"Outline: {reader.outline}\n")
            out_file.write(f"Page labels: {reader.page_labels}\n")
            out_file.write(f"Page layout: {reader.page_layout}\n")
            #print(f"Oggetto {obj_num}: {obj}")
            
            logger.info("Structure extraction completed successfully!")

        except FileNotFoundError as fnf_error:
            num_errors += 1
            logger.error(fnf_error)
        except PdfReadError as pdf_error:
            num_errors += 1
            logger.error("Error reading PDF file: %s", pdf_error)
        except Exception as e:
            num_errors += 1
            logger.error("An unexpected error occurred: %s", e)

    logger.info("Tot errors: %d", num_errors)


if __name__ == "__main__":

    print("Start PDF structure extraction")
    main()
    print("End PDF structure extraction")
 