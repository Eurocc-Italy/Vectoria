import argparse
from eurocc_v1.paths import DATA_DIR
from eurocc_v1.lib.api.v1 import create_and_write_index

def cli():
    parser = argparse.ArgumentParser(description="EuroCC v1 demo")
    parser.add_argument("-i", "--input-docs-dir", type=str, required=False, default=str(DATA_DIR / "raw"), help="")
    parser.add_argument("-o", "--output-index-dir", type=str, required=False, default=".", help="")
    parser.add_argument("-m", "--hf-embedder-model-name", type=str, required=False, default="BAAI/bge-m3", help="")
    return parser.parse_args()

def main(args):
    create_and_write_index(args.input_docs_dir, args.output_index_dir, args.hf_embedder_model_name)

if __name__ == "__main__":
    main(cli()) 