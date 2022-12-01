import os
import argparse
import json
from extractor_hw import extraction
import traceback

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

# argument parsing
def argument_parser():
    parser = argparse.ArgumentParser(description="""Creator""")
    parser.add_argument(
        "--dir",
        action="store",
        dest="dir",
        required=True,
        help="""directory containing all pdf files""",
    )
    return parser.parse_args()


def validation(args):
    """
    check if the pdf directory provided is a valid path if not raise an exception

    Arguments
    - argument parser

    Raises
    - An assertion error if the path provided is not a valid directory
    """
    assert os.path.exists(args.dir), " Path to the files provided does not exist "


def get_all_documents(root_path, storage):
    """
    This function takes a path to a root folder and returns a list of
    all files by recursively traversing all the folders inside the root folder.

    - Args: A valid path to a folder, an emoty list
    - Returns: List of all files with paths relative to rootpath

    """
    for files in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, files)):
            get_all_documents(os.path.join(root_path, files), storage)
        else:
            storage.append(os.path.join(root_path, files))
    return storage


def main():
    args = argument_parser()
    dir = args.dir
    validation(args)
    pdfs = []
    pdfs = get_all_documents(dir, pdfs)
    finaljson = {}
    extractor = extraction()
    for filepdf in pdfs:
        # check if the file is a pdf file
        if filepdf.endswith(".pdf"):
            truepath = os.path.join(dir, filepdf)
            try:
                output = extractor.extract_pdf(truepath)
                finaljson[truepath] = output
            except Exception:
                print(
                    "Ran into the following problem while extractin text from: ",
                    filepdf,
                )
                print(traceback.format_exc())
                continue
    with open(cnvrg_workdir + "/result.json", "w") as outfile:
        json.dump(finaljson, outfile)


if __name__ == "__main__":
    main()
