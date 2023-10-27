import traceback
import base64
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from extractor_hw import extraction

extractor = extraction()


def predict(data):
    prediction = {}
    for pdfnumber, filepdf in enumerate(data["pdf"]):
        decoded = base64.b64decode(filepdf)  # decode the input file
        savepath = "file.pdf"
        f = open(savepath, "wb")
        f.write(decoded)
        f.close()
        try:
            text = extractor.extract_pdf(savepath)
        except:
            text = str(traceback.format_exc())
        prediction[pdfnumber] = text

    return prediction
