from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from craft_text_detector import Craft
import torch
import numpy as np
import pypdfium2 as pdfium
import fitz


class extraction:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.craft = Craft(
            output_dir=None,
            crop_type="poly",
            export_extra=False,
            link_threshold=0.1,
            text_threshold=0.3,
            cuda=torch.cuda.is_available(),
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detection(self, img):
        """Use craft detector to detect text lines in an image and return a dictionary containing bounding boxes
        Args:
        - Takes single numpy array image as input for doing text detection

        Returns
        - original image
        - A dictionary with key bbox containing a list of detected text in bounding boxes
        """
        prediction_result = self.craft.detect_text(img)
        return img, prediction_result

    def recoginition(self, img, prediction_result):
        """
        OCR on using TrOCR

        Args:
        - img is the numpy array image on which the ocr will done
        - prediction_result is a dictionary contaning a list of bounding boxes, ocr will be done on each bbox

        returns a single raw text containing the text in the image appended for each bbox
        """
        text = []
        for i, j in enumerate(prediction_result["boxes"]):
            roi = img[
                int(prediction_result["boxes"][i][0][1]) : int(
                    prediction_result["boxes"][i][2][1]
                ),
                int(prediction_result["boxes"][i][0][0]) : int(
                    prediction_result["boxes"][i][2][0]
                ),
            ]
            image = Image.fromarray(roi).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            text.append(generated_text)
            print("line " + str(i) + " has been recoginized")
        return prediction_result["boxes"], ("\n").join(text)

    def do_ocr(self, inp):
        "Takes a single numpy array image as input and returns the text"
        img, results = self.detection(inp)
        bboxes, text = self.recoginition(img, results)
        return text

    def pdf_to_img(self, filepath):
        "This function is used to convert pdf pages to images for ocr"
        pdf = pdfium.PdfDocument(filepath)
        renderer = pdf.render_to(
            pdfium.BitmapConv.numpy_ndarray, scale=2, rev_byteorder=True
        )
        return [img for img, _ in renderer]

    def _process_file_pdf(self, file_path):
        """
        This function extracts text from the pdf.
        
        Args:
            - Filepath of the pdf.
        
        Returns:
            - A dictionary containing extracted text from pdf with keys 
            representing each page and values representing text from each page.
        """
        doc = fitz.open(file_path)
        output = {}
        for i,x in enumerate(doc):
            page = doc.load_page(i)
            output[i]=self.cleanlines(page.get_text())
            
        return output
    
    def cleanlines(self, value):
        value = value.strip()
        return ' '.join(value.splitlines())

    def extract_pdf(self, filename):
        """
        This function takes pdf filepath as input and extracts text from that pdf.
        It extracts text from each page where possible and for pages from which extracted
        text is empty it does ocr.

        Args:
            - filepath of the pdf

        Returns:
            - A dictionary containing extracted text from pdf with keys
            representing each page and values representing text from each page.
        """
        output = self._process_file_pdf(filename)
        images = self.pdf_to_img(filename)

        for page in output.keys():
            if output[page] == "":
                output[page] = self.do_ocr(images[page])

        return output
