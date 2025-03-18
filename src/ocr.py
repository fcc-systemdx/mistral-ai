import os
import sys
import json
import datetime
import logging
from pathlib import Path
import argparse

from dotenv import load_dotenv
from mistralai import Mistral

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def write_json_file(data: dict, file_path: Path) -> None:
    """Write JSON data to a file with pretty formatting."""
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')


def write_text_file(text: str, file_path: Path) -> None:
    """Write text to a file."""
    file_path.write_text(text, encoding='utf-8')


def load_api_key() -> str:
    """Load API key from .env file."""
    load_dotenv(override=True)
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError('MISTRAL_API_KEY environment variable not set.')
    return api_key


def create_output_directory(base_dir: Path = None) -> Path:
    """
    Create an output directory with a timestamp.
    If base_dir is not provided, use the current working directory.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = base_dir or Path.cwd()
    output_dir = base_dir / f'output_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def upload_pdf(client: Mistral, pdf_path: Path):
    """
    Upload the PDF to Mistral and return the uploaded file object.
    Raises FileNotFoundError if the PDF file does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF file not found: {pdf_path}')
    with pdf_path.open('rb') as file:
        uploaded_pdf = client.files.upload(
            file={"file_name": pdf_path.name, "content": file.read()},
            purpose="ocr"
        )
    return uploaded_pdf


def process_ocr(client: Mistral, pdf_path: Path, output_dir: Path) -> Path:
    """
    Perform OCR on the PDF, save the JSON response and extracted text,
    and return the path to the JSON response file.
    """
    uploaded_pdf = upload_pdf(client, pdf_path)
    # Get a signed URL
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    logging.info(f"Running OCR for {pdf_path.name}...")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        include_image_base64=True
    )
    logging.info(f"OCR completed for {pdf_path.name}.")

    # Create a subfolder for this PDF file's output
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    response_output_path = pdf_output_dir / "ocr_response.json"
    # Save OCR response to a json file
    write_json_file(ocr_response.model_dump(), response_output_path)
    logging.info(f"OCR response saved to {response_output_path}")

    # Save OCR results to a text file
    ocr_text = "\n\n".join(page.markdown for page in ocr_response.pages) if ocr_response.pages else ""
    output_txt_path = pdf_output_dir / f"{pdf_path.stem}.txt"
    write_text_file(ocr_text, output_txt_path)
    logging.info(f"OCR results saved to {output_txt_path}")

    return response_output_path


def get_images_from_page(page: dict) -> dict:
    """
    Extract image data from a page dictionary and return a dictionary
    mapping image ids to data URIs.
    """
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif'
    }
    images_dict = {}
    for image in page.get('images', []):
        img_id = image.get('id')
        base64_image = image.get('image_base64')
        if base64_image and img_id:
            # Remove header if present
            base64_data = base64_image.split(",", 1)[1] if "," in base64_image else base64_image
            ext = Path(img_id).suffix.lower()
            mime_type = mime_types.get(ext, 'image/png')
            data_uri = f"data:{mime_type};base64,{base64_data}"
            images_dict[img_id] = data_uri
    return images_dict


def json_to_markdown(json_file: Path, output_md: Path) -> None:
    """
    Convert the OCR JSON file to a Markdown file with inline embedded images using base64 data.
    """
    data = json.loads(json_file.read_text(encoding='utf-8'))
    markdown_lines = []
    # If 'pages' key is missing, treat the data as a single page
    pages = data.get('pages', [data] if isinstance(data, dict) else data)
    for page in pages:
        md = page.get('markdown', page.get('text', '')).strip()
        images_dict = get_images_from_page(page)
        # Replace image placeholders with inline base64 images
        for img_id, data_uri in images_dict.items():
            placeholder = f"![{img_id}]({img_id})"
            md = md.replace(placeholder, f"![{img_id}]({data_uri})")
        # Append any images not already included in the markdown
        if images_dict and not any(f"![{img_id}](" in md for img_id in images_dict):
            for img_id, data_uri in images_dict.items():
                md += f"\n![{img_id}]({data_uri})\n"
        markdown_lines.append(md)
    write_text_file("\n\n".join(markdown_lines), output_md)
    logging.info(f"Markdown file saved to {output_md}")


def process_pdf_files(in_dir: Path, out_dir: Path) -> None:
    """
    Process all PDF files in the input directory and save results to the output directory.
    """
    try:
        api_key = load_api_key()
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    client = Mistral(api_key=api_key)
    
    # Create a timestamped output directory
    timestamp_dir = create_output_directory(out_dir)
    logging.info(f"Created output directory: {timestamp_dir}")
    
    # Get all PDF files in the input directory
    pdf_files = list(in_dir.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in {in_dir}")
        return
    
    logging.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        logging.info(f"Processing {pdf_path.name}...")
        try:
            response_output_path = process_ocr(client, pdf_path, timestamp_dir)
            output_md_path = timestamp_dir / pdf_path.stem / f"{pdf_path.stem}.md"
            json_to_markdown(response_output_path, output_md_path)
            logging.info(f"Successfully processed {pdf_path.name}")
        except Exception as e:
            logging.error(f"Error processing {pdf_path.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='OCR tool using Mistral OCR API')
    parser.add_argument('-i', '--input_dir', type=Path, default=Path('in_pdfs'), 
                        help='Directory containing PDF files to process (default: in_pdfs)')
    parser.add_argument('-o', '--output_dir', type=Path, default=Path('out'), 
                        help='Base directory for output files (default: out)')
    args = parser.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir

    # Ensure directories exist
    if not in_dir.exists():
        logging.error(f"Input directory does not exist: {in_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f'Input directory: {in_dir}')
    logging.info(f'Output directory: {out_dir}')

    process_pdf_files(in_dir, out_dir)


if __name__ == '__main__':
    main()
