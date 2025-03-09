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
    load_dotenv()
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError('MISTRAL_API_KEYが設定されていません')
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
        raise FileNotFoundError(f'指定されたPDFファイルが見つかりません: {pdf_path}')
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
    # 署名付きURLを取得
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed_url.url},
        include_image_base64=True
    )

    response_output_path = output_dir / "ocr_response.json"
    # OCRレスポンスをJSONファイルに保存
    write_json_file(ocr_response.model_dump(), response_output_path)
    logging.info(f"OCRレスポンスを {response_output_path} に保存しました。")

    # OCR結果をテキストファイルに保存
    ocr_text = "\n\n".join(page.markdown for page in ocr_response.pages) if ocr_response.pages else ""
    output_txt_path = output_dir / "output.txt"
    write_text_file(ocr_text, output_txt_path)
    logging.info(f"OCR結果を {output_txt_path} に保存しました。")

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
    logging.info(f"Markdownファイルを {output_md} に保存しました。")


def main():
    parser = argparse.ArgumentParser(description='OCR tool using Mistral OCR API')
    parser.add_argument('-p', '--pdf', type=Path, required=True, help='Path to the PDF file to process')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Base directory for output files')
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    output_base: Path = args.output

    logging.info(f'Using PDF file: {pdf_path}')
    logging.info(f'Output base directory: {output_base}')

    try:
        api_key = load_api_key()
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    client = Mistral(api_key=api_key)
    output_dir = create_output_directory(output_base)

    try:
        response_output_path = process_ocr(client, pdf_path, output_dir)
    except Exception as e:
        logging.error(f"OCR処理中にエラーが発生しました: {e}")
        sys.exit(1)

    output_md_path = output_dir / 'output.md'
    try:
        json_to_markdown(response_output_path, output_md_path)
    except Exception as e:
        logging.error(f"Markdown変換中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
