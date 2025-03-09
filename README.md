# PDF to Markdown Conversion Code (Mistral AI OCR API)

This project, named `mistral-ocr-pdf2markdown`, converts PDF documents into Markdown format by leveraging the Mistral OCR API. It extracts text and images from PDFs and generates a Markdown file with inline images. This tool requires a valid Mistral API key and Python 3 to run.

---

## Prerequisites

- Python 3
- [mistralai](https://pypi.org/project/mistralai/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

Install the required packages:

```bash
pip install mistralai python-dotenv
```

## Setup

1. **Clone the repository or place the files**

   Ensure the following directory structure:

   ```
   ├── ocr.py
   ├── README.md
   └── .env
   ```

2. **Environment Variable Setup**

   Create a `.env` file in the project root with the following content:

   ```env
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

   Alternatively, you can set the environment variable directly.

## Usage

Run the script from the terminal:

```bash
python ocr.py --pdf /path/to/your/file.pdf --output /path/to/output_directory
```

- `--pdf`: Path to the PDF file to process
- `--output`: Base directory for output files

## Project Structure

```
├── ocr.py         # Main script for OCR processing
├── README.md      # This README file
└── .env           # Environment variable file (contains MISTRAL_API_KEY)
```

## License

This project is licensed under the MIT License. Please adhere to the license terms of the dependent libraries and the Mistral OCR API.

## Disclaimer

This tool uses the Mistral OCR API. Be aware of any rate limits or billing constraints associated with the API. For more details, see the [Mistral API documentation](https://docs.mistral.ai/api/).

---
# PDFからMarkdownの変換コード（Mistral AI OCR API）
このプロジェクト（`mistral-ocr-pdf2markdown`）は、Mistral OCR API を活用して PDF 文書を Markdown 形式に変換するツールです。PDFからテキストや画像を抽出し、インライン画像付きのMarkdownファイルを生成します。

## 解説記事
[Mistral OCR APIを使ってPDFをMarkdownファイルに変換してみた！（画像埋め込み対応🚀）](https://zenn.dev/rynskrmt/articles/123d022aee4728)

## 必要条件

- Python 3
- [mistralai](https://pypi.org/project/mistralai/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

以下のコマンドで必要なパッケージをインストールします:

```bash
pip install mistralai python-dotenv
```

## セットアップ

1. **リポジトリのクローンまたはファイルの配置**

   下記のようなディレクトリ構成にしてください（例）:

   ```
   ├── ocr.py
   ├── README.md
   └── .env
   ```

2. **環境変数の設定**

   プロジェクトのルートディレクトリに `.env` ファイルを作成し、以下の内容を記述してください:

   ```env
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

   もしくは、環境変数を直接設定してください。

## 使い方

ターミナルでスクリプトがあるディレクトリに移動し、以下のように実行します:

```bash
python ocr.py --pdf /path/to/your/file.pdf --output /path/to/output_directory
```

- `--pdf`: 処理するPDFファイルのパス
- `--output`: 出力ファイルの基底ディレクトリ

## プロジェクト構成

```
├── ocr.py         # OCR処理を行うメインスクリプト
├── README.md      # このREADMEファイル
└── .env           # 環境変数ファイル（MISTRAL_API_KEY を含む）
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。依存しているライブラリやMistral OCR APIの利用規約にも準拠してください。

## 免責事項

本ツールはMistral OCR APIを利用しています。APIの利用にあたっては、レートリミットや課金などの制約がある場合があります。詳細は[Mistral APIドキュメント](https://docs.mistral.ai/api/)をご確認ください。
