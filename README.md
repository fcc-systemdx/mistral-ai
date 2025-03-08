# PDF to Markdown Conversion Sample Code (Mistral AI OCR API)

PDFからMarkdownの変換サンプルコード（Mistral AI OCR API）

This project, named `mistral-ocr-pdf2markdown`, converts PDF documents into Markdown format by leveraging the Mistral OCR API. It extracts text and images from PDFs and generates a Markdown file with inline images. This tool requires a valid Mistral API key and Python 3 to run.

このプロジェクト（`mistral-ocr-pdf2markdown`）は、Mistral OCR API を活用して PDF 文書を Markdown 形式に変換するツールです。PDFからテキストや画像を抽出し、インライン画像付きのMarkdownファイルを生成します。

---

## English

### Prerequisites

- Python 3
- [mistralai](https://pypi.org/project/mistralai/)

Install the required package:

```bash
pip install mistralai
```

### Setup

1. **Clone the repository or place the files**

   Ensure the following directory structure:

   ```
   ├── ocr.py
   ├── README.md
   └── (Optional: .env)  # Only if you use dotenv (not required in this setup)
   ```

2. **Environment Variable Setup**

   Set your Mistral API key as an environment variable named `MISTRAL_API_KEY`.

   For bash:

   ```bash
   export MISTRAL_API_KEY=your_actual_api_key_here
   ```

   For Windows (Command Prompt):

   ```bat
   set MISTRAL_API_KEY=your_actual_api_key_here
   ```

### Usage

Run the script from the terminal:

```bash
python ocr.py --pdf /path/to/your/file.pdf --output /path/to/output_directory
```

- `--pdf`: Path to the PDF file to process
- `--output`: Base directory for output files

### Project Structure

```
├── ocr.py         # Main script for OCR processing
├── README.md      # This README file
└── .env           # Environment variable file (contains MISTRAL_API_KEY; do not publish publicly)
```

### License

This project is licensed under the MIT License. Please adhere to the license terms of the dependent libraries and the Mistral OCR API.

### Disclaimer

This tool uses the Mistral OCR API. Be aware of any rate limits or billing constraints associated with the API. For more details, see the [Mistral API documentation](https://docs.mistral.ai/api/).

---

## 日本語

### 必要条件

- Python 3
- [mistralai](https://pypi.org/project/mistralai/)

以下のコマンドで必要なパッケージをインストールします:

```bash
pip install mistralai
```

### セットアップ

1. **リポジトリのクローンまたはファイルの配置**

   下記のようなディレクトリ構成にしてください（例）:

   ```
   ├── ocr.py
   ├── README.md
   └── (.env)  # dotenv を使用する場合のみ（本設定では不要）
   ```

2. **環境変数の設定**

   Mistral APIキーを環境変数 `MISTRAL_API_KEY` に設定してください。

   bashの場合:

   ```bash
   export MISTRAL_API_KEY=your_actual_api_key_here
   ```

   Windowsの場合（コマンドプロンプト）:

   ```bat
   set MISTRAL_API_KEY=your_actual_api_key_here
   ```

### 使い方

ターミナルでスクリプトがあるディレクトリに移動し、以下のように実行します:

```bash
python ocr.py --pdf /path/to/your/file.pdf --output /path/to/output_directory
```

- `--pdf`: 処理するPDFファイルのパス
- `--output`: 出力ファイルの基底ディレクトリ

### プロジェクト構成

```
├── ocr.py         # OCR処理を行うメインスクリプト
├── README.md      # このREADMEファイル
└── .env           # 環境変数ファイル
```

### ライセンス

このプロジェクトはMITライセンスの下で公開されています。依存しているライブラリやMistral OCR APIの利用規約にも準拠してください。

### 免責事項

本ツールはMistral OCR APIを利用しています。APIの利用にあたっては、レートリミットや課金などの制約がある場合があります。詳細は[Mistral APIドキュメント](https://docs.mistral.ai/api/)をご確認ください。
