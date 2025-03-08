# PDFからMarkdownの変換サンプルコード（Mistral AI OCR API）

## 注意事項
- Mistral APIキーとAPIクレジットは各自で用意してください

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
   ├── .env
   ```

2. **環境変数の設定**

   プロジェクトのルートディレクトリに `.env` ファイルを作成し、Mistral APIキーを設定します:

   ```env
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

   **注意:** `.env` ファイルには機密情報が含まれているため、公開リポジトリに含めないようにしてください。

## 使い方

ターミナルでスクリプトがあるディレクトリに移動し、次のように実行します:

- **PDFファイルや出力先を変更する場合**は、以下のようにコマンドライン引数を利用します:

  ```bash
  python ocr.py --pdf /path/to/your/file.pdf --output ./
  ```

  - **--pdf:** 処理するPDFファイルのパス
  - **--output:** 出力ファイルの基底ディレクトリ（デフォルト: スクリプトのディレクトリ）

実行すると、以下の処理が行われます:

- 指定した出力ディレクトリ（タイムスタンプ付きのフォルダ）が作成されます。
- PDFファイルがアップロードされ、OCR処理が実施されます。
- OCRレスポンスがJSON形式で保存され、抽出されたテキストがテキストファイルとして保存されます。
- インライン画像付きのMarkdownファイルが生成されます。

## プロジェクト構成

```
├── ocr.py         # OCR処理を行うメインスクリプト
├── README.md      # このREADMEファイル
├── .env           # 環境変数ファイル（MISTRAL_API_KEY を含む）※公開しないこと
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。また、依存しているライブラリやMistral OCR APIの利用規約にも準拠してください。


## 免責事項

本ツールはMistral OCR APIを利用しています。APIの利用にあたっては、レートリミットや課金などの制約がある場合があります。詳細は[Mistral APIドキュメント](https://docs.mistral.ai/api/)をご確認ください。
