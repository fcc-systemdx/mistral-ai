import os
import sys
import logging
from pathlib import Path
import argparse
import re
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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

def sanitize_filename(filename: str, max_length: int = 40) -> str:
    """
    ファイル名を安全な形式に変換し、最大長を制限します。
    """
    # 特殊文字を除去し、スペースをアンダースコアに置換
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    
    # 最大長を制限
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename

def split_markdown_by_h1(content: str) -> list[tuple[str, str]]:
    """
    マークダウンの内容を見出し1（#）で分割します。
    各セクションのタイトルと内容のタプルのリストを返します。
    """
    # 見出し1のパターン
    h1_pattern = r'^#\s+(.+)$'
    
    # ファイルを分割
    sections = []
    current_title = "イントロダクション"  # デフォルトのタイトル
    current_content = []
    
    for line in content.split('\n'):
        if line.startswith('# '):
            # 前のセクションを保存
            if current_content:
                sections.append((current_title, '\n'.join(current_content)))
            # 新しいセクションを開始
            current_title = line[2:].strip()
            current_content = []
        else:
            current_content.append(line)
    
    # 最後のセクションを保存
    if current_content:
        sections.append((current_title, '\n'.join(current_content)))
    
    return sections

def add_related_files_context(section_title: str, all_sections: list[tuple[str, str]], section_index: int, doc_title: str) -> str:
    """
    関連ファイル情報を含むコンテキストを生成します。
    """
    related_info = [
        "# 関連ファイル",
        f"- タイトル: {doc_title}"
    ]
    
    # 前の4つのセクションを追加（存在する場合）
    for i in range(max(0, section_index - 4), section_index):
        prev_title = all_sections[i][0]
        related_info.append(f"- {4 - (section_index - i)}つ前に出力したファイル: {prev_title}")
    
    # 次のセクションを追加（存在する場合）
    if section_index + 1 < len(all_sections):
        next_title = all_sections[section_index + 1][0]
        related_info.append(f"- 次に出力するファイル: {next_title}")
    
    return "\n".join(related_info)

def process_markdown_files(in_dir: Path, out_dir: Path) -> None:
    """
    入力ディレクトリ内のマークダウンファイルを処理し、
    見出し1ごとに分割して出力ディレクトリに保存します。
    """
    # タイムスタンプ付きの出力ディレクトリを作成
    timestamp_dir = create_output_directory(out_dir)
    logging.info(f"出力ディレクトリを作成しました: {timestamp_dir}")
    
    # 入力ディレクトリ内のマークダウンファイルを取得
    md_files = list(in_dir.glob("*.md"))
    if not md_files:
        logging.warning(f"マークダウンファイルが見つかりません: {in_dir}")
        return
    
    logging.info(f"処理するマークダウンファイル数: {len(md_files)}")
    
    # 各マークダウンファイルを処理
    for md_path in md_files:
        logging.info(f"処理中: {md_path.name}")
        try:
            # マークダウンファイルの内容を読み込む
            content = md_path.read_text(encoding='utf-8')
            
            # 入力ファイル名をドキュメントタイトルとして使用
            doc_title = md_path.stem
            
            # 出力用のサブディレクトリを作成
            doc_output_dir = timestamp_dir / sanitize_filename(doc_title)
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 見出し1で分割
            sections = split_markdown_by_h1(content)
            
            # 各セクションを個別のファイルとして保存
            for i, (section_title, section_content) in enumerate(sections):
                # 関連ファイル情報を追加
                context = add_related_files_context(section_title, sections, i, doc_title)
                
                # 元のコンテンツの前に関連ファイル情報を追加
                enhanced_content = f"{context}\n\n# {section_title}\n{section_content}"
                
                # ファイル名を生成
                filename = f"{sanitize_filename(doc_title)}_{sanitize_filename(section_title)}.md"
                output_path = doc_output_dir / filename
                
                # ファイルに保存
                output_path.write_text(enhanced_content, encoding='utf-8')
                logging.info(f"保存しました: {output_path}")
            
            logging.info(f"{md_path.name}の処理が完了しました")
            
        except Exception as e:
            logging.error(f"{md_path.name}の処理中にエラーが発生しました: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='マークダウンファイルを分割するツール')
    parser.add_argument('-i', '--input_dir', type=Path, default=Path('in_markdowns'),
                        help='処理するマークダウンファイルが含まれるディレクトリ (デフォルト: in_markdowns)')
    parser.add_argument('-o', '--output_dir', type=Path, default=Path('out_md'),
                        help='出力ファイルのベースディレクトリ (デフォルト: out_md)')
    args = parser.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir

    # ディレクトリの存在確認
    if not in_dir.exists():
        logging.error(f"入力ディレクトリが存在しません: {in_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f'入力ディレクトリ: {in_dir}')
    logging.info(f'出力ディレクトリ: {out_dir}')

    process_markdown_files(in_dir, out_dir)

if __name__ == '__main__':
    main()
