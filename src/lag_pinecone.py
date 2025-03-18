import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import datetime
import re
import time
import random
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# ベクトル化のためのライブラリをインポート
# OpenAIのEmbeddingsを使用する場合
from openai import OpenAI

# 他の埋め込みモデルを使用する場合はコメントを外してください
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pineconeの設定
DEFAULT_INDEX_NAME = "db-test"
DEFAULT_NAMESPACE = "markdown-docs"

# リトライの設定
MAX_RETRIES = 5  # 最大リトライ回数
INITIAL_RETRY_DELAY = 1  # 初期リトライ間隔（秒）
MAX_RETRY_DELAY = 60  # 最大リトライ間隔（秒）

def create_pinecone_index(index_name: str, dimension: int = 3072) -> None:
    """
    Pineconeのインデックスを作成します。
    dimension: 使用する埋め込みモデルの次元数（OpenAI ada-002は1536次元）
    """
    try:
        # Pineconeクライアントの初期化
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 既存のインデックスを確認
        if index_name in [index.name for index in pc.list_indexes()]:
            logger.info(f"インデックス '{index_name}' は既に存在します。")
            return
        
        # インデックスの作成
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        logger.info(f"インデックス '{index_name}' を作成しました。")
    
    except Exception as e:
        logger.error(f"Pineconeインデックスの作成中にエラーが発生しました: {e}")
        sys.exit(1)

def get_embedding_with_retry(text: str, client: OpenAI) -> List[float]:
    """
    テキストをOpenAIのモデルを使用してベクトル化します。
    レート制限に達した場合は指数バックオフでリトライします。
    """
    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY
    
    while retry_count < MAX_RETRIES:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            # レート制限エラーの検出（エラーメッセージやステータスコードで判断）
            is_rate_limit_error = (
                "rate limit" in str(e).lower() or 
                "too many requests" in str(e).lower() or
                "429" in str(e)
            )
            
            if is_rate_limit_error:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.error(f"最大リトライ回数に達しました。埋め込み生成を中止します: {e}")
                    return []
                
                # 指数バックオフ + ジッタを使用して待機時間を計算
                jitter = random.uniform(0, 0.1 * retry_delay)
                sleep_time = min(retry_delay + jitter, MAX_RETRY_DELAY)
                
                logger.warning(f"レート制限に達しました。{sleep_time:.2f}秒後にリトライします（リトライ {retry_count}/{MAX_RETRIES}）")
                time.sleep(sleep_time)
                
                # 次のリトライの待機時間を2倍に増やす（指数バックオフ）
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                logger.error(f"埋め込み生成中にエラーが発生しました: {e}")
                return []
    
    return []

def chunk_markdown(markdown_text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    マークダウンテキストを適切なサイズのチャンクに分割します。
    チャンクサイズはトークン数ではなく文字数で近似しています。
    """
    if len(markdown_text) <= chunk_size:
        return [markdown_text]
    
    chunks = []
    start = 0
    
    while start < len(markdown_text):
        # チャンクの終了位置を計算
        end = start + chunk_size
        
        # 文の途中で切らないように調整
        if end < len(markdown_text):
            # 次のピリオド、疑問符、感嘆符、改行を探す
            next_period = markdown_text.find('. ', end)
            next_question = markdown_text.find('? ', end)
            next_exclamation = markdown_text.find('! ', end)
            next_newline = markdown_text.find('\n', end)
            
            # 最も近い区切り文字を見つける
            possible_ends = [pos for pos in [next_period, next_question, next_exclamation, next_newline] if pos != -1]
            
            if possible_ends:
                end = min(possible_ends) + 2  # ピリオドとスペースを含める
            else:
                # 見つからない場合はそのままのendを使用
                pass
        
        # チャンクを追加
        chunks.append(markdown_text[start:end])
        
        # 次のチャンクの開始位置を設定（オーバーラップを考慮）
        start = end - chunk_overlap
    
    return chunks

def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """
    ファイルパスからメタデータを抽出します。
    """
    # ファイル名からドキュメントタイトルとセクションタイトルを抽出
    file_name = file_path.stem  # 拡張子を除いたファイル名
    
    # ファイル名がドキュメント名_セクション名の形式であると仮定
    parts = file_name.split('_', 1)
    
    doc_title = parts[0] if len(parts) > 0 else ""
    section_title = parts[1] if len(parts) > 1 else ""
    
    # ファイルの内容から関連ファイル情報を抽出
    content = file_path.read_text(encoding='utf-8')
    
    # 関連ファイル情報を抽出するための正規表現
    related_files = []
    for line in content.split('\n'):
        if line.startswith('- ') and '出力したファイル:' in line:
            match = re.search(r'出力したファイル: (.+)', line)
            if match:
                related_files.append(match.group(1))
        elif line.startswith('- 次に出力するファイル:'):
            match = re.search(r'次に出力するファイル: (.+)', line)
            if match:
                related_files.append(match.group(1))
    
    return {
        "doc_title": doc_title,
        "section_title": section_title,
        "file_name": file_name,
        "file_path": str(file_path),
        "related_files": related_files
    }

def process_markdown_files(input_dir: Path, index_name: str, namespace: str, chunk_size: int, chunk_overlap: int) -> None:
    """
    マークダウンファイルを処理し、Pineconeに保存します。
    """
    try:
        # APIキーが設定されているか確認
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEYが環境変数に設定されていません。")
            sys.exit(1)
        
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEYが環境変数に設定されていません。")
            sys.exit(1)
        
        # Pineconeクライアントの初期化
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # インデックスが存在するか確認し、なければ作成
        if index_name not in [index.name for index in pc.list_indexes()]:
            create_pinecone_index(index_name)
        
        # インデックスに接続
        index = pc.Index(index_name)
        
        # OpenAIクライアントの初期化
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # マークダウンファイルのリストを取得（再帰的に検索）
        md_files = list(input_dir.glob("**/*.md"))
        
        if not md_files:
            logger.warning(f"マークダウンファイルが見つかりません: {input_dir}")
            return
        
        logger.info(f"処理するマークダウンファイル数: {len(md_files)}")
        
        # バッチでアップロード用のベクトルを格納するリスト
        vectors_to_upsert = []
        batch_size = 100  # 一度にアップロードするベクトルの数
        
        # 処理したベクトルの総数
        total_vectors = 0
        
        # 各マークダウンファイルを処理
        for md_path in tqdm(md_files, desc="ファイル処理中"):
            try:
                # ファイルからメタデータを抽出
                metadata = extract_metadata(md_path)
                
                # ファイルの内容を読み込む
                content = md_path.read_text(encoding='utf-8')
                
                # 内容をチャンクに分割
                chunks = chunk_markdown(content, chunk_size, chunk_overlap)
                
                # 各チャンクを処理
                for i, chunk in enumerate(chunks):
                    # チャンクをベクトル化（リトライロジックつき）
                    embedding = get_embedding_with_retry(chunk, openai_client)
                    
                    if not embedding:
                        logger.warning(f"チャンクのベクトル化に失敗しました: {md_path.name}, チャンク {i+1}/{len(chunks)}")
                        continue
                    
                    # チャンク固有のメタデータを作成
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["chunk_total"] = len(chunks)
                    chunk_metadata["text"] = chunk[:1000]  # テキストの一部を保存（Pineconeの制限あり）
                    
                    # ベクトルをリストに追加
                    vector_id = f"{md_path.name}_{i}"
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": chunk_metadata
                    })
                    
                    # バッチサイズに達したらアップロード
                    if len(vectors_to_upsert) >= batch_size:
                        index.upsert(
                            vectors=vectors_to_upsert,
                            namespace=namespace
                        )
                        logger.info(f"{len(vectors_to_upsert)}件のベクトルをネームスペース '{namespace}' にアップロードしました")
                        total_vectors += len(vectors_to_upsert)
                        vectors_to_upsert = []
                
                logger.info(f"処理完了: {md_path.name}")
                
            except Exception as e:
                logger.error(f"{md_path.name}の処理中にエラーが発生しました: {e}")
                continue
        
        # 残りのベクトルをアップロード
        if vectors_to_upsert:
            index.upsert(
                vectors=vectors_to_upsert,
                namespace=namespace
            )
            total_vectors += len(vectors_to_upsert)
            logger.info(f"残り{len(vectors_to_upsert)}件のベクトルをネームスペース '{namespace}' にアップロードしました")
        
        logger.info(f"すべてのマークダウンファイルの処理が完了しました。総ベクトル数: {total_vectors}")
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        sys.exit(1)

def test_query(index_name: str, namespace: str, query: str, top_k: int = 5) -> None:
    """
    クエリをテストします。
    """
    try:
        # APIキーが設定されているか確認
        if not PINECONE_API_KEY or not OPENAI_API_KEY:
            logger.error("APIキーが設定されていません。")
            return
        
        # Pineconeクライアントの初期化
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # インデックスが存在するか確認
        if index_name not in [index.name for index in pc.list_indexes()]:
            logger.error(f"インデックス '{index_name}' が存在しません。")
            return
        
        # インデックスに接続
        index = pc.Index(index_name)
        
        # OpenAIクライアントの初期化
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # クエリをベクトル化（リトライロジックつき）
        query_embedding = get_embedding_with_retry(query, openai_client)
        
        if not query_embedding:
            logger.error("クエリのベクトル化に失敗しました。")
            return
        
        # ベクトル検索を実行
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        
        # 結果を表示
        logger.info(f"クエリ: {query}")
        logger.info(f"ネームスペース: {namespace}")
        logger.info("検索結果:")
        
        for i, match in enumerate(results.matches, 1):
            logger.info(f"\n結果 {i}: スコア {match.score:.4f}")
            logger.info(f"ドキュメント: {match.metadata.get('doc_title')}")
            logger.info(f"セクション: {match.metadata.get('section_title')}")
            logger.info(f"ファイル: {match.metadata.get('file_name')}")
            logger.info(f"テキスト抜粋: {match.metadata.get('text')[:200]}...")
        
    except Exception as e:
        logger.error(f"クエリテスト中にエラーが発生しました: {e}")

def main():
    parser = argparse.ArgumentParser(description='マークダウンファイルをPineconeにインデックス化するツール')
    parser.add_argument('-i', '--input_dir', type=Path, required=True,
                        help='処理するマークダウンファイルが含まれるディレクトリ')
    parser.add_argument('-n', '--index_name', type=str, default=DEFAULT_INDEX_NAME,
                        help=f'Pineconeのインデックス名 (デフォルト: {DEFAULT_INDEX_NAME})')
    parser.add_argument('-ns', '--namespace', type=str, default=DEFAULT_NAMESPACE,
                        help=f'Pineconeのネームスペース名 (デフォルト: {DEFAULT_NAMESPACE})')
    parser.add_argument('-c', '--chunk_size', type=int, default=512,
                        help='チャンクサイズ（文字数、デフォルト: 512）')
    parser.add_argument('-o', '--chunk_overlap', type=int, default=50,
                        help='チャンクオーバーラップ（文字数、デフォルト: 50）')
    parser.add_argument('-t', '--test', action='store_true',
                        help='インデックス作成後にテストクエリを実行')
    parser.add_argument('-q', '--query', type=str,
                        help='テストクエリ（--testオプションと共に使用）')
    parser.add_argument('--only_test', action='store_true',
                        help='インデックス作成をスキップしてテストのみ実行')
    
    args = parser.parse_args()
    
    if args.only_test:
        if not args.query:
            parser.error("--only_testオプションを使用する場合は--queryも指定してください")
        test_query(args.index_name, args.namespace, args.query)
        return
    
    # 入力ディレクトリの存在確認
    if not args.input_dir.exists():
        logger.error(f"入力ディレクトリが存在しません: {args.input_dir}")
        sys.exit(1)
    
    logger.info(f'入力ディレクトリ: {args.input_dir}')
    logger.info(f'Pineconeインデックス名: {args.index_name}')
    logger.info(f'Pineconeネームスペース: {args.namespace}')
    logger.info(f'チャンクサイズ: {args.chunk_size}文字')
    logger.info(f'チャンクオーバーラップ: {args.chunk_overlap}文字')
    
    # マークダウンファイルを処理して、Pineconeにインデックス化
    process_markdown_files(args.input_dir, args.index_name, args.namespace, args.chunk_size, args.chunk_overlap)
    
    # テストクエリを実行
    if args.test:
        query = args.query or "このドキュメントの主な内容は何ですか？"
        logger.info("\nテストクエリを実行します...")
        test_query(args.index_name, args.namespace, query)

if __name__ == '__main__':
    main() 