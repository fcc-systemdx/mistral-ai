import os
import sys
import logging
import argparse
import json
import time
import random
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import anthropic

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Pineconeの設定
DEFAULT_INDEX_NAME = "markdown-knowledge"
DEFAULT_NAMESPACE = "markdown-docs"

# リトライの設定
MAX_RETRIES = 5  # 最大リトライ回数
INITIAL_RETRY_DELAY = 1  # 初期リトライ間隔（秒）
MAX_RETRY_DELAY = 60  # 最大リトライ間隔（秒）

def sanitize_for_pinecone_id(text: str) -> str:
    """
    Pineconeのベクトルキーとして使用できる文字列に変換します。
    非ASCII文字を含む名前を安全なASCII文字列に変換します。
    """
    # 非ASCII文字をASCII文字に変換または削除
    # 方法1: 非ASCII文字をハッシュ値に置き換え
    if any(ord(c) > 127 for c in text):
        # 元の文字列のハッシュ値を計算し、十分に短い一意のIDを生成
        hashed = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
        
        # ASCII文字のみ抽出し、非ASCII文字はハッシュ値で置き換え
        ascii_part = ''.join(c for c in text if ord(c) < 128 and c.isalnum() or c in '_-.')
        ascii_part = ascii_part[:20]  # 長さを制限
        
        # 空の場合はハッシュ値のみを使用
        if not ascii_part:
            return f"doc_{hashed}"
        
        return f"{ascii_part}_{hashed}"
    
    # 特殊文字を置き換え (ASCII範囲内の特殊文字)
    safe_text = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in text)
    
    # 空文字列になった場合のデフォルト値
    if not safe_text:
        safe_text = "document"
    
    return safe_text

def get_embedding_with_retry(text: str, client: OpenAI) -> List[float]:
    """
    テキストをOpenAIのモデルを使用してベクトル化します。
    レート制限に達した場合は指数バックオフでリトライします。
    トークン数が多すぎる場合は自動的に分割して平均化します。
    """
    # テキストが空または短すぎる場合
    if not text or len(text) < 10:
        logger.warning("テキストが空または短すぎます。埋め込み生成をスキップします。")
        return []
    
    # 長いテキストの処理
    MAX_TOKENS = 7000  # 安全マージンを考慮して最大トークン数を設定
    
    # テキストの長さを概算（英語のトークン数は文字数の約1/4、日本語はより多い）
    # 安全のために保守的に見積もる
    if len(text) > MAX_TOKENS * 3:  # 日本語を考慮して3で割る
        # テキストを分割
        logger.info(f"テキストが長すぎるため分割します（文字数: {len(text)}）")
        
        # 単純な分割方法（意味的な区切りを考慮する場合は改良が必要）
        # 段落または文で分割
        parts = []
        paragraphs = text.split('\n\n')
        
        current_part = ""
        for paragraph in paragraphs:
            # もし現在のパートに段落を追加しても長さ制限以内なら追加
            if len(current_part + paragraph) < MAX_TOKENS * 3:
                current_part += paragraph + "\n\n"
            else:
                # 現在のパートが空でなければ追加
                if current_part:
                    parts.append(current_part)
                
                # 新しいパートを開始
                # 段落自体が長すぎる場合は文単位で分割
                if len(paragraph) > MAX_TOKENS * 3:
                    sentences = re.split(r'(?<=[。.！!?？])', paragraph)
                    current_part = ""
                    for sentence in sentences:
                        if len(current_part + sentence) < MAX_TOKENS * 3:
                            current_part += sentence
                        else:
                            if current_part:
                                parts.append(current_part)
                            
                            # 文自体が長すぎる場合は強制的に分割
                            if len(sentence) > MAX_TOKENS * 3:
                                for i in range(0, len(sentence), MAX_TOKENS * 2):
                                    parts.append(sentence[i:i + MAX_TOKENS * 2])
                                current_part = ""
                            else:
                                current_part = sentence
                else:
                    current_part = paragraph + "\n\n"
        
        # 最後のパートを追加
        if current_part:
            parts.append(current_part)
        
        logger.info(f"テキストを {len(parts)} 個のパートに分割しました")
        
        # 各パートの埋め込みを取得
        embeddings = []
        for i, part in enumerate(parts):
            logger.info(f"パート {i+1}/{len(parts)} の埋め込みを生成中...")
            part_embedding = _get_single_embedding(part, client)
            if part_embedding:
                embeddings.append(part_embedding)
        
        # 埋め込みが取得できなかった場合
        if not embeddings:
            logger.error("すべてのパートの埋め込み生成に失敗しました。")
            return []
        
        # 埋め込みベクトルの平均を計算
        avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
        
        # 正規化（長さを1に）
        magnitude = sum(x**2 for x in avg_embedding) ** 0.5
        if magnitude > 0:
            normalized_embedding = [x / magnitude for x in avg_embedding]
            return normalized_embedding
        else:
            logger.error("埋め込みベクトルの正規化に失敗しました。")
            return []
    
    # 通常の長さのテキストの処理
    return _get_single_embedding(text, client)

def _get_single_embedding(text: str, client: OpenAI) -> List[float]:
    """
    単一のテキストセグメントの埋め込みを取得します（リトライロジック付き）
    """
    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY
    
    # テキストの長さを制限（モデルの制限内に収める）
    if len(text) > 20000:  # 非常に長いテキストの場合は切り詰める
        text = text[:20000]
        logger.warning("テキストが非常に長いため切り詰めました。")
    
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
            
            # トークン数制限エラーの検出
            is_token_limit_error = (
                "maximum context length" in str(e).lower() or
                "reduce your prompt" in str(e).lower() or
                "tokens" in str(e).lower() and "requested" in str(e).lower()
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
            elif is_token_limit_error:
                # テキストを半分に切って再試行
                half_length = len(text) // 2
                if half_length < 100:  # テキストが既に非常に短い場合
                    logger.error(f"テキストが短いにもかかわらずトークン制限エラーが発生しました: {e}")
                    return []
                
                logger.warning(f"トークン制限エラーが発生しました。テキストを短くして再試行します。")
                text = text[:half_length]
            else:
                logger.error(f"埋め込み生成中にエラーが発生しました: {e}")
                return []
    
    return []

def search_documents(query: str, index_name: str, namespace: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict[str, Any]]:
    """
    クエリに関連する文書を検索します。
    """
    try:
        # APIキーが設定されているか確認
        if not PINECONE_API_KEY or not OPENAI_API_KEY:
            logger.error("APIキーが設定されていません。")
            return []
        
        # Pineconeクライアントの初期化
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # インデックスが存在するか確認
        if index_name not in [index.name for index in pc.list_indexes()]:
            logger.error(f"インデックス '{index_name}' が存在しません。")
            return []
        
        # インデックスに接続
        index = pc.Index(index_name)
        
        # OpenAIクライアントの初期化
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # クエリをベクトル化（リトライロジックつき）
        query_embedding = get_embedding_with_retry(query, openai_client)
        
        if not query_embedding:
            logger.error("クエリのベクトル化に失敗しました。")
            return []
        
        # クエリパラメータの設定
        query_params = {
            "namespace": namespace,
            "vector": query_embedding,
            "top_k": top_k,
            "include_values": True,
            "include_metadata": True
        }
        
        # フィルタが指定されている場合は追加
        if filter_dict:
            query_params["filter"] = filter_dict
        
        # ベクトル検索を実行
        results = index.query(**query_params)
        
        # 検索結果を整形
        documents = []
        for match in results.matches:
            documents.append({
                "score": match.score,
                "doc_title": match.metadata.get("doc_title", ""),
                "section_title": match.metadata.get("section_title", ""),
                "text": match.metadata.get("text", ""),
                "file_name": match.metadata.get("file_name", ""),
                "file_path": match.metadata.get("file_path", "")
            })
        
        return documents
        
    except Exception as e:
        logger.error(f"文書検索中にエラーが発生しました: {e}")
        return []

def generate_answer_with_retry(query: str, documents: List[Dict[str, Any]]) -> str:
    """
    検索結果を使用して回答を生成します。レート制限に達した場合はリトライします。
    トークン数制限にも対応します。
    """
    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY
    
    # ドキュメントの情報をコンテキストに追加（トークン数を考慮）
    context = ""
    MAX_CONTEXT_LENGTH = 15000  # 安全マージンを考慮
    
    # すべてのドキュメントを追加すると長すぎる場合があるので、制限する
    for i, doc in enumerate(documents, 1):
        doc_text = f"\n\nドキュメント {i}:\n"
        doc_text += f"タイトル: {doc['doc_title']}\n"
        doc_text += f"セクション: {doc['section_title']}\n"
        doc_text += f"内容: {doc['text']}\n"
        
        # コンテキストが長くなりすぎる場合は、ここでループを抜ける
        if len(context + doc_text) > MAX_CONTEXT_LENGTH:
            logger.warning(f"コンテキストが長すぎるため、{i-1}個のドキュメントのみを使用します（残り{len(documents)-i+1}個はスキップ）")
            break
        
        context += doc_text
    
    while retry_count < MAX_RETRIES:
        try:
            # Anthropicクライアントの初期化
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            # プロンプトを作成
            prompt = f"""
以下のコンテキストに基づいて、質問に答えてください。
質問に関連する情報がコンテキストに含まれていない場合は、「この質問に答えるための情報がコンテキストにありません」と答えてください。
回答はコンテキストに基づくものだけにしてください。自分の知識に基づく追加情報は提供しないでください。

コンテキスト:
{context}

質問: {query}

回答:
"""
            
            # Claude Sonnet 3.7で回答を生成
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.content[0].text
        
        except Exception as e:
            # レート制限エラーの検出（エラーメッセージやステータスコードで判断）
            is_rate_limit_error = (
                "rate limit" in str(e).lower() or 
                "too many requests" in str(e).lower() or
                "429" in str(e)
            )
            
            # トークン数制限エラーの検出
            is_token_limit_error = (
                "maximum context length" in str(e).lower() or
                "reduce your prompt" in str(e).lower() or
                "tokens" in str(e).lower() and "requested" in str(e).lower()
            )
            
            if is_rate_limit_error:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.error(f"最大リトライ回数に達しました。回答生成を中止します: {e}")
                    return f"回答生成中にエラーが発生しました: レート制限に達しました。しばらく時間をおいてから再試行してください。"
                
                # 指数バックオフ + ジッタを使用して待機時間を計算
                jitter = random.uniform(0, 0.1 * retry_delay)
                sleep_time = min(retry_delay + jitter, MAX_RETRY_DELAY)
                
                logger.warning(f"レート制限に達しました。{sleep_time:.2f}秒後にリトライします（リトライ {retry_count}/{MAX_RETRIES}）")
                time.sleep(sleep_time)
                
                # 次のリトライの待機時間を2倍に増やす（指数バックオフ）
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            elif is_token_limit_error:
                # コンテキストを短くして再試行
                context_length = len(context)
                if context_length < 1000:  # コンテキストが既に十分短い場合
                    logger.error(f"コンテキストが短いにもかかわらずトークン制限エラーが発生しました: {e}")
                    return f"回答生成中にエラーが発生しました: 入力トークン数が多すぎます。クエリを短くするか、検索結果数を減らしてください。"
                
                # コンテキストを半分に削減
                logger.warning(f"トークン制限エラーが発生しました。コンテキストを短くして再試行します。")
                context = context[:context_length//2]
            else:
                logger.error(f"回答生成中にエラーが発生しました: {e}")
                return f"回答生成中にエラーが発生しました: {e}"
    
    return "回答の生成に失敗しました。しばらく時間をおいてから再試行してください。"

def interactive_qa(index_name: str, namespace: str, filter_dict: Dict = None):
    """
    対話形式のQ&Aセッションを実行します。
    """
    logger.info("対話型Q&Aセッションを開始します。'exit'または'quit'と入力すると終了します。")
    logger.info(f"インデックス: {index_name}, ネームスペース: {namespace}")
    
    if filter_dict:
        logger.info(f"フィルタ適用中: {filter_dict}")
    
    while True:
        try:
            # ユーザーからの入力を受け取る
            query = input("\n質問を入力してください: ")
            
            # 終了条件
            if query.lower() in ["exit", "quit", "終了"]:
                logger.info("Q&Aセッションを終了します。")
                break
            
            # 空の入力をスキップ
            if not query.strip():
                continue
            
            # 関連文書を検索
            logger.info("関連文書を検索中...")
            documents = search_documents(query, index_name, namespace, top_k=5, filter_dict=filter_dict)
            
            if not documents:
                logger.info("関連する文書が見つかりませんでした。")
                continue
            
            # 回答を生成
            logger.info("回答を生成中...")
            answer = generate_answer_with_retry(query, documents)
            
            # 結果を表示
            logger.info("\n回答:")
            print(f"\n{answer}\n")
            
            # 参照情報を表示
            logger.info("参照:")
            for i, doc in enumerate(documents, 1):
                logger.info(f"{i}. {doc['doc_title']} - {doc['section_title']} (スコア: {doc['score']:.4f})")
        
        except KeyboardInterrupt:
            logger.info("\nQ&Aセッションを終了します。")
            break
        
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pineconeに保存された知識を使った質問応答アプリケーション')
    parser.add_argument('-n', '--index_name', type=str, default=DEFAULT_INDEX_NAME,
                        help=f'Pineconeのインデックス名 (デフォルト: {DEFAULT_INDEX_NAME})')
    parser.add_argument('-ns', '--namespace', type=str, default=DEFAULT_NAMESPACE,
                        help=f'Pineconeのネームスペース名 (デフォルト: {DEFAULT_NAMESPACE})')
    parser.add_argument('-q', '--query', type=str,
                        help='一回限りの質問（指定しない場合は対話モードになります）')
    parser.add_argument('-k', '--top_k', type=int, default=5,
                        help='検索結果の上位件数 (デフォルト: 5)')
    parser.add_argument('-f', '--filter', type=str,
                        help='メタデータフィルタ（JSON形式の文字列）')
    parser.add_argument('-d', '--doc_title', type=str,
                        help='特定のドキュメントタイトルでフィルタリング')
    parser.add_argument('-r', '--rate_limit', type=int, default=5,
                        help='1分あたりのAPI呼び出し制限 (デフォルト: 5)')
    
    args = parser.parse_args()
    
    # APIキーが設定されているか確認
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEYが環境変数に設定されていません。")
        sys.exit(1)
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEYが環境変数に設定されていません。")
        sys.exit(1)
    
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEYが環境変数に設定されていません。")
        sys.exit(1)
    
    # インデックスが存在するか確認
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if args.index_name not in [index.name for index in pc.list_indexes()]:
        logger.error(f"インデックス '{args.index_name}' が存在しません。Pineconeにインデックスを作成してください。")
        sys.exit(1)
    
    # フィルタの設定
    filter_dict = None
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError:
            logger.error("フィルタのJSON形式が不正です。")
            sys.exit(1)
    elif args.doc_title:
        # ドキュメントタイトルでフィルタリング
        filter_dict = {"doc_title": {"$eq": args.doc_title}}
    
    # 単一の質問モード
    if args.query:
        logger.info(f"質問: {args.query}")
        logger.info(f"インデックス: {args.index_name}, ネームスペース: {args.namespace}")
        
        if filter_dict:
            logger.info(f"フィルタ適用中: {filter_dict}")
        
        # 関連文書を検索
        documents = search_documents(args.query, args.index_name, args.namespace, args.top_k, filter_dict)
        
        if not documents:
            logger.info("関連する文書が見つかりませんでした。")
            sys.exit(0)
        
        # 回答を生成
        answer = generate_answer_with_retry(args.query, documents)
        
        # 結果を表示
        logger.info("\n回答:")
        print(f"\n{answer}\n")
        
        # 参照情報を表示
        logger.info("参照:")
        for i, doc in enumerate(documents, 1):
            logger.info(f"{i}. {doc['doc_title']} - {doc['section_title']} (スコア: {doc['score']:.4f})")
    
    # 対話モード
    else:
        interactive_qa(args.index_name, args.namespace, filter_dict)

if __name__ == '__main__':
    main() 