import os
import sys
import logging
import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pineconeの設定
DEFAULT_INDEX_NAME = "markdown-knowledge"
DEFAULT_NAMESPACE = "markdown-docs"

# リトライの設定
MAX_RETRIES = 5  # 最大リトライ回数
INITIAL_RETRY_DELAY = 1  # 初期リトライ間隔（秒）
MAX_RETRY_DELAY = 60  # 最大リトライ間隔（秒）

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
                model="text-embedding-ada-002",
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
    """
    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY
    
    while retry_count < MAX_RETRIES:
        try:
            # OpenAIクライアントの初期化
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # コンテキストを作成
            context = ""
            for i, doc in enumerate(documents, 1):
                context += f"\n\nドキュメント {i}:\n"
                context += f"タイトル: {doc['doc_title']}\n"
                context += f"セクション: {doc['section_title']}\n"
                context += f"内容: {doc['text']}\n"
            
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
            
            # ChatGPTで回答を生成
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは質問に対して、提供されたコンテキストのみに基づいて回答する優秀なアシスタントです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
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
                    logger.error(f"最大リトライ回数に達しました。回答生成を中止します: {e}")
                    return f"回答生成中にエラーが発生しました: レート制限に達しました。しばらく時間をおいてから再試行してください。"
                
                # 指数バックオフ + ジッタを使用して待機時間を計算
                jitter = random.uniform(0, 0.1 * retry_delay)
                sleep_time = min(retry_delay + jitter, MAX_RETRY_DELAY)
                
                logger.warning(f"レート制限に達しました。{sleep_time:.2f}秒後にリトライします（リトライ {retry_count}/{MAX_RETRIES}）")
                time.sleep(sleep_time)
                
                # 次のリトライの待機時間を2倍に増やす（指数バックオフ）
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
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