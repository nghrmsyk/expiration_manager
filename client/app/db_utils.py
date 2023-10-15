import sqlite3
import os
# データベース接続とテーブル作成
class DatabaseManager:
    """商品データのデータベースを管理するクラス

    Attributes:
        image_dir (str): 画像を保存するディレクトリのパス
        db_path (str): データベースのパス
    """
        
    def __init__(self):
        """初期化メソッド
        
        画像ディレクトリとデータベースのパスを設定する
        """
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(file_dir, "DB", "images")
        self.db_path = os.path.join(file_dir, "DB", "product.db")

    def connect(self):
        """データベースに接続するメソッド
        
        接続がない場合は新しく作成する

        Returns:
            sqlite3.Connection: データベース接続オブジェクト
        """
        #DBを配置するフォルダと画像を保存するフォルダ作成
        os.makedirs(self.image_dir, exist_ok=True)
        #DB接続・なければ作成
        return sqlite3.connect(self.db_path)

    def create(self):
        """商品テーブルを作成するメソッド
        
        すでに商品テーブルが存在する場合は何もしない
        """
        sql = '''CREATE TABLE IF NOT EXISTS product( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name VARCHAR(50),
                item_name VARCHAR(50),
                expiry_type CHAR(4),
                expiry_date 
                )'''
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute(sql)
        conn.commit()
        conn.close()

    def insert(self,user_name, item_name, expiry_type, expiry_date):
        """商品データをデータベースに挿入するメソッド

        Args:
            item_name (str): 商品名
            expiry_type (str): 期限の種類
            expiry_date (str): 期限の日付

        Returns:
            int: 新しく追加されたデータのID
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO product (user_name, item_name, expiry_type, expiry_date)
                        VALUES (?, ?, ?, ?)''', (user_name, item_name, expiry_type, expiry_date))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return new_id
    
    def fetch_all_products(self, user_name):
        """すべての商品データを期限の昇順で取得するメソッド

        Args:
            user_name(str): ユーザ名

        Returns:
            list: 商品データのリスト
        """
        conn = self.connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM product WHERE user_name = ? ORDER BY expiry_date", (user_name,))
        table = cursor.fetchall()
        conn.close()
            
        return table
    
    def delete(self, id):
        """指定されたIDの商品データをデータベースから削除するメソッド

        Args:
            id (int): 削除する商品データのID
        """
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute("DELETE FROM product WHERE id=?", (id,))
       
        conn.commit()
        conn.close()
        # 関連する切り出し画像を削除
        image_path = os.path.join(self.image_dir, f"{id}.png")
        if os.path.exists(image_path):
            os.remove(image_path)

class UserManager():
    def __init__(self):
        """初期化メソッド
        
        データベースのパスを設定する
        """
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(file_dir, "DB")
        self.db_path = os.path.join(file_dir, "DB", "product.db")

    def connect(self):
        """データベースに接続するメソッド
        
        接続がない場合は新しく作成する

        Returns:
            sqlite3.Connection: データベース接続オブジェクト
        """
        #DBを配置するフォルダと画像を保存するフォルダ作成
        os.makedirs(self.image_dir, exist_ok=True)
        #DB接続・なければ作成
        return sqlite3.connect(self.db_path)

    def create(self, make_init_user = True):
        """ユーザテーブルを作成するメソッド
        
        すでにテーブルが存在する場合は何もしない
        """
        sql = '''CREATE TABLE IF NOT EXISTS users( 
                name VARCHAR(50)
                )'''
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute(sql)
        conn.commit()
        conn.close()

        #ユーザが登録されていなければguestユーザを作成
        if make_init_user and not len(self.get_users()):
            self.register("guest")

    def get_users(self):
        """すべてのユーザ情報を取得するメソッド

        Returns:
            list: ユーザ名のリスト
        """
        conn = self.connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users")
        table = cursor.fetchall()
        conn.close()

        return table

    def register(self,name):
        """新規にユーザ情報を追加するメソッド

        Args:
            item_name (name): ユーザ名

        Returns:
            int: 新しく追加されたデータのID
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return new_id

    def delete(self, name):
        """指定されたユーザを削除するメソッド

        Args:
            id (int): 削除する商品データのID
        """
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute("DELETE FROM users WHERE name=?", (name,))
        conn.commit()
        conn.close()