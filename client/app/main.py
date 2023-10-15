from typing import Any
import streamlit as st
import pandas as pd
import sqlite3
import enum
from dataclasses import dataclass,field
import os
from PIL import Image
import PIL
import datetime
import copy
import requests
import json
import uuid

AVAILABLE_IMAGE_TYPE = ["jpg", "png", "jpeg"]
EXPIRY_TYPE_DICT = {"消費期限" : 0, "賞味期限" : 1}

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


class ImageUploader():
    """画像を指定されたサーバーにアップロードするクラス

    Attributes:
        server_url (str): アップロード先のサーバーのURL
        image (obj): アップロードする画像オブジェクト
    """
    def __init__(self, image):
        """初期化メソッド

        Args:
            image (obj): アップロードする画像オブジェクト
        """
        self.server_url = "http://172.30.0.3:8000/food-expiration/"
        self.image = image

    def get_content_type(self):
        """ファイル拡張子に基づいてMIMEタイプを取得するメソッド

        Returns:
            str: MIMEタイプ
        """
        if self.image.name.endswith(".png"):
            return "image/png"
        elif self.image.name.endswith(".jpg") or self.image.name.endswith(".jpeg"):
            return "image/jpeg"
        else:
            return "application/octet-stream"
        
    def upload(self):
        """画像をAPIにアップロードし、データを取得するメソッド

        Returns:
            dict: サーバーからの応答データ。エラーが発生した場合はNone
        """
        mime_type = self.get_content_type()
        files = {"file": (self.image.name, self.image.getvalue(), mime_type)}

        try:
            response = requests.post(self.server_url, files=files)
            response_data = response.json()
            return response_data
        except Exception as e:
            st.text("a")
            st.error(f"エラー: {str(e)}")
            return None

# 画像処理サーバへのリクエスト
class ImageProcessor:
    """画像処理を行うクラス

    Attributes:
        length (int): 画像の目標サイズ
        image (obj): 処理する画像オブジェクト
    """
    def __init__(self, image):
        """初期化メソッド

        Args:
            image (obj): 処理する画像オブジェクト
        """
        self.length = 150
        self.image = image

    def crop(self, xmin, ymin, xmax, ymax):
        """指定された座標とサイズに基づいて画像をクロッピングするメソッド

        Args:
            xmin (float): x座標の最小値
            ymin (float): y座標の最小値
            xmax (float): x座標の最大値
            ymax (float): x座標の最小値

        Returns:
            ImageProcessor: 自身のインスタンス
        """

        self.image = self.image.crop((xmin, ymin, xmax, ymax))
        return self

    def square(self):
        """画像を正方形にリサイズするメソッド

        Returns:
            ImageProcessor: 自身のインスタンス
        """
        width, height = self.image.size

        # 長い辺を基準に拡大・縮小の比率を計算
        if width > height:
            ratio = self.length / width
        else:
            ratio = self.length / height

        # アスペクト比を保持しながら画像をリサイズ
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        self.image = self.image.resize((new_width, new_height), Image.ANTIALIAS)

        # 正方形の背景を作成
        background = Image.new('RGB', (self.length, self.length), (255, 255, 255))
        
        # 画像を中央に配置するためのオフセットを計算
        offset = ((self.length - new_width) // 2, (self.length - new_height) // 2)
                
        # 画像を背景にペースト
        background.paste(self.image, offset)
            
        self.image = background
        return self

@dataclass
class InputData:
    """入力データを表すデータクラス

    Attributes:
        id (str): データの一意のID。
        image (PIL.Image): 画像ファイル。
        item_name (str): 商品名。
        expiry_type (str): 期限の種類（例："消費期限"）。
        expiry_date (datetime.date): 期限の日付。
        enable (bool): データが有効かどうかを示すフラグ。
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image: Image = None #画像ファイル
    item_name: str = ""
    expiry_type: str = "消費期限"
    expiry_date: type(datetime.date) = datetime.date.today()
    enable: bool = True


from pydantic import BaseModel
from typing import List

class Ingredient(BaseModel):
    食材: str
    期限種類: str
    期限: str

class Ingredients(BaseModel):
    食材リスト : List[Ingredient]
    目的: str


class DishProposer():
    def __init__(self):
        self.server_url = "http://172.30.0.3:8000/propose_dish/"
    
    def proposal(self,ingredients):
        response = self.upload(ingredients)
        return response

    def upload(self, data):
        """食材リストをアップロードするメソッド

        Returns:
            dict: サーバーからの応答データ。エラーが発生した場合はNone
        """
        try:
            response = requests.post(self.server_url, 
                                     data=json.dumps(data.dict()),
                                     headers = {'Content-Type': 'application/json'} 
            )
            response_data = response.json()
            return response_data
        except Exception as e:
            st.error(f"エラー: {str(e)}")
            return None


class App:
    """Streamlitアプリの主要な動作を管理するクラス

    Attributes:
        autoinput_image: 自動入力の画像
        input_data: 入力データのリスト
        input_column_width: 入力の列幅
        input_button_column_width: 入力ボタンの列幅
        db: データベースマネージャ
        pre_session_uploaded_image: 前回セッションでアップロードされた画像
        column_width: 列幅
        delete_item_id: 削除するアイテムのID
    """
    def __init__(self, db, user_db):
        """初期化メソッド

        Args:
            db (DatabaseManager): データベースマネージャオブジェクト。
        """
        #入力データ関係の初期化
        self.autoinput_image = None
        self.input_data = []
        self.input_column_width = [4,3,3,2,1]
        self.input_button_column_width = [4,4,8 ,4] 

        #DB関係の初期化
        self.db = db
        self.db.create()

        self.user_db = user_db
        self.user_db.create()
        self.new_user = ""
        self.del_user = ""

        #画像を画像処理サーバに送信する機能
        self.pre_session_uploaded_image = None

        #出力関係
        self.column_width = [4,3,3,3,2]
        self.delete_item_id = []

        #ログイン関係
        self.user = "guest"

    def make_input_data(self, image, data_dict):   
        """画像とデータ辞書から入力データを作成するメソッド

        Args:
            image: 処理する画像
            data_dict (dict): データの辞書

        Returns:
            list: InputDataオブジェクトのリスト
        """     
        items = []
        for row in data_dict["data"]:
                with Image.open(image) as img:
                    img = ImageProcessor(img)
                    xmin,ymin,xmax,ymax = row['coordinate'].values()
                    img.crop(xmin,ymin,xmax,ymax)
                    img.square()
                    #期限種類
                    if row["type"]:
                        expiry_type = row['type']
                    else:
                        expiry_type = "消費期限"
                    #期限選択
                    if row['date']:
                        expiry_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
                    else:
                        expiry_date = datetime.date.today()

                    item = InputData(
                        image = img.image,
                        item_name = row["name"],
                        expiry_type = expiry_type,
                        expiry_date = expiry_date
                    )
                    items.append(item)
        return items

    def autoinput(self):
        """画像をアップロードして消費期限を自動入力するメソッド"""
        columns = st.columns([6,3])
        with columns[0]:
            image = st.file_uploader("写真をアップロードすると消費期限が自動で入力されます", type=AVAILABLE_IMAGE_TYPE, key='auto_uploader')

            if ((image and not self.pre_session_uploaded_image) or #前回は画像がなかったが今回は画像がある
                (image and self.pre_session_uploaded_image and image!=self.pre_session_uploaded_image)):#前回と今回と画像が異なる
               
                #ファイルをサーバーへ転送
                uploader = ImageUploader(image)
                data_dict = uploader.upload()
                if "data" in data_dict.keys(): 
                    #入力データ更新
                    self.input_data = self.make_input_data(image, data_dict)
                else:
                    st.text(data_dict["detail"])
                #画像を保持
                self.autoinput_image = image
            self.pre_session_uploaded_image = image
                
        with columns[1]:
            if self.autoinput_image:
                st.image(self.autoinput_image, use_column_width =True)

    def input(self):    
        """入力フォームを表示するメソッド""" 
        button_columns = st.columns(self.input_button_column_width)

        #登録
        with button_columns[0]:
            self.register()
        # 入力データ追加
        with button_columns[1]:
            if st.button("入力欄追加"):
                self.input_data.append(InputData())
        #削除
        with button_columns[3]:
            if st.button("削除実行"):
                #無効なデータは全て削除
                self.input_data = [row for row in self.input_data if row.enable]

        #入力データを表示
        for i, row in enumerate(self.input_data):
            st.markdown("---") 
            new_image = st.file_uploader("画像変更", type=AVAILABLE_IMAGE_TYPE, key=f"uploader_{i}")
            if new_image:
                with Image.open(new_image) as img:
                    row.image = ImageProcessor(img).square().image
            columns = st.columns(self.input_column_width)
            with columns[4]:
                state = st.checkbox("削除",key={f"delete_{row.id}"})
                row.enable = not state

            #i行目の入力データを表示
            with columns[0]:
                if row.image:
                    st.image(row.image)         
            with columns[1]:
                row.item_name = st.text_input("品名", value=row.item_name ,key=f"name_{i}")
            with columns[2]:
                row.expiry_type = st.selectbox("期限種類", ["消費期限", "賞味期限"], index=EXPIRY_TYPE_DICT[row.expiry_type], key=f'type_{i}')
            with columns[3]:
                row.expiry_date = st.date_input("期限", value=row.expiry_date, key=f'date_{i}')

    def register(self):
        """データをデータベースに登録するメソッド"""

        if st.button("登録"):
            for row in self.input_data:
               #データベースに追加
                new_id = self.db.insert(self.user, row.item_name, row.expiry_type, row.expiry_date)
                #画像をimageフォルダへ保存
                if row.image:
                    image_path = os.path.join(self.db.image_dir, f"{new_id}.png")
                    row.image.save(image_path, 'PNG')

            #入力データリセット
            self.input_data = []

    def colored_write(self,expiry_date):
        """期限に基づいて色分けされたテキストを表示するメソッド

        Args:
            expiry_date (str): 期限の日付
        """
        expiry_date = datetime.datetime.strptime(expiry_date, '%Y-%m-%d').date()
        remaining = (expiry_date - datetime.date.today()).days

        if remaining < 0:
            color, text_color =  '#FF6666', "#000000"
        elif remaining == 0:
            color, text_color = '#FFA500', "#000000"
        elif remaining <= 3:
            color, text_color = '#FFFF66', "#000000"
        else:
            color, text_color = "", ""

        colored_text = f"<div style='background-color:{color}; color:{text_color};'>{expiry_date}</div>"
        st.markdown(colored_text, unsafe_allow_html=True)

    def display(self):
        """登録データを表示するメソッド"""
        if st.button("削除実行",key="display_delete_button"):
            for id in self.delete_item_id:
                self.db.delete(id)
            
        #削除候補をリセット
        self.delete_item_id = []

        #データベースから画像を引っ張ってきて表示
        for i, row in enumerate(self.db.fetch_all_products(self.user)):
            st.markdown("---") 
            columns = st.columns(self.column_width)
            
            #画像表示
            image_path = os.path.join(self.db.image_dir, f"{row['id']}.png")
            if os.path.exists(image_path):
                columns[0].image(Image.open(image_path))
            else:
                # 50pxの高さの空のスペースを確保する
                columns[0].markdown('<div style="height:150px;"></div>', unsafe_allow_html=True)  
            #品名 
            columns[1].write(row["item_name"])
            #期限種類
            columns[2].write(row["expiry_type"])
            #期限
            with columns[3]:
                self.colored_write(row["expiry_date"])
            #削除用チェックボックス
            with columns[4]:
                if st.checkbox("削除",key={f"delete_{row['id']}"}):
                    self.delete_item_id.append(row["id"])
    
    def dish(self):
        purpose = st.selectbox("食事の目的", ["夕食", "昼食", "朝食", "おやつ"], key="シチュエーション")
        placeholder = st.empty()
        if st.button("提案"):
            with placeholder.container():
                ing_list  = []
                for row in self.db.fetch_all_products(self.user):
                    ing_list.append(Ingredient(
                        食材 = row["item_name"],
                        期限種類 = row["expiry_type"],
                        期限 = row["expiry_date"]
                    ))
                ingredients = Ingredients(
                    食材リスト = ing_list,
                    目的= "夕食"
                )
            
                dishpropopser = DishProposer()
                dishes = dishpropopser.proposal(ingredients)

                for i, item in enumerate(dishes["Dishes"]):
                    dish =item["dish"]
                    ings = item["ingredients"]
                    steps = item["steps"]

                    st.header(f"料理{i+1}: {dish}")
                    st.subheader("食材")
                    st.write(', '.join(ings))
                    st.subheader(f"手順")
                    for i,x in enumerate(steps):
                        st.write(f"{i+1}: {x}")
                    st.markdown("---")

    def login(self):
        """ユーザ切替画面を表示するメソッド"""
        #ユーザ選択
        records = self.user_db.get_users()
        self.users = [r["name"] for r in records]
        user_selection = st.empty()
        st.markdown('---')

        #ユーザ登録
        if st.button("登録",key='user_register'):
            if self.new_user:
                self.user_db.register(self.new_user)
                records = self.user_db.get_users()
                self.users = [r["name"] for r in records]
        self.new_user = st.text_input("登録するユーザ名を入力して下さい", value="")
        st.markdown('---')

        #ユーザ削除
        if st.button("削除",key='user_delete'):
            if self.del_user:
                self.user_db.delete(self.del_user)
                records = self.user_db.get_users()
                self.users = [r["name"] for r in records]
        self.del_user = st.selectbox('削除するユーザ',self.users)

        #ユーザ選択
        self.user = user_selection.selectbox('ユーザを選択して下さい',self.users)

    
if __name__ == "__main__":
    #プログラム実行時に最初に一回だけ実行
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if not st.session_state.initialized:
        #最初に一回だけ実行
        db = DatabaseManager()
        user_db = user_manager = UserManager()
        st.session_state.app = App(db, user_db)
        st.session_state.initialized = True

    #タイトル表示
    st.set_page_config(layout="wide")
    st.title("消費期限管理アプリ")

    tabs = st.tabs(["登録","表示","料理提案","ユーザ切替"])
    
    with tabs[3]:
        #ユーザ切り替え
        st.session_state.app.login()

    with tabs[0]:
        #写真入力フォーム
        st.session_state.app.autoinput()

        #入力フォーム
        st.markdown("---") 
        st.session_state.app.input()

    with tabs[1]:
    #登録データの表示
        #st.subheader("登録データ一覧")
        st.session_state.app.display()   

    with tabs[2]:
        #料理提案
        st.session_state.app.dish()





