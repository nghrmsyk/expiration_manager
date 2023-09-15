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
    def __init__(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(file_dir, "DB", "images")
        self.db_path = os.path.join(file_dir, "DB", "product.db")

    def connect(self):
        #DBを配置するフォルダと画像を保存するフォルダ作成
        os.makedirs(self.image_dir, exist_ok=True)
        #DB接続・なければ作成
        return sqlite3.connect(self.db_path)
        
    def __dell__(self):
        self.conn.close()

    def create_table(self):
        sql = '''CREATE TABLE IF NOT EXISTS product( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name VARCHAR(50),
                expiry_type CHAR(4),
                expiry_date 
                )'''
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute(sql)
        conn.commit()
        conn.close()
     

    def insert(self, item_name, expiry_type, expiry_date):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO product (item_name, expiry_type, expiry_date)
                        VALUES (?, ?, ?)''', (item_name, expiry_type, expiry_date))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return new_id
    
    def fetch_all_products(self):
        conn = self.connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM product ORDER BY expiry_date")
        table = cursor.fetchall()
        conn.close()
            
        return table
    
    def delete(self, id):
        conn = self.connect()
        cursor =  conn.cursor()
        cursor.execute("DELETE FROM product WHERE id=?", (id,))
       
        conn.commit()
        conn.close()
        # 関連する切り出し画像を削除
        image_path = os.path.join(self.image_dir, f"{id}.png")
        if os.path.exists(image_path):
            os.remove(image_path)

# 画像処理サーバへのリクエスト
class ImageProcessor:
    def __init__(self):
        self.server_url = "http://172.30.0.3:8000/food-expiration/"

    def upload_image(self, image):
        """Upload the image to the API and get data."""
        file_name = image.name
        mime_type = self.get_content_type(file_name)
        files = {"file": (file_name, image.getvalue(), mime_type)}

        try:
            response = requests.post(self.server_url, files=files)
            response_data = response.json()
            return response_data
        except Exception as e:
            st.error(f"エラー: {str(e)}")
            return None
        
    def process_image(self, image):
        data_dict = self.upload_image(image)
        items = self.set_input_data(image, data_dict)
        return items

    @staticmethod
    def get_content_type(file_name):
        """Get MIME type based on file extension"""
        if file_name.endswith(".png"):
            return "image/png"
        elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            return "image/jpeg"
        else:
            return "application/octet-stream"
        
    def image_crop(self, image,coordinate):
        # 各アイテムに対応する画像領域をクロッピング
        image = Image.open(image)
        img_width, img_height = image.size
        cx, cy, w, h = coordinate['cx'], coordinate['cy'], coordinate['w'], coordinate['h']
        left = int((cx - w/2) * img_width)
        upper = int((cy - h/2) * img_height)
        right = int((cx + w/2) * img_width)
        lower = int((cy + h/2) * img_height)
        return image.crop((left, upper, right, lower))
        
    
    def set_input_data(self, image, data_dict):        
        items = []
        for row in data_dict["data"]:
            cropped_image = self.image_crop(image, row['coordinate'])

            item = InputData(
                image = cropped_image,
                item_name = row["name"],
                expiry_type = row["type"],
                expiry_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
            )
            items.append(item)
        return items
            


@dataclass
class InputData:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image: PIL.PngImagePlugin.PngImageFile = None #画像ファイル
    item_name: str = ""
    expiry_type: str = "消費期限"
    expiry_date: type(datetime.date) = datetime.date.today()
    enable: bool = True

class App:
    def __init__(self,db, uploader):
        #入力データ関係の初期化
        self.autoinput_image = None
        self.input_data = []
        self.input_column_width = [4,3,3,2,5,1]
        self.input_button_column_width = [2,4,11 ,3] 

        #DB関係の初期化
        self.db = db
        self.db.create_table()

        #画像を画像処理サーバに送信する機能
        self.uploader = uploader
        self.pre_session_uploaded = None

        #出力関係
        self.column_width = [4,3,3,3,2]
        self.delete_item_id = []


    def autoinput(self):
        columns = st.columns([6,3])
        with columns[0]:
            image = st.file_uploader("写真入力", type=AVAILABLE_IMAGE_TYPE, key='auto_uploader')

            if ((image and not self.pre_session_uploaded) or #前回は画像がなかったが今回は画像がある
                (image and self.pre_session_uploaded and image!=self.pre_session_uploaded)):#前回と今回と画像が異なる
                self.autoinput_image = Image.open(image)
                #ファイルをサーバーへ転送
                self.input_data = self.uploader.process_image(image)
            self.pre_session_uploaded = image
                
        with columns[1]:
            if self.autoinput_image:
                st.image(self.autoinput_image, use_column_width =True)

    def input(self):     
        button_columns = st.columns(self.input_button_column_width)

        #登録
        with button_columns[0]:
            self.register()
        # 入力データ追加
        with button_columns[1]:
            if st.button("入力データ追加"):
                self.input_data.append(InputData())
        #削除
        with button_columns[3]:
            if st.button("入力データ削除"):
                #無効なデータは全て削除
                self.input_data = [row for row in self.input_data if row.enable]

        #入力データを表示
        for i, row in enumerate(self.input_data):
            st.markdown("---") 
            columns = st.columns(self.input_column_width)
            with columns[5]:
                state = st.checkbox("削除",key={f"delete_{row.id}"})
                row.enable = not state
                    
            with columns[4]:
                #i行目の画像入力フォーム
                input_new_image = st.file_uploader("画像変更", type=AVAILABLE_IMAGE_TYPE, key=f"uploader_{i}")
                if input_new_image:
                    row.image = Image.open(input_new_image)

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
        if st.button("登録"):
            for row in self.input_data:
               #データベースに追加
                new_id = self.db.insert(row.item_name, row.expiry_type, row.expiry_date)
                #画像をimageフォルダへ保存
                if row.image:
                    image_path = os.path.join(self.db.image_dir, f"{new_id}.png")
                    row.image.save(image_path, 'PNG')

            #入力データリセット
            self.input_data = []
    def colored_write(self,expiry_date):
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
        header_cols = st.columns(self.column_width)
        header_cols[0].write("画像")
        header_cols[1].write("品名")
        header_cols[2].write("期限種類")
        header_cols[3].write("期限")
        with header_cols[4]:
            if st.button("登録データ削除"):
                for id in self.delete_item_id:
                    self.db.delete(id)
            
        #削除候補をリセット
        self.delete_item_id = []

        #データベースから画像を引っ張ってきて表示
        for i, row in enumerate(self.db.fetch_all_products()):
            st.markdown("---") 
            columns = st.columns(self.column_width)
            
            #画像表示
            image_path = os.path.join(self.db.image_dir, f"{row['id']}.png")
            if os.path.exists(image_path):
                columns[0].image(Image.open(image_path))
            else:
                # 50pxの高さの空のスペースを確保する
                columns[0].markdown('<div style="height:100px;"></div>', unsafe_allow_html=True)  
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


    
if __name__ == "__main__":
    #プログラム実行時に最初に一回だけ実行
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if not st.session_state.initialized:
        #最初に一回だけ実行
        uploader = ImageProcessor()
        db = DatabaseManager()
        st.session_state.app = App(db, uploader)
        st.session_state.initialized = True

    #タイトル表示
    st.set_page_config(layout="wide")
    st.title("消費期限管理アプリ")

    #入力フォーム
    st.session_state.app.autoinput()
    st.session_state.app.input()

    #登録データの表示
    st.markdown("---") 
    st.subheader("登録データ一覧")
    st.session_state.app.display()   

