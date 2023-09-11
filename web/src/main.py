import streamlit as st
import pandas as pd
import sqlite3
from PIL import Image
import os
from datetime import date

# Constants
DATABASE = 'products.db'
IMAGE_DIR = "images"

def create_database():
    """Create a SQLite database and table if they don't exist."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS products 
                 (id INTEGER PRIMARY KEY, 
                 image_path TEXT, 
                 品名 TEXT, 
                 期限種類 TEXT, 
                 期限 DATE)''')
    conn.commit()
    return conn, c

def save_uploaded_image(uploaded_image):
    """Save the uploaded image and return its path."""
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)

    # Saving image with a unique name based on its order of uploading
    image_path = os.path.join(IMAGE_DIR, f"image_{len(os.listdir(IMAGE_DIR)) + 1}.png")
    
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    return image_path

def get_expiry_color_and_text_color(delta):
    """Get background color based on days left for expiry."""
    if delta < 0:
        return '#FF6666', "#000000"  # Red
    elif delta == 0:
        return '#FFA500', "#000000"  # Orange
    elif delta <= 3:
        return '#FFFF66', "#000000"  # Yellow
    else:
        return "", ""

def delete_image(image_path):
    """Delete an image file."""
    if os.path.exists(image_path):
        os.remove(image_path)

def main():
    st.title('消費期限管理アプリ')

    uploaded_image = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])

    input_cols = st.columns([3, 3, 3])
    name = input_cols[0].text_input("品名")
    type_expiry = input_cols[1].selectbox("期限種類", ["賞味期限", "消費期限"])
    expiry_date = input_cols[2].date_input("期限")

    conn, c = create_database()

    if st.button("行を追加") and uploaded_image:
        image_path = save_uploaded_image(uploaded_image)
        c.execute("INSERT INTO products (image_path, 品名, 期限種類, 期限) VALUES (?, ?, ?, ?)",
                  (image_path, name, type_expiry, expiry_date))
        conn.commit()

    c.execute("SELECT * FROM products ORDER BY 期限 ASC")
    rows = c.fetchall()

    # Display Header
    header_cols = st.columns([1, 4, 2, 2, 2, 1])
    header_cols[1].write("画像")
    header_cols[2].write("品名")
    header_cols[3].write("期限種類")
    header_cols[4].write("期限")

    for index, row in enumerate(rows, start=1):
        today = date.today()
        expiry = pd.to_datetime(row[4]).date()
        delta = (expiry - today).days
        
        color, text_color = get_expiry_color_and_text_color(delta)
    
        cols = st.columns([1, 4, 2, 2, 2, 1])
        cols[0].write(index)  # Display index
        if row[1]:
            cols[1].image(row[1], width=200)
        cols[2].write(row[2])
        cols[3].write(row[3])
        cols[4].markdown(f"<div style='background-color:{color}; color:{text_color};'>{row[4]}</div>", unsafe_allow_html=True)

        if cols[5].button("削除", key=f"delete_{row[0]}"):
            delete_image(row[1])  # Delete the corresponding image
            c.execute(f"DELETE FROM products WHERE id = {row[0]}")
            conn.commit()

    conn.close()

if __name__ == '__main__':
    main()
