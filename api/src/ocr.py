from google.cloud import vision
import datetime
import re
import io

def ocr(image):
    # 画像をバイト形式に変換
    byte_io = io.BytesIO()
    image.save(byte_io, 'JPEG')
    image_bytes = byte_io.getvalue()

    # 画像を読み込み、API に送信してテキストを検出
    image_ocr = vision.Image(content=image_bytes)
    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=image_ocr)
    texts = response.text_annotations

    return texts[1:]

def inside(point, range):
    """ある点がバウンディングボックス内にあるかどうかチェックする関数"""
    x,y = point.x, point.y
    xmin = range[0]
    ymin = range[1]
    xmax = range[2]
    ymax = range[3]
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def get_texts(box, texts):
    """バウンディングボックス内の文字列を取得する関数"""
    included_texts = []

    for text in texts:
        inside_count = sum(1 for vertex in text.bounding_poly.vertices if inside(vertex, box))
        if inside_count >= 2:
            included_texts.append(text.description)

    return "".join(included_texts)

def is_valid_date(year, month, date):
    year, month, date= int(year), int(month), int(date)
    try:
        datetime.date(year, month, date)
        return True
    except ValueError:
        return False

def find_expiration_date(txt):
    """マッチングをかける"""
    #上から順に優先順位が高い
    patterns = [
        #XX年XX月XX日
        r"(?P<year>\d{2})年(?P<month>\d{1,2})月(?P<date>\d{1,2})日?",   #年,月 必須
        r"(?P<year>\d{2})年(?P<month>\d{1,2}).?(?P<date>\d{1,2})日?",   #年 必須
        r"(?P<year>\d{2}).?(?P<month>\d{1,2})月(?P<date>\d{1,2})日?",   #月 必須

        #20XX.XX.XX
        r"20(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #年, 月を表すドット必須
        r"20(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #年を表すドット必須
        r"20(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #月を表すドット必須

        ##XX.XX.XX
        r"(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #年, 月を表すドット必須
        r"(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #年を表すドット必須
        r"(?P<year>\d{2})\.(?P<month>\d{1,2})\.(?P<date>\d{1,2})",  #月を表すドット必須

        #それ以外
        r"20(?P<year>\d{2}).?(?P<month>\d{1,2}).?(?P<date>\d{1,2})",  
        r"(?P<year>\d{2}).?(?P<month>\d{1,2}).?(?P<date>\d{1,2})"
    ]
    for p in patterns:
        for match in re.finditer(p, txt):
            y = match.group("year")
            m = match.group("month")
            d = match.group("date")
            if is_valid_date(y,m,d):
                return f"20{y}-{m}-{d}"
    else:
        return ""

def find_date_type(txt):
    patterns = [
        #完全一致
        r"消費期限",
        r"賞味期限",
        #三文字一致　(1文字　脱落or読み間違い)
        r"消費期", #先頭
        r"賞味期",
        r"消.?期限", #２番目
        r"賞.?期限",
        r"消費.?限", #3番目
        r"賞味.?限",
        r"費期限", #末尾脱落
        r"味期限", 
        #それ以外 2文字マッチ
        r"賞?味?期?限?",
        r"消?費?期?限?",    
    ]
    for p in patterns:
        for match in re.finditer(p, txt):
            type_str = match.group()
            #二文字以上　かつ　賞,味,消,費 いずれかを含む
            if len(type_str) >= 2 and ("賞" in type_str or "味" in type_str):
                return "賞味期限"
            if len(type_str) >=2 and ("消" in type_str or "費" in type_str):
                return "消費期限"
    else:
        return ""
    