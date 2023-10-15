from PIL import Image
import requests

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

        response = requests.post(self.server_url, files=files)
        response_data = response.json()
        return response_data


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