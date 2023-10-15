# expiration_manager

## 準備
### Google Cloud Vision API
Google Cloud のサーバを用意してVision APIを有効化してください。

### OpenAI API
OpenAIへ登録してAPIを有効化してください。

### api keyとモデルのパラメータファイル
server/app/configディレクトに次のファイルを用意してください。
- google-api-key.json : google cloudのapi key
- openai-api-key.txt : OpenAIのapi key
- model.pth: 物体検出モデルの学習済みの重み(detectron2のfaster_rcnn_X_101_32x8d_FPN_3x)

物体検出モデルのパラメータを変更する場合はconfig.yamlを変更してください。  
https://detectron2.readthedocs.io/en/latest/modules/config.html

## 環境構築
### 開発環境
```
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml up
```
server・clientそれぞれにおいて、  
appディレクトリがコンテナの/appへマウントされます。

### 本番環境
```
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up
```
server・clientそれぞれにおいて、  
appディレクトリのファイルがコンテナの/appへコピーされます。
