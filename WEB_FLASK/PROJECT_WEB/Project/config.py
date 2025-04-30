import os
# sqlite rdbms 파일 경로 관련 : db가 1개 파일 (확장자.db)
BASE_DIR = os.path.dirname(__file__)
DB_NAME_SQLITE = 'flask.db'


DB_SQLITE_URI = f'sqlite:///{os.path.join(BASE_DIR, DB_NAME_SQLITE)}'
# DB_MYSQL_URI = 'mysql+pymysql://root:1234@localhost:3306/testdb'
# DB_MARIA_URI = 'maria+mariadbconnector://root:root!@127.0.0.1:3308/db_ai'

# db관련 기능 구현 시 사용할 전역변수
SQLALCHEMY_DATABASE_URI = DB_SQLITE_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False
