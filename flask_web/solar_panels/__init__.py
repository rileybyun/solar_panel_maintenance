## "__init__.py" => flask run 시 자동 실행
from flask import Flask

def create_app():
    app = Flask(__name__)

    # @app.route('/')
    # def hello():
    #     return 'Hello World!'

    from .views import main_views
    app.register_blueprint(main_views.bp)   # app에 블루프린트 객체 bp 등록

    return app