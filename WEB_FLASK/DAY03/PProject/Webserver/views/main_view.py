## APP MODULIZATION - Main Module

## 모듈 로딩
from flask import Blueprint

## 매개변수 
""" 블루 프린트 인스턴스 이름           name: str, 
    블루프린트가 정의된 현재 모듈 이름  import_name: str,
    static_folder: str | PathLike[str] | None = None,
    static_url_path: str | None = None,
    template_folder: str | PathLike[str] | None = None,
    기본 URL을 생략한 시작 URL (http://127.0.0.1:5000)  url_prefix: str | None = None,
    subdomain: str | None = None,
    url_defaults: dict[str, Any] | None = None,
    root_path: str | None = None,
    cli_group: str | None = _sentinel """
main_bp = Blueprint('MAIN', 
                    __name__, 
                    template_folder='templates',
                    url_prefix='/')

## URL rule 등록 데코레이터

@main_bp.route('/')
def index():
    return "Hello~^^"





















    