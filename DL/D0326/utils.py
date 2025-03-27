""" 
사용자 정의 함수 모음
1. printTensorInfo() : 텐서의 속성 출력

"""
## 모듈로딩
import torch


## 텐서의 속성 확인
def printTensorInfo(tname, tensor):
    """ 
    텐서의 속성 출력<br>
    tname : 이름string<br>
    tensor : tensor 객체<br>
    no return<br>
    """
    print(f"{tname}---------------------------------------")
    print(f" shape  : {tensor.shape}    {tensor.size()}")
    print(f" ndim   : {tensor.ndim}       {tensor.dim()}")
    print(f" device : {tensor.device}")
    print(f" dtype  :  {tensor.dtype}")
    print(f" data   :  {tensor}")
    