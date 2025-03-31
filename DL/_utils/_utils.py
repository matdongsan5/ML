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
    
def printStorage(obj, obj_name):
    print(f"\n==== [{obj_name}] 기 본 정 보 ====")
    print(f'Shape : {obj.shape}')
    print(f'Dim : {obj.ndim}D')
    print(f'DType : {obj.dtype}')
    print(f'itemsize: {obj.itemsize}')
    print("==== STORAGE ====")
    print("Offset: ", obj.storage_offset())
    print("Strides: ",obj.stride())
    print("=====================")
    print(obj.untyped_storage())