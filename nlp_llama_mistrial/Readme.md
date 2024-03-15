#### 1. 结构
~~~
.
├── infer_api_client    %采用API方式进行推理在本地笔记本上的脚本
│   ├── req.py
│   └── utils.py
├── infer_api_server   %采用API方式进行推理在10.21.4.51服务器上的脚本
│   ├── infer_llama_desc_api.py
│   └── infer_mistrial_ocr_api.py
├── infer_local   %本地直接模型推理的脚本
│   ├── infer_llama_desc.py
│   └── infer_mistrial_ocr.py
└── Readme.md

~~~

#### 2.功能
~~~
desc: 指代输出场景播报
ocr： 指代识别出receipt的total cost
~~~

#### 3.运行方法
##### 3.1 infer_local:
~~~
python infer_llama_desc.py
python infer_mistrial_ocr.py
~~~

##### 3.2 infer_api
~~~
在server侧
python infer_llama_desc_api.py
python infer_mistrial_ocr_api.py
~~~

~~~
在client侧
1. 连接进researchvpn.polyu.edu.hk
2. 连接摄像头
3. 运行脚本
python req.py
~~~
