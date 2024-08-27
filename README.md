# gguf-tiny
中文预训练大模型，天枢v2,gguf格式支持

# 训练目的
天枢是我本人训练的'大'语言模型，在某个方面来说，并不大，参数量仅18M，相较于目前开源的模型参数量在 2b,7b,9b,12b等量级上来说实在是不值一提。

所以我构建这个项目的原因并不是训练一个高性能，泛化能力强的语言模型，与之不同的是，我想要构建垂直模型，在未来，模型的轻量化蒸馏、量化等等技术成熟的时候，
人人都有可能获得自己的私人语言模型，在生活中的方方面面能够更针对性的服务人们的生活，故此我打算进行初步试验。

# 使用教程
首先是下载github上的包

`git clone https://github.com/jinliuxi1024/gguf-tiny`

由于我没有导出相关的依赖，不过放心，本项目依赖比较少，缺什么包安装什么

接着就是预处理文本分词器，支持各类文本，但是训练文本有要求，所以参考data_process提供的转化jsonl文本，当然也可以直接将txt作为训练文本，但是这样训练出来的模型仅有补全功能。

预训练分词器执行

`tokenizer_merge.py`文件，需要自己指定训练文本路径，此处比较宽松，是文档可读入都可以训练。

预训练完成后执行训练gpt模型，gpt模型架构使用llama3架构


执行`core.py`文件，同样需要自己指定训练文件路径，支持特定的jsonl和txt文件

模型转换为safetensor格式

在`core.py`有对应的函数，自己指定模型的路径

转化为gguf文件

转化为gguf文件比较麻烦，具体细节参考Nemo文件夹下的内容，正确对齐即可，需要将生成的safetensor和对应的index.json以及spm.model修改为tokenizer.model,并修改配置对齐core.py中的模型配置，对齐config.json等文件，tokenizer_config.json不会不用改。

使用llama.cpp运行gguf文件

`llama-cli -m 你的模型路径  --in-prefix "<|im_start|>少年: " --reverse-prompt "<|im_start|>" -co -p "<|im_start|>天枢核心: 主驱动<|im_end|>" -cnv -c 768`

当然如果不想折腾，可以git clone 本人的huggingface上提供的已经预训练好的模型直接运行（待更新）


# 模型配置说明

| 模型激活参数 | 模型架构 | 分词器 | 层数 | 自注意力头数 |
|---|---|---|---|---|
| 18M | llama3 | spm | 12 | 8 |

# 训练损失展示

<img width="985" alt="截屏2024-08-27 01 13 52" src="https://github.com/user-attachments/assets/67ef1058-936d-46b5-b642-674586e521c5">

# gguf推理展示

<img width="558" alt="截屏2024-08-27 11 25 25" src="https://github.com/user-attachments/assets/9c31464e-1cd2-4ac3-9693-ee0b477c0db2">



