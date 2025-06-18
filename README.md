Main tools in src, data for original videos, output for pictures

# MemeLLM

这个项目提供表情包数据库以及根据需求搜索的功能。

## usage

### 环境配置

`pip install -r requirements.txt`

### embedding 

将表情包放在 `src/output` 目录下。使用meme中的文字（或者它的含义）作为文件标题。

例如，需要将`src/output/vv`和`src/output/tiansuo`下的图片加入搜索库，运行：

`python find_meme/embedding2.py --path vv tiansuo`

如果你需要重置搜索库，需要再上一行命令后增加`--reset`。

### 询问

运行

`python find_meme/interface4.py`

并打开`localhost:7680`即可。
