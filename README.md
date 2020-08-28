# Text_category
## Usage
### Import

```
from text_category.config import Config
from text_category.text_category import Train, Test, Predict
```

### Config.yml

```
logging:
    root: INFO
```

### Train / Test / Predict

```
import os
import hao
from text_category.config import Config
from text_category.text_category import Train, Test, Predict

root_dir = hao.paths.project_root_path()

dataset_path = os.path.join(root_dir, "examples")
model_name = "FastText"
config = Config(model_name, dataset_path)

if __name__ == '__main__':
    Train(model_name, config)
    # Test(model_name, config)
    # line = "明報 專訊 新年 長假 過後 面對 繁重 工作 學業 情緒 低落 面色 暗沉 墨汁 減 減壓 減壓 方法 美術 教師 國畫 一門 修心 養性 陶冶 性情 藝術 學習 國畫 生活 留點 空白 有助 鬆弛 神經 樂趣 完成品 繪畫 過程 一種 享受"
    # result = Predict(model_name, config).predict_line(line)
    # print(result)

# dataset_path目录中，必须含有
#   「train_dev_test」：
#       train.txt、dev.txt、test.txt
#       其中每个文件的具体内容样式为：
#           数据组织形式为：每行为一条数据：str("分词后的句子")+"映射后的标签"
#           example:明報 專訊 新年 長假 過後 面對 繁重 工作 學業 情緒 低落 面色 暗沉 墨汁 減 減壓 減壓 方法 美術 教師 國畫 一門 修心 養性 陶冶 性情 藝術 學習 國畫 生活 留點 空白 有助 鬆弛 神經 樂趣 完成品 繪畫 過程 一種 享受+1 
#   「classes」：
#       class.txt
#       具体内容样式为：
#           7.1.艺术文学-美术.txt + 0
#           7.2.艺术文学-书法.txt + 1
```

## Install

```
pip install text_category
```