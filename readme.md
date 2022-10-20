[![ReadMe Card](https://github-readme-stats.vercel.app/api/pin/?username=TephrocactusHC&repo=plate-recognition&show_owner=True)](https://github.com/anuraghazra/github-readme-stats)
<br>
## 项目名称：车牌识别

### 项目描述
本项目用于静态图片的车牌自动识别功能。

**运行方法**：
运行直接在pycharm中运行`ui.py`文件，然后选择图片。或者在项目对应的目录下通过命令行运行`python ui.py`

项目中data文件夹提供模型的训练集和测试集
model为训练好的模型
train*.jpg图片均为我们所选择的展示图片，可以正常工作，识别准确
bad*.jpg则为项目中识别效果较差甚至是无法识别的图片
所有函数均被封装完成在ui.py中进行调用，不要直接运行其他文件。

依赖环境包括
- tensorflow== 2.x
- numpy
- opencv>= 4.0
- pyside6

如果项目启动不成功，请检查pyside6的动态链接库是否加载成功，如果不成功请卸载后重装或在linux环境下运行程序。

## 效果说明
由于使用了一个级联分类器辅助车牌定位，因此效果大打折扣，而且用openCV那些东西也不是最好的效果了。如果采用SOTA的模型进行定位，效果有质的提升。对于图像识别，还算是比较简单的，即使是很简单的神经网络也能取得非常好的效果。

## 关于本仓库
由于网速问题，没有上传训练集，请在网络上搜寻并自行下载。
