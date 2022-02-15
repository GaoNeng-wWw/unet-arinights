# 计算机视觉辅助检查脑部源石结晶源代码
---

视频地址:https://www.bilibili.com/video/BV1RP4y1w7gm

---

## Q&A(视频中的一些补充)

- Q：为什么选择UNET
- A：UNET的overlap-tile对内存较为友好。同时不增加超参数的正确率也较高。所以我们最终选择了UNET，即便如此。在机器上依旧要么炸现存，要么炸内存。最终不得已我们选择花了十几块钱去搞了一台服务器，在上面跑。
- Q：数据集从哪里找
- A：https://www.kaggle.com/c/data-science-bowl-2018/data

如果需要编译完成的模型
链接: https://pan.baidu.com/s/1LGXsvU4ACxICgQEGDxBy0A 提取码: ig1p
