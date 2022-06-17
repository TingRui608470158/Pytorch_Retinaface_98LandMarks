# RetinaFace的98 Landmarks版本
參考[biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface.git)，將其從5 Landmarks改為98 landmarks。
68 Landmarks 版本要到 Retinaface_68LandMarks branch去，使用300W dataset。

## Dataset WFLW
1. 下載[WFLW資料集](https://wywu.github.io/projects/LAB/WFLW.html
2. 檔案路徑如下
```
./data/
    WFLW/
        WFLW_annotations/...
        WFLW_images/...
```
## Training 
```
python3 train.py
```

## Detect 
```
python3 detect.py
```



- [ ] 增加MobilenetV2 Backbone
- [ ] pitch row yaw detect 
