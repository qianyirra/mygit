import json
import tqdm 
import numpy as np 
import glob
import imagesize
from joblib import Parallel, delayed
import os 
import fire

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea

    if union == 0:
        return 0.0
    return interArea / union

def get_hw(k):
    width, height = imagesize.get(k)
    return [k,width,height]
##输入为raw data，Omniparser得到的结果，缓存的所有图像的宽高信息
def main(data_path="inp.json",bbox_dir="log",cache_path="hw_cache.json",iou_threshold=0.1, save_path="clean.json"):
    omni_data = {}
    for omni_path in glob.glob(f"{bbox_dir}/*.json"):#遍历所有log中的文件，glob.glob返回所有符合的文件路径列表
        with open(omni_path, 'r', encoding='utf-8') as f:
            for line in f:
                omni_data.update(json.loads(line))#按行读取json，每一行是一个JSON对象

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            hw_cache = json.load(f)
    else:
        hw_cache = {}

    to_update = [k for k in omni_data if k not in hw_cache]# # 筛选未缓存的图像路径

    results = Parallel(n_jobs=8)(
        delayed(get_hw)(k) for k in tqdm.tqdm(to_update)# 多线程并行获取宽高（n_jobs=8，8线程加速）
    )

    for k, w, h in results:
        hw_cache[k] = [w, h]

    with open(cache_path, "w") as f:
        json.dump(hw_cache, f)

    with open(data_path,"r") as f:
        data = json.load(f)

    filtered = 0
    skip = 0
    passed=0
    baddata=0

    def keep(item):
        nonlocal filtered, skip, passed, baddata
        if item["image"] not in omni_data:#如果该样本没有对应的Omniparser结果就跳过
            skip +=1
            return False
        if omni_data[item["image"]] is None:#图片没有对应的数据就跳过
            return False
        W,H = hw_cache[item["image"]]
        x0,y0,x1,y1 = item["bbox"]#实际bbox
        if x0 < 0 or x0 > 1000:#归一化bbox坐标超出[0,1000] → 异常数据（过滤）
            baddata+=1
            return False
        if x1 < 0 or x1 > 1000:
            baddata+=1
            return False
        if y0 < 0 or y0 > 1000:
            baddata+=1
            return False
        if y1 < 0 or y1 > 1000:
            baddata+=1
            return False
        x0 = int(x0/1000 * W)
        y0 = int(y0/1000 * H)
        x1 = int(x1/1000 * W)
        y1 = int(y1/1000 * H)
        boxA = (x0, y0, x1, y1)##将[0,1000]的坐标恢复到正常大小的坐标
        for other in omni_data[item["image"]]:
            boxB = other["bbox"]##Omniparser得到的bbox大小
            if iou(boxA, boxB) >= iou_threshold:#只要重叠区域大于0.1就要该样本
                passed+=1
                return True
        filtered+=1
        return False

    new_data = []
    for item in tqdm.tqdm(data):#进度条
        if not keep(item.copy()):#如果不符合条件就跳过
            continue
        new_data.append(item)#符合条件就加入新数据集，保存的还是在[0,1000]

    print("filtered data: ", filtered, "; skipped data: ", skip, "; passed data: ", passed, "; bad data", baddata)
    print("Before filtering: ", len(data), "; after_filtering: ", len(new_data))

    with open(save_path, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)