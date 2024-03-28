from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import ultralytics.engine.results
import os
import rasterio
from pyproj import Transformer

Image.MAX_IMAGE_PIXELS = None

def cut_images(input_file:str, output_url:str, size:int, stride:int)->tuple:
    all_image = {}
    file_name = input_file.split('/')[-1]
    with Image.open(input_file) as im:
        width, height = im.size
        for i in range(0, width-size, stride):
            for j in range(0, height-size, stride):
                box = (i, j, i+size, j+size)
                cropped = im.crop(box)
                # cropped.save(output_url+file_name.split('.')[0] + f'_{i}_{j}_{stride}.' + file_name.split('.')[-1])
                all_image[(i,j)] = cropped
        return all_image

def compute_iou(box1, box2):
    """计算两个框的交并比"""

    # 计算交集矩形的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算交集面积与每个框面积的比例
    overlap1 = inter_area / box1_area if box1_area > 0 else 0
    overlap2 = inter_area / box2_area if box2_area > 0 else 0
    
    # 取两个比例中的最大值作为最终的重合度百分比
    iou = max(overlap1, overlap2)
    return iou

def custom_sort(boxes):
    # 首先按照置信度对框进行排序
    boxes_sorted_by_confidence = sorted(boxes, key=lambda x: x[4], reverse=True)

    # 检查是否有至少两个框，并且最高的两个置信度满足特定条件
    if len(boxes_sorted_by_confidence) >= 2 and boxes_sorted_by_confidence[0][4] > 0.6 and boxes_sorted_by_confidence[1][4] > 0.6 and abs(boxes_sorted_by_confidence[0][4] - boxes_sorted_by_confidence[1][4]) < 0.05:
        # 计算这两个框的面积
        area1 = (boxes_sorted_by_confidence[0][2] - boxes_sorted_by_confidence[0][0]) * (boxes_sorted_by_confidence[0][3] - boxes_sorted_by_confidence[0][1])
        area2 = (boxes_sorted_by_confidence[1][2] - boxes_sorted_by_confidence[1][0]) * (boxes_sorted_by_confidence[1][3] - boxes_sorted_by_confidence[1][1])
        
        # 如果第二个框的面积大于第一个框的面积，则交换这两个框的位置
        if area2 > area1:
            boxes_sorted_by_confidence[0], boxes_sorted_by_confidence[1] = boxes_sorted_by_confidence[1], boxes_sorted_by_confidence[0]
    
    # 返回最终排序后的框列表
    return boxes_sorted_by_confidence


def remove_overlaps(boxes, iou_threshold):
    """移除重合度高的框"""
    # 根据面积大小排序
    # boxes = sorted(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    # 按照置信度排序
    # boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    boxes = custom_sort(boxes)
    keep = []
    while boxes:
        box = boxes.pop(0)
        keep.append(box)
        boxes = [b for b in boxes if compute_iou(box, b) < iou_threshold]
    return keep

def screen_boxes(boxes, scale):
    """根据尺度筛选框"""
    return [box for box in boxes if (box[2]-box[0])*(box[3]-box[1]) >= (scale/2)**2]

def storm_detect(input_img: str, model_path: str, size: int, stride: int,confidence:float=0.5):
    model = YOLO(model_path)
    if not os.path.exists("./temp/input"):
        os.makedirs("./temp/input")
    all_files = cut_images(input_img, output_url="./temp/input/", size=size, stride=stride)
    detect_result = []
    for file_keypoint in all_files.keys():
        img = all_files[file_keypoint]
        results:ultralytics.engine.results.Results = model(img,save=True)
        conf = results[0].boxes.conf.cpu().numpy()
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        obj_types = results[0].boxes.cls.cpu().numpy()
        for i in range(len(conf)):
            if conf[i] > confidence:
                x1,y1,x2,y2 = xyxy[i]
                x1 = int(x1 + file_keypoint[0])
                y1 = int(y1 + file_keypoint[1])
                x2 = int(x2 + file_keypoint[0])
                y2 = int(y2 + file_keypoint[1])
                # 将检测结果的坐标变为相对于原始图像的坐标
                detect_result.append((x1,y1,x2,y2,conf[i],obj_types[i]))
    detect_result = screen_boxes(detect_result, size)
    return detect_result

def display_result(input_img:str, output_img:str, detect_result:list):
    with Image.open(input_img) as im:
        draw = ImageDraw.Draw(im)
        for item in detect_result:
            x1, y1, x2, y2, obj_conf,obj_type = item
            obj_type = "cyclone" if int(obj_type)==0  else "anticyclone"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=20)
            text_position = (x1, y1 - 10)
            label = f"{obj_type}: {obj_conf:.2f}"
            draw.text(text_position, label, fill="red", font=ImageFont.truetype("msyhbd.ttc", 200))
        im.save(output_img)

def transform_CRS(detect_result:list,input_img:str):
    with rasterio.open(input_img) as src:
        transform = src.transform
    transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)

    result = []
    for item in detect_result:
        x1, y1, x2, y2, obj_conf,obj_type = item
        size = ((x2-x1),(y2-y1))
        x = (x1+x2)/2
        y = (y1+y2)/2
        x, y = transform * (x, y)
        lon, lat = transformer.transform(x, y)
        result.append((lon, lat, obj_conf,obj_type,size))
    return result


def save_result(detect_result:list,output_file:str="output.txt"):
    with open(output_file,'w') as f:
        for item in detect_result:
            f.write(str(item)+'\n')
    
def main(input_img: str, output_label: str, model_path: str, size_stride: list[tuple],confidence:float=0.5,exclude:float=0.5):
    all_result = []
    for size,stride in size_stride:
        all_result += storm_detect(input_img, model_path, size, stride,confidence)
        # clean_temp()
    fixed_detect_result = remove_overlaps(all_result,exclude)
    # display_result(input_img, output_img, fixed_detect_result)
    fixed_detect_result = transform_CRS(fixed_detect_result,input_img)
    save_result(fixed_detect_result,output_label)

if __name__ == "__main__":
    main(
        input_img=r".\新建文件夹\20230708.tif",
        output_label="20230708_output.tif",
        model_path=r'./best (2).pt',
        size_stride=[(4500,900),(3000,600),(2000,400)],
        confidence=0.6,
        exclude=0.15)