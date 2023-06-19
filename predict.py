from packages import *
from utils import *

def predict(im_path="img path", model=None, conf=0.7):
    outputs = []
    file_name = im_path.split("/")[-1]
    results = model.predict(source=im_path, verbose=False, iou=0.85, device=0)  # predict on an image
    for result in results:
        score = result.boxes.conf.cpu().numpy()  # 객체 당 score tensor 리스트
        class_id = result.boxes.cls.cpu().numpy()  # 객체 당 class 리스트
        xyxy = result.boxes.xyxy.cpu().numpy()  # 객체 당 xyxy 리스트

        for i in range(len(class_id)):
            if score[i] >= conf:
                x1, y1, x2, y2 = [float(x) for x in xyxy[i]]
                outputs.append([file_name, class_id[i], score[i], x1, y1, x2, y1, x2, y2, x1, y2])
    return outputs


if __name__ == "__main__":
    model = YOLO(model="runs/car/weights/best.pt", task="detect")
    dataset_path = "datasets/car/test/*.png"
    
    results = []
    
    for im_path in tqdm.tqdm(glob.glob(dataset_path)):
        outputs = predict(im_path, model, conf=0.2)
        for output in outputs:
            results.append(output)
        
    df_submission = pd.DataFrame(results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d_%H-%M-%S') 
    df_submission.to_csv(f"results/{time_now}.csv", index=False)