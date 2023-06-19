from packages import *

def make_train_val(root_path : str ="datasets/car", run_ok : bool =False):
    if run_ok:
        # make train, val folder
        train_path = os.path.join(root_path, "train")
        train_images_path = os.path.join(train_path, "images")
        train_labels_path = os.path.join(train_path, "labels")
        val_path = os.path.join(root_path, "val")
        val_images_path = os.path.join(val_path, "images")
        val_labels_path = os.path.join(val_path, "labels")
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(val_path):
            shutil.rmtree(val_path)
        os.makedirs(train_images_path, exist_ok=True)
        os.makedirs(train_labels_path, exist_ok=True)
        os.makedirs(val_images_path, exist_ok=True)
        os.makedirs(val_labels_path, exist_ok=True)
        # copy classes.txt
        classes_path = os.path.join(root_path, "classes.txt")
        shutil.copy(classes_path, os.path.join(train_labels_path, "classes.txt"))
        shutil.copy(classes_path, os.path.join(val_labels_path, "classes.txt"))

        # load data
        data_path = os.path.join(root_path, "data")
        images_path = sorted(glob.glob(os.path.join(data_path, "*.png")))
        # labels_path = sorted(glob.glob(os.path.join(data_path, "*.txt")))

        # split train val
        train, val = train_test_split(images_path, test_size=0.2, random_state=42)

        # write train
        print("train")
        for im_path in tqdm.tqdm(train):
            lb_path = im_path.replace(".png", ".txt")
            im_save_path = im_path.replace("/data", "/train/images")
            lb_save_path = lb_path.replace("/data", "/train/labels")
            shutil.copy(im_path, im_save_path)
            img_height, img_width, _ = cv2.imread(im_path).shape
            with open(lb_path, "r") as f:
                labels = f.readlines()
                f.close()
            new_labels = []
            for label in labels:
                line = list(map(float, label.strip().split(' ')))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x_center, y_center = float(((x_min + x_max) / 2) / img_width), float(((y_min + y_max) / 2) / img_height)
                width, height = abs(x_max - x_min) / img_width, abs(y_max - y_min) / img_height
                # print(x_center, y_center, width, height)
                new_labels.append([class_name, x_center, y_center, width, height])
            lines = ""
            for new_label in new_labels:
                lines += f"{str(new_label[0])} {str(new_label[1])} {str(new_label[2])} {str(new_label[3])} {str(new_label[4])}\n"    
            with open(lb_save_path, "w") as f:
                f.writelines(lines)
                f.close()

        # write val
        print("val")
        for im_path in tqdm.tqdm(val):
            lb_path = im_path.replace(".png", ".txt")
            im_save_path = im_path.replace("/data", "/val/images")
            lb_save_path = lb_path.replace("/data", "/val/labels")
            shutil.copy(im_path, im_save_path)
            img_height, img_width, _ = cv2.imread(im_path).shape
            with open(lb_path, "r") as f:
                labels = f.readlines()
                f.close()
            new_labels = []
            for label in labels:
                line = list(map(float, label.strip().split(' ')))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x_center, y_center = float(((x_min + x_max) / 2) / img_width), float(((y_min + y_max) / 2) / img_height)
                width, height = abs(x_max - x_min) / img_width, abs(y_max - y_min) / img_height
                # print(x_center, y_center, width, height)
                new_labels.append([class_name, x_center, y_center, width, height])
            lines = ""
            for new_label in new_labels:
                lines += f"{str(new_label[0])} {str(new_label[1])} {str(new_label[2])} {str(new_label[3])} {str(new_label[4])}\n"    
            with open(lb_save_path, "w") as f:
                f.writelines(lines)
                f.close()
        
        print("Done.")

    else:
        print("run_ok is False")