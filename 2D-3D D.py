import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
import demo
import subprocess

def process_image(image_path):
    # Modelinizi yükleyin ve ayarlayın
    net = PoseEstimationWithMobileNet()
    net = load_state(net, 'model_weights.pth')
    height_size = 368  # Görüntü yüksekliği (model için uygun bir değer seçin)

    # Görseli oku
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Model ile tahmin yapma
    heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride=8, upsample_ratio=4, cpu=False)

    # İşleme ve sonuçları kaydetme
    rect_path = image_path.replace('.%s' % (image_path.split('.')[-1]), '_rect.txt')
    total_keypoints_num = 0
    all_keypoints_by_type = []
    num_keypoints = Pose.num_kpts

    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    rects = []

    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        valid_keypoints = []
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
        valid_keypoints = np.array(valid_keypoints)
        
        if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
            pmin = valid_keypoints.min(0)
            pmax = valid_keypoints.max(0)
            center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
            radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))
        elif pose_entries[n][10] == -1.0 and pose_entries[n][13] == -1.0 and pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
            center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int)
            radius = int(1.45*np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0))
            center[1] += int(0.05*radius)
        else:
            center = np.array([img.shape[1]//2,img.shape[0]//2])
            radius = max(img.shape[1]//2,img.shape[0]//2)

        x1 = center[0] - radius
        y1 = center[1] - radius

        rects.append([x1, y1, 2*radius, 2*radius])

        # Dikdörtgenleri çizme
        cv2.rectangle(img, (x1, y1), (x1 + 2*radius, y1 + 2*radius), (0, 255, 0), 2)

    # Dikdörtgenlerin koordinatlarını dosyaya kaydetme
    np.savetxt(rect_path, np.array(rects), fmt='%d')
    print(f"Results saved to {rect_path}")

    # Görüntüyü kaydetme
    output_image_path = image_path.replace('.%s' % (image_path.split('.')[-1]), '_result.jpg')
    cv2.imwrite(output_image_path, img)
    print(f"Annotated image saved to {output_image_path}")

    # 3D modelleme işlemini yapma
    model_command = f"python -m apps.simple_test -r 256 --input {rect_path}"
    subprocess.run(model_command, shell=True, check=True)
    print("3D modeling done.")

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filename:
        process_image(filename)

root = tk.Tk()
root.title("Image Pose Estimation")

browse_button = tk.Button(root, text="Select Image", command=browse_file)
browse_button.pack(pady=20)

root.mainloop()