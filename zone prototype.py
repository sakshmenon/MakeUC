import cv2
import torch
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

def center_check(box, vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag):
    if (box[2] + box[0])/2 < 140:
        horizontal_flag = -1
        center_flag = 0
        if (old_horizontal != horizontal_flag):
            print('move camera left, or move object right')

    elif (box[2] + box[0])/2 > 1140:
        horizontal_flag = 1
        center_flag = 0
        if (old_horizontal != horizontal_flag):
            print('move camera right, or move object left')
            
    elif (box[3] + box[1])/2 < 200:
        vertical_flag = -1
        center_flag = 0
        if (old_vertical != vertical_flag):
            print('move camera up, or move object down')
            
    elif (box[3] + box[1])/2 > 520:
        vertical_flag = 1
        center_flag = 0
        if (old_vertical != vertical_flag):
            print('move camera down, or move object up')
            
    else:
        id_tag = box[4]
        if (id_tag == old_id_tag):
            center_flag+=1
        
            vertical_flag = 0 
            horizontal_flag = 0

            if center_flag >= 13:
                cap.release()
                cv2.destroyAllWindows()
                print("done?")
                print(model.names[box[6]])
        else:
            old_id_tag = id_tag
    return vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device('cpu')
    print('Using device:', device)


model = YOLO('yolov8x.pt')


x_line1=200
x_line2 =520  # Horizontal line 2
y_line1=140
y_line2 = 1140

frame_count = 0

id_tag, old_id_tag = 0,0
max = 0

vertical_flag, horizontal_flag, old_vertical, old_horizontal = 0,0,0,0
center_flag = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    if success:
        # Run YOLOv8 inference on the frame
        resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.line(resized_frame, (0, x_line1), (width, x_line1), (255, 0, 0), 10)
        cv2.line(resized_frame, (0, x_line2), (width, x_line2), (255, 0, 0), 10)
        cv2.line(resized_frame, (y_line1, 0), (y_line1, height), (255, 0, 0), 10)  # Vertical line 1
        cv2.line(resized_frame, (y_line2, 0), (y_line2, height), (255, 0, 0), 10)  # Vertical line 2


        # Visualize the results on the frame
        cropped_frame = resized_frame#[x_line1:, y_line1:]#[0:x_line1:x_line2, 0:y_line1:y_line2]

        # Visualize the results on the cropped frame
        results = model.track(cropped_frame, conf=0.5, tracker = "bytetrack.yaml")

        # Combine the original frame with the annotated detections
        annotated_cropped_frame = results[0].plot()
        annotated_frame = resized_frame.copy()
        annotated_frame = annotated_cropped_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        num_objects = len(results[0].boxes)
        
        boxes = results[0].boxes.data


        if len(boxes) > 0:
            index_flag = -1
            for box in boxes:
                if box[5]> max:
                    max = box[5]
                    index_flag+=1
            
            box = boxes[index_flag]

            vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag   = center_check(box, vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag)

        frame_count += 1

    else:
        break

# Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
