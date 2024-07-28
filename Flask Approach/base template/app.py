import cv2
from flask import Flask, render_template, render_template_string, Response

app = Flask(__name__)
cap = cv2.VideoCapture(0)


import torch
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab

def center_check(box, vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag, class_tag, old_class_tag, annotated_frame):
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
        try:
            class_tag = int(box[6])
        except:
            class_tag = -1
        if (id_tag == old_id_tag) and (class_tag == old_class_tag):
            center_flag+=1
        
            vertical_flag = 0 
            horizontal_flag = 0

            if center_flag >= 13:
                cv2.imwrite("centered_object.jpg",annotated_frame)
                cap.release()
                cv2.destroyAllWindows()
                print("done?")
                print(model.names[int(box[6])])
        else:
            old_id_tag = id_tag
            old_class_tag = class_tag
            center_flag = 0

    return vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag, class_tag, old_class_tag

model = YOLO('yolov8n.pt')

def gen():  
    x_line1=200
    x_line2 =520  # Horizontal line 2
    y_line1=140
    y_line2 = 1140

    id_tag, old_id_tag = 0,0
    class_tag, old_class_tag = -1, -1
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
            cv2.line(resized_frame, (0, x_line1), (width, x_line1), (255, 0, 0), 1)
            cv2.line(resized_frame, (0, x_line2), (width, x_line2), (255, 0, 0), 1)
            cv2.line(resized_frame, (y_line1, 0), (y_line1, height), (255, 0, 0), 1)  # Vertical line 1
            cv2.line(resized_frame, (y_line2, 0), (y_line2, height), (255, 0, 0), 1)  # Vertical line 2


            # Visualize the results on the frame
            cropped_frame = resized_frame
            # Visualize the results on the cropped frame
            results = model.track(cropped_frame, conf=0.5, tracker = "bytetrack.yaml")

            # Combine the original frame with the annotated detections
            annotated_cropped_frame = results[0].plot()
            annotated_frame = resized_frame.copy()
            annotated_frame = annotated_cropped_frame

            # Display the annotated frame
            cv2.imwrite("YOLOv8 Inference.jpg", annotated_frame)
            yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('YOLOv8 Inference.jpg', 'rb').read() + b'\r\n')


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
                    else:
                        index_flag+=1
                
                box = boxes[index_flag]

                vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag, class_tag, old_class_tag   = center_check(box, vertical_flag, horizontal_flag, old_vertical, old_horizontal, id_tag, old_id_tag, center_flag, class_tag, old_class_tag, annotated_frame)

        else:
            break

  
    # while True:
    #     ret, image = video_capture.read()
    #     cv2.imwrite('t.jpg', image)
    cap.release()


@app.route('/')
def index():
    """Video streaming"""
    #return render_template('index.html')
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()