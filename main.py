import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import Point, Polygon
from collections import defaultdict, deque

track_history = defaultdict(lambda: deque(maxlen=18))
polygon_pts = [(150, 350), (480, 350), (480, 150), (160, 150)]

counting_region = Polygon(polygon_pts)
count_ids = []
total_count = 0

model = YOLO("yolov8s-seg.pt")  # segmentation model
cap = cv2.VideoCapture("C:\\Users\\Administrator\\Desktop\\gettyimages-129877882-640_adpp.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('InstanceSegmentation-counting.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True, classes=[0],conf=0.20)

    if results[0].boxes.id is not None and results[0].masks is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotator.draw_region(reg_pts=polygon_pts, color=(0, 0, 180), thickness=2)

        for box, mask, track_id in zip(boxes, masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id))
            x1, y1, x2, y2 = box
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(im0, (cx, cy), 3, (255, 255, 255), cv2.FILLED)
            track_line = track_history[track_id]
            track_line.append(center)
            prev_position = track_line[-2] if len(track_line) > 1 else None
            is_inside = counting_region.contains(Point(center))

            if prev_position is not None and is_inside and track_id not in count_ids:
                count_ids.append(track_id)
                total_count += 1
    total = str(total_count)
    cv2.putText(im0, f"Total number of people entered: {total}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    out.write(im0)
    '''cv2.imshow("instance-segmentation-object-counting", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break'''

out.release()
cap.release()
cv2.destroyAllWindows()
