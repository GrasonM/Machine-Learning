import cv2
import numpy as np

# Paths to the model and config files
model_path = 'mobilenet_iter_73000.caffemodel'
config_path = 'deploy.prototxt'

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Initialize video source, webcam id 0 might need to be changed depending on the device
cap = cv2.VideoCapture(0)

# Define entry and exit lines
entry_line_position = 200
exit_line_position = 450

# Counters for people entering and exiting
entry_count = 0
exit_count = 0

# Function to check if a point (x, y) crosses the line
def is_crossing_line(y, line):
    return y < line

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    person_in_frame = False
    positions = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # COCO class label for person is 15
                person_in_frame = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw detection and bounding box
                label = f"Person: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Calculate center y-coordinate of the bounding box
                centerY = (startY + endY) / 2
                positions.append(centerY)

    # Check each person's position and count entry or exits
    for pos in positions:
        if is_crossing_line(pos, entry_line_position):
            entry_count += 1
        elif is_crossing_line(pos, exit_line_position):
            exit_count += 1

    # Draw entry and exit lines
    cv2.line(frame, (0, entry_line_position), (w, entry_line_position), (0, 255, 0), 2)
    cv2.line(frame, (0, exit_line_position), (w, exit_line_position), (0, 0, 255), 2)

    # Display the counts
    cv2.putText(frame, f"Entries: {entry_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Check for alarm condition
    if entry_count != exit_count:
        print("Alarm: Entry and Exit counts mismatch!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()