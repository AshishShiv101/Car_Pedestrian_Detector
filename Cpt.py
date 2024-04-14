import cv2

img_file = "Car.jpg"
video = cv2.VideoCapture("car_video1.mp4")
classifier_file = "car_detector.xml"

# Load cascade classifiers
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker_file = "haarcascade_fullbody.xml"  # Corrected
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()

    if not read_successful:
        break

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for x, y, w, h in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Object Detector", frame)

    key = cv2.waitKey(50)  # Reduced delay to 10 milliseconds
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
