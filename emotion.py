import cv2
from fer import FER

# Load the face detection classifier
face_cap = cv2.CascadeClassifier("C:/Users/parma/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# Initialize video capture
video_cap = cv2.VideoCapture(0)
# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extract the face from the frame
        face_img = video_data[y:y + h, x:x + w]
        # Detect emotions on the face
        emotions = emotion_detector.detect_emotions(face_img)
        if emotions:
            # Get the dominant emotion
            dominant_emotion, emotion_score = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
            # Overlay the dominant emotion on the video frame
            cv2.putText(video_data, f'{dominant_emotion}: {emotion_score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Display the frame with face and emotion detection
    cv2.imshow("video_live", video_data)
    
    # Break the loop if the 'a' key is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture object
video_cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()