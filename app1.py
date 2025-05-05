import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model and define class labels
model = load_model('best_model.h5')
class_names = ['Grenade', 'Gun', 'Knife']

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to model's input size
    image = cv2.resize(frame, (224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display prediction on the original frame
    label = f"{pred_class}: {confidence*100:.2f}%"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Weapon Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
