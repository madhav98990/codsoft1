import cv2

# Update with the correct absolute path to your Haarcascade file
harcascade = r"C:\Users\madha\Downloads\archive (5)\haarcascade_frontalface_default.xml"

def describe_image(image_path):
    # Load the image
    img = cv2.imread(r"C:\Users\madha\OneDrive\Pictures\photo.webp")
    if img is None:
        print("Error: Image not loaded correctly.")
        return

    # Load the Haarcascade classifier
    facecascade = cv2.CascadeClassifier(harcascade)
    if facecascade.empty():
        print("Error: Haarcascade file not loaded.")
        return
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = facecascade.detectMultiScale(img_gray, 1.1, 4)
    
    # Describe the image
    height, width, channels = img.shape
    print(f"Image Description:")
    print(f"- Dimensions: {width} x {height} pixels")
    print(f"- Number of channels: {channels}")
    print(f"- Number of faces detected: {len(faces)}")
    
    if len(faces) > 0:
        print(f"- Face Coordinates (x, y, width, height):")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  Face {i+1}: (x: {x}, y: {y}, width: {w}, height: {h})")
            # Draw rectangles around the detected faces
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Resize and show the image with detected faces
    resized_img = cv2.resize(img, (800, 600))
    cv2.imshow("Detected Faces", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace with the correct path to your image file
describe_image(r"C:\Users\madha\OneDrive\Pictures\photo.webp")
