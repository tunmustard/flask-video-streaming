import cv2
from base_camera import BaseCamera

############
import face_recognition
# Load a sample picture and learn how to recognize it.
anton_image = face_recognition.load_image_file("./data/train/anton/anton-train.jpg")
anton_encoding = face_recognition.face_encodings(anton_image)[0]

# Load a second sample picture and learn how to recognize it.
marina_image = face_recognition.load_image_file("./data/train/marina/marina-train.jpg")
marina_encoding = face_recognition.face_encodings(marina_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    anton_encoding,
    marina_encoding
]
known_face_names = [
    "Anton",
    "Marina"
]
####################

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face in this frame of video
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            	# See if the face is a match for the known face(s)
            	matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            	name = "Unknown"
            	# If a match was found in known_face_encodings, just use the first one.
            	if True in matches:
                	first_match_index = matches.index(True)
                	name = known_face_names[first_match_index]
                	print("Name %s"%name)

            	# Draw a box around the face
            	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            	# Draw a label with a name below the face
            	cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            	font = cv2.FONT_HERSHEY_DUPLEX
            	cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
