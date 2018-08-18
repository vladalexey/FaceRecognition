import face_recognition
import cv2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
derrick_image = face_recognition.load_image_file("Derrick.jpg")
derrick_face_encoding = face_recognition.face_encodings(derrick_image)[0]

# Load a second sample picture and learn how to recognize it.
j_image = face_recognition.load_image_file("J-2018.jpg")
j_face_encoding = face_recognition.face_encodings(j_image)[0]
#
# Load a second sample picture and learn how to recognize it.
evie_image = face_recognition.load_image_file("Evie.jpg")
evie_face_encoding = face_recognition.face_encodings(evie_image)[0]

# Load a second sample picture and learn how to recognize it.
thao_image = face_recognition.load_image_file("Thao2.jpg")
thao_face_encoding = face_recognition.face_encodings(thao_image)[0]

# Load a second sample picture and learn how to recognize it.
mai_image = face_recognition.load_image_file("Mai2.jpg")
mai_face_encoding = face_recognition.face_encodings(thao_image)[0]

# Load a second sample picture and learn how to recognize it.
tam_image = face_recognition.load_image_file("Tam.jpg")
tam_face_encoding = face_recognition.face_encodings(tam_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    derrick_face_encoding,
    j_face_encoding,
    evie_face_encoding,
    thao_face_encoding,
    tam_face_encoding,
    mai_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Derrick",
    "Jacob",
    "Evie",
    "Thao",
    "Tam",
    "Mai"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:

                # first_match_index = matches.index(True)
                # name = known_face_names[first_match_index]

                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # name = data["names"][i]

                    name = known_face_names[i]
                    # counts[name] = counts.get(name, 0) + 1

                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
                print(name, counts[name])

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()