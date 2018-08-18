import face_recognition

# Load the jpg files into numpy arrays
evie_image = face_recognition.load_image_file("Evie2.jpg")
hiep_image = face_recognition.load_image_file("Hiep.jpg")
group_image = face_recognition.load_image_file("Group.jpg")

obama_image = face_recognition.load_image_file("obama.jpg")
biden_image = face_recognition.load_image_file("biden.jpg")

unknown_image = face_recognition.load_image_file("Evie3.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    evie_face_encoding = face_recognition.face_encodings(evie_image)[0]
    hiep_face_encoding = face_recognition.face_encodings(hiep_image)[0]
    group_face_encoding = face_recognition.face_encodings(group_image)

    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    evie_face_encoding,
    hiep_face_encoding,

    obama_face_encoding,
    biden_face_encoding,

    group_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results[0]))
print("Is the unknown face a picture of Obama? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))