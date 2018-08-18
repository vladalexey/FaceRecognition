# import the necessary packages
import concurrent.futures
from imutils.video import VideoStream
import numpy as np
import heapq
import imutils
import face_recognition
import argparse
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
# ap.add_argument("c", "--cpus", type=int, default=1, help="number of cpus will be used")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

processing_this_frame = True

# # load the input image and convert it from BGR to RGB
# image = cv2.imread(args["image"])
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
# print("[INFO] recognizing faces...")
# boxes = face_recognition.face_locations(rgb,
#                                         model=args["detection_method"])
# encodings = face_recognition.face_encodings(rgb, boxes)
#
# # initialize the list of names for each face detected
# names = []
#
# # loop over the facial embeddings
# for encoding in encodings:
#     # attempt to match each face in the input image to our known
#     # encodings
#     matches = face_recognition.compare_faces(data["encodings"], encoding)
#     name = "Unknown"
#     # check to see if we have found a match
#     if True in matches:
#         # find the indexes of all matched faces then initialize a
#         # dictionary to count the total number of times each face
#         # was matched
#         matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#         counts = {}
#
#         # loop over the matched indexes and maintain a count for
#         # each recognized face face
#         for i in matchedIdxs:
#             name = data["names"][i]
#             counts[name] = counts.get(name, 0) + 1
#
#         # determine the recognized face with the largest number of
#         # votes (note: in the event of an unlikely tie Python will
#         # select first entry in the dictionary)
#         name = max(counts, key=counts.get)
#
#     # update the list of names
#     names.append(name)
#
# # loop over the recognized faces
# for ((top, right, bottom, left), name) in zip(boxes, names):
#     # draw the predicted face name on the image
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#     y = top - 15 if top - 15 > 15 else top + 15
#     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75, (0, 255, 0), 2)
#
# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    r = frame.shape[1] / float(rgb.shape[1])

    if processing_this_frame:

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding, tolerance=0.45)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                countsWeight = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    countsWeight[name] = countsWeight.get(name, 0)

                for name in counts.keys():
                    countsWeight[name] = (counts[name] + counts[name] * 0.1) / data["namescount"][name] * 100

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)

                name = max(countsWeight, key=countsWeight.get)

                if len(countsWeight) > 1:

                    knownStrictEncodings = []
                    knownStrictNames = []

                    largest = countsWeight[name]
                    print("Largest " + name, largest)

                    del countsWeight[name]

                    name2 = max(countsWeight, key=countsWeight.get)
                    se_largest = countsWeight[name2]
                    print("Largest 2 " + name2, se_largest)

                    if largest - se_largest < 12:

                        print("[Strict compare]")

                        for index in range(len(data["names"])):

                            if data["names"][index] == name or data["names"][index] == name2:

                                knownStrictEncodings.append(data["encodings"][index])
                                knownStrictNames.append(data["names"][index])

                        set_tolerance = 0.01

                        strictMatches = face_recognition.compare_faces(knownStrictEncodings, encoding, set_tolerance)

                        while True:
                            if True in strictMatches:
                                break

                            else:
                                set_tolerance += 0.01
                                strictMatches = face_recognition.compare_faces(knownStrictEncodings, encoding,
                                                                               set_tolerance)

                        strictCount = {}
                        strictCountsWeight = {}

                        for (index, test) in enumerate(strictMatches):
                            if test:
                                strictCount[knownStrictNames[index]] = strictCount.get(knownStrictNames[index], 0) + 1
                                strictCountsWeight[knownStrictNames[index]] = strictCountsWeight.get(knownStrictNames[index], 0)

                        print("[Strict Count]", end="")
                        print(strictCount)

                        for name in strictCount.keys():
                            strictCountsWeight[name] = (strictCount[name] + strictCount[name] * 0.1) / data["namescount"][name] * 100

                        print("[Strict Weight]", end='')
                        print(strictCountsWeight)

                        if len(strictCount) > 0:
                            name = max(strictCountsWeight, key=strictCountsWeight.get)

                else:

                    print(name, data["namescount"][name], counts[name], countsWeight[name])

            # update the list of names
            names.append(name)

    processing_this_frame = not processing_this_frame

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()