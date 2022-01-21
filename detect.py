# Import dependencies
import os
import shutil
import cv2
from time import time
from datetime import datetime

# Global variables
first_frame = None
video = cv2.VideoCapture(0)
fps_limit = 1
start_time = 1
min_area = 10000
nbr_img = 0

# Create img/ dir if necessary
if not os.path.exists('./img'):
    os.makedirs('./img')

# Capture loop
while True:
    _, frame = video.read()

    # Fps limit
    if (time() - start_time) > fps_limit:
        gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray0, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            continue

        delta_frame = cv2.absdiff(first_frame, gray)
        thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)
        (contours, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue  # Not much movement

            # Too much movement, need to capture that
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(os.path.join('./img', (datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')), frame)
            nbr_img += 1
        cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %H:%M:%S"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Uncomment to show live cam !
        # cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        first_frame = gray

        # Zip img by 1000 (memory gestion lets go !)
        if nbr_img >= 1000:
            now = datetime.now()
            date_time = now.strftime("%d_%m_%Y-%H:%M:%S")
            dirname = "arch" + date_time
            shutil.make_archive(dirname, 'zip', "img")
            files = os.listdir('img')
            for i in range(0, len(files)):
                os.remove('img/' + files[i])
            nbr_img = 0

            # total, used, free = shutil.disk_usage("/")
            # print("Total: %d GB" % (total // (2 ** 30)))
            # print("Used: %d GB" % (used // (2 ** 30)))
            # print("Free: %d GB" % (free // (2 ** 30)))

        start_time = time()

video.release()
cv2.destroyAllWindows()
