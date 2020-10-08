import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Bounding box extraction adapted from https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/
# Creates a bounding box using RGB or HSV bounds
def create_bounding_box(frame, color_frame, lower_bound, upper_bound):
    mask = cv2.inRange(color_frame, lower_bound, upper_bound)
    bins = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bins) > 0:
        area = max(bins, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(area)
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
    
    return mask, frame

# crops a frame to specified dimensions
def crop_frame(frame, x, y, h, w):
    start = (x, y)
    end = (x + w, y + h)
    return start, end, frame[y:y+h, x:x+w]

# Dominant color extraction adapted from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        start, end, cropped = crop_frame(frame.copy(), 100, 170, 250, 500)
        #cv2.rectangle(frame, start, end, (255, 0, 0), 3) #show cropped rectangle on frame
        kmeans_original = cropped.reshape((cropped.shape[0] * cropped.shape[1],3))
        cluster = KMeans(n_clusters=3)
        cluster.fit(kmeans_original)
        hist = find_histogram(cluster)
        bar = plot_colors(hist, cluster.cluster_centers_)

        bgr_original = frame
        bgr_color = np.array([35, 60, 150])
        bgr_lower_bound = np.array([10, 40, 110])
        bgr_upper_bound = np.array([60, 80, 190])
        _, bgr_frame = create_bounding_box(frame.copy(), bgr_original, bgr_lower_bound, bgr_upper_bound)

        hsv_original = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        hsv_color = np.array([13, 200, 150])
        hsv_lower_bound = np.array([5, 150, 120])
        hsv_upper_bound = np.array([10, 220, 180])
        _, hsv_frame = create_bounding_box(frame.copy(), hsv_original, hsv_lower_bound, hsv_upper_bound)

        cv2.imshow('BGR', bgr_frame)
        cv2.imshow('HSV', hsv_frame)
        cv2.imshow('Dominant color', bar)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

