import cv2

class VehicleDetector:
    def __init__(self, config_file, frozen_model, labels_file):
        self.obj_model = cv2.dnn_DetectionModel(frozen_model, config_file)
        self.configure_model()
        self.class_labels = self.load_labels(labels_file)

    def configure_model(self):
        self.obj_model.setInputSize(320, 320)
        self.obj_model.setInputScale(1.0 / 127.5)
        self.obj_model.setInputMean((127.5, 127.5, 127.5))
        self.obj_model.setInputSwapRB(True)

    def load_labels(self, filename):
        with open(filename, 'rt') as fpt:
            return fpt.read().rstrip('\n').split('\n')

    def detect_vehicles(self, image):
        ClassIndex, confidence, bbox = self.obj_model.detect(image, confThreshold=0.5)

        if len(ClassIndex) == 0:
            return None, None

        best_contour = None
        best_contour_area = 0
        best_vehicle_box = None

        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < ClassInd <= len(self.class_labels):
                x, y, w, h = boxes
                vehicle_roi = image[y:y + h, x:x + w]

                # Process vehicle ROI
                vehicle_gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                _, vehicle_thresh = cv2.threshold(vehicle_gray, 150, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(vehicle_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > best_contour_area:
                        best_contour_area = area
                        best_contour = contour
                        best_vehicle_box = boxes

        return best_contour, best_vehicle_box
