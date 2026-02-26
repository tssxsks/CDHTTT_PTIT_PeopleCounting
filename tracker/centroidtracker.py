from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    """
    Tracker don gian dua tren tam (centroid) cua bounding box.

    Y tuong:
    - Moi doi tuong duoc gan mot ID.
    - O moi frame, tinh centroid cua cac bbox moi.
    - Ghep centroid moi voi centroid cu bang khoang cach Euclid nho nhat.
    - Neu mot doi tuong mat qua nhieu frame lien tiep thi xoa khoi tracker.
    """

    def __init__(self, maxDisappeared=50, maxDistance=50):
        # Giu nguyen ten tham so de tuong thich voi code cu
        if maxDisappeared < 0:
            raise ValueError("maxDisappeared phai >= 0")
        if maxDistance < 0:
            raise ValueError("maxDistance phai >= 0")

        # ID se duoc tang dan moi khi dang ky doi tuong moi
        self.nextObjectID = 0

        # objects[objectID] = (cX, cY)
        self.objects = OrderedDict()

        # disappeared[objectID] = so frame lien tiep khong thay doi tuong
        self.disappeared = OrderedDict()

        # Cau hinh tracker
        self.maxDisappeared = int(maxDisappeared)
        self.maxDistance = float(maxDistance)

    def register(self, centroid):
        # Dang ky doi tuong moi va reset bo dem mat tich
        centroid = self._as_centroid_tuple(centroid)
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Xoa doi tuong khoi tracker
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def reset(self):
        # Xoa toan bo trang thai tracker, dung khi doi video/nguon
        self.nextObjectID = 0
        self.objects.clear()
        self.disappeared.clear()

    def _mark_missing(self, objectID):
        # Tang bo dem mat tich; neu vuot nguong thi xoa doi tuong
        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.maxDisappeared:
            self.deregister(objectID)

    @staticmethod
    def _as_centroid_tuple(centroid):
        # Chuan hoa centroid thanh tuple int de de so sanh / ve
        return int(centroid[0]), int(centroid[1])

    @staticmethod
    def _build_input_centroids(rects):
        # Chuyen danh sach bbox (x1, y1, x2, y2) thanh mang centroid [N, 2]
        input_centroids = np.zeros((len(rects), 2), dtype=np.int32)
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
        return input_centroids

    def update(self, rects):
        """
        Cap nhat tracker voi cac bbox cua frame hien tai.

        Dau vao:
        - rects: danh sach bbox, moi bbox co dang (x1, y1, x2, y2)

        Dau ra:
        - OrderedDict {objectID: (cX, cY)}
        """
        if rects is None:
            rects = []
        else:
            rects = list(rects)

        # Khong co bbox nao: danh dau tat ca doi tuong la mat tam thoi
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self._mark_missing(objectID)
            return self.objects

        inputCentroids = self._build_input_centroids(rects)

        # Chua co doi tuong nao -> dang ky tat ca centroid moi
        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
            return self.objects

        # Lay danh sach ID va centroid dang duoc theo doi
        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()), dtype=np.int32)

        # Ma tran khoang cach giua centroid cu va centroid moi
        D = dist.cdist(objectCentroids, inputCentroids)

        # Sap xep hang theo khoang cach nho nhat (uu tien ghep cap gan nhat)
        rows = D.min(axis=1).argsort()

        # Cot ung voi khoang cach nho nhat cua tung hang
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for row, col in zip(rows, cols):
            # Bo qua neu hang/cot nay da duoc ghep truoc do
            if row in usedRows or col in usedCols:
                continue

            # Qua xa -> khong ghep
            if D[row, col] > self.maxDistance:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = self._as_centroid_tuple(inputCentroids[col])
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Xac dinh cac hang/cot chua duoc ghep
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)

        # Nhieu doi tuong cu hon centroid moi -> mot so doi tuong co the da mat
        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                self._mark_missing(objectIDs[row])
        else:
            # Nhieu centroid moi hon doi tuong cu -> dang ky doi tuong moi
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects
