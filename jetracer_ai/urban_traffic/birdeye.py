import cv2
import numpy as np

class BirdEyeTransform:
    """
    Inverse Perspective Mapping (IPM) / Bird's Eye View (BEV) Transformer for JetRacer Urban Navigation.
    Converts front-facing perspective camera feed into top-down Bird's Eye View (BEV).
    """
    def __init__(
        self,
        src_top_w=0.45,    # Top width ratio of trapezoid (0.0 .. 1.0)
        src_bottom_w=0.90, # Bottom width ratio of trapezoid (0.0 .. 1.0)
        src_top_y=0.55,    # Top Y ratio of trapezoid (horizon cut off)
        src_bottom_y=0.95, # Bottom Y ratio of trapezoid
        dst_w_margin=0.20, # Destination margin ratio left/right
        image_shape=(224, 224)
    ):
        self.image_shape = image_shape
        self.h, self.w = image_shape[:2]
        
        self.src_top_w = src_top_w
        self.src_bottom_w = src_bottom_w
        self.src_top_y = src_top_y
        self.src_bottom_y = src_bottom_y
        self.dst_w_margin = dst_w_margin
        
        self.update_matrices()

    def update_matrices(self, src_top_w=None, src_bottom_w=None, src_top_y=None, src_bottom_y=None, dst_w_margin=None):
        if src_top_w is not None: self.src_top_w = src_top_w
        if src_bottom_w is not None: self.src_bottom_w = src_bottom_w
        if src_top_y is not None: self.src_top_y = src_top_y
        if src_bottom_y is not None: self.src_bottom_y = src_bottom_y
        if dst_w_margin is not None: self.dst_w_margin = dst_w_margin

        w, h = self.w, self.h

        # Source Trapezoid Points on raw perspective image
        top_left     = [w * (0.5 - self.src_top_w / 2.0), h * self.src_top_y]
        top_right    = [w * (0.5 + self.src_top_w / 2.0), h * self.src_top_y]
        bottom_right = [w * (0.5 + self.src_bottom_w / 2.0), h * self.src_bottom_y]
        bottom_left  = [w * (0.5 - self.src_bottom_w / 2.0), h * self.src_bottom_y]

        self.src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])

        # Destination Rectangle Points in Bird's Eye View
        dst_top_left     = [w * self.dst_w_margin, 0]
        dst_top_right    = [w * (1.0 - self.dst_w_margin), 0]
        dst_bottom_right = [w * (1.0 - self.dst_w_margin), h]
        dst_bottom_left  = [w * self.dst_w_margin, h]

        self.dst_pts = np.float32([dst_top_left, dst_top_right, dst_bottom_right, dst_bottom_left])

        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

    def transform(self, image):
        """Warp front perspective image into Bird's Eye View."""
        if image is None:
            return None
        if image.shape[0] != self.h or image.shape[1] != self.w:
            image = cv2.resize(image, (self.w, self.h))
        return cv2.warpPerspective(image, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, bev_image):
        """Unwarp Bird's Eye View image back to front perspective."""
        if bev_image is None:
            return None
        return cv2.warpPerspective(bev_image, self.M_inv, (self.w, self.h), flags=cv2.INTER_LINEAR)

    def transform_points(self, points):
        """Transform perspective 2D points (x, y) to Bird's Eye View points."""
        if not points:
            return []
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped_pts = cv2.perspectiveTransform(pts, self.M)
        return [tuple(map(int, pt[0])) for pt in warped_pts]

    def untransform_points(self, bev_points):
        """Transform Bird's Eye View 2D points (x, y) back to perspective points."""
        if not bev_points:
            return []
        pts = np.array(bev_points, dtype=np.float32).reshape(-1, 1, 2)
        unwarped_pts = cv2.perspectiveTransform(pts, self.M_inv)
        return [tuple(map(int, pt[0])) for pt in unwarped_pts]

    def draw_roi(self, image):
        """Draw the perspective trapezoid ROI on the raw image for calibration."""
        if image is None:
            return None
        overlay = image.copy()
        pts = self.src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        return overlay
