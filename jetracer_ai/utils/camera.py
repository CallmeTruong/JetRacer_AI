import os
import cv2
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    rospy = None
    Image = None
    HAS_ROS = False


class CameraStream:
    """
    ROS Camera Subscriber with automatic BGR decoding
    and integrated debug video recording capability.
    Bo ton 100% logic gii m Compressed & Raw ROS image t jetracer-car.
    """

    def __init__(
        self,
        topic_name="/csi_cam_0/image_raw",
        width=500,
        height=300,
        record_video=False,
        output_path="recordings/jetracer_run.avi",
        fps=20,
    ):
        self.width = width
        self.height = height
        self.latest_image = None
        self.record_video = record_video
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None

        if HAS_ROS and rospy is not None:
            try:
                if hasattr(rospy, "core") and not rospy.core.is_initialized():
                    rospy.init_node("jetracer_camera_stream_node", anonymous=True)
                elif hasattr(rospy, "init_node"):
                    rospy.init_node("jetracer_camera_stream_node", anonymous=True)
            except Exception as e:
                print(f"Bypassing ROS node initialization: {e}")

            self.sub = rospy.Subscriber(
                topic_name, Image, self._camera_callback, queue_size=1
            )
            rospy.loginfo(f"📷 Subscribed to Camera Topic: {topic_name}")
        else:
            print(f"[*] Running CameraStream without ROS environment.")
            self.sub = None

        if self.record_video:
            self._init_video_writer()

    def _camera_callback(self, image_msg):
        try:
            encoding = image_msg.encoding.lower()

            if "compressed" in encoding:
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            else:
                if "rgb8" in encoding or "bgr8" in encoding:
                    channels = 3
                elif "rgba8" in encoding or "bgra8" in encoding:
                    channels = 4
                elif "mono8" in encoding:
                    channels = 1
                else:
                    channels = 3

                img_buf = np.frombuffer(image_msg.data, dtype=np.uint8)

                if image_msg.step > 0:
                    cv_image = img_buf.reshape(
                        image_msg.height, image_msg.step
                    )[:, : image_msg.width * channels]
                    cv_image = cv_image.reshape(
                        image_msg.height, image_msg.width, channels
                    )
                else:
                    cv_image = img_buf.reshape(
                        image_msg.height, image_msg.width, channels
                    )

                if "rgb8" in encoding:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                elif "rgba8" in encoding:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
                elif "bgra8" in encoding:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

            if cv_image is None or cv_image.size == 0:
                return

            self.latest_image = cv2.resize(
                cv_image, (self.width, self.height)
            )

            if self.record_video and self.video_writer is not None:
                self.video_writer.write(self.latest_image)

        except Exception as e:
            if rospy:
                rospy.logerr_throttle(2.0, f"❌ ROS image decoding error: {e}")

    def _init_video_writer(self):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            if rospy:
                rospy.loginfo(f"📹 Started video recording at: {self.output_path}")
        except Exception as e:
            if rospy:
                rospy.logerr(f"VideoWriter initialization error: {e}")

    def get_frame(self):
        if self.latest_image is None:
            return None
        return self.latest_image.copy()

    def release(self):
        if self.sub:
            self.sub.unregister()
        if self.video_writer is not None:
            self.video_writer.release()
            if rospy:
                rospy.loginfo("📹 Closed video recording stream.")
