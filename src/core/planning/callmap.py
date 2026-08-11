import requests
import json
import os

# ================= CẤU HÌNH =================
DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
url = f"{DOMAIN}/api/maps/get_active_map/"
token = "bd4ebd21586b71713fe4b114d18c7093"   # ⚠️ Thay token đúng của bạn
map_type = "map_a"               # Hoặc map_a, map_b
map_file = os.path.join(os.path.dirname(__file__), "../utils/map.json")
# ============================================

def fetch_and_save_map():
    try:
        response = requests.get(url, params={"token": token, "map_type": map_type})
        response.raise_for_status()
        data = response.json()

        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Đã cập nhật bản đồ vào: {map_file}")
        return True

    except Exception as e:
        print(f"❌ Lỗi khi fetch map: {e}")
        return False

if __name__ == "__main__":
    if fetch_and_save_map():
        print("✅ Map updated. Khi chạy ROS sẽ dùng file map.json mới.")
