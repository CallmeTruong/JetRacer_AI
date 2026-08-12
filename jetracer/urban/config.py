# Urban Navigation System Configuration

DETECTION_CLASSES = [
    'green_light',     # Đèn xanh
    'red_light',       # Đèn đỏ
    'turn_left_sign',  # Biển rẽ trái
    'turn_right_sign', # Biển rẽ phải
    'stop_sign',       # Biển dừng (STOP)
    'crosswalk'        # Đường đi bộ (Pedestrian Crosswalk - xe phải dừng trước vạch này khi có đèn đỏ/biển STOP)
]

ROUTE_COMMANDS = ['LEFT', 'STRAIGHT', 'RIGHT']

COMMAND_TO_INDEX = {
    'LEFT': 0,
    'STRAIGHT': 1,
    'RIGHT': 2
}

INDEX_TO_COMMAND = {
    0: 'LEFT',
    1: 'STRAIGHT',
    2: 'RIGHT'
}

NUM_WAYPOINTS = 5  # Output 5 (x, y) trajectory waypoints
