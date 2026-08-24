"""
Constants for Urban Traffic and Evade Autonomous Driving.
"""

DETECTION_CLASSES = [
    'green_light',
    'red_light',
    'turn_left_sign',
    'turn_right_sign',
    'stop_sign',
    'crosswalk'
]

NUM_WAYPOINTS = 5
ROUTE_COMMANDS = ['STRAIGHT', 'LEFT', 'RIGHT']
COMMAND_TO_INDEX = {'STRAIGHT': 0, 'LEFT': 1, 'RIGHT': 2}
