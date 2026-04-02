from enum import Enum

"""
탐색 라벨링 정의
"""
class DetectionLabel(str, Enum):
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    PASS = "PASS"

