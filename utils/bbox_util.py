def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(point_1, point_2):
    return ((point_1[0]-point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5