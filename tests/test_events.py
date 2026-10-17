from perception.events import visits

FLOWERS = [{"flower_id": "f1", "box": [100, 100, 160, 160]}, {"flower_id": "f2", "box": [500, 500, 560, 560]}]


def pt(t, x, y):
    return {"t": t, "center": [x, y], "box": [x - 12, y - 12, x + 12, y + 12], "interpolated": False}


def test_visit_needs_dwell_and_tolerates_gaps():
    points = [pt(0.0, 130, 130), pt(0.5, 132, 128), pt(1.0, 165, 130),   # in f1 (last one via the margin)
              pt(1.5, 300, 300),                                        # between flowers
              pt(2.0, 530, 530), pt(2.2, 530, 531),                     # f2, but only 0.2 s: a fly-over
              pt(5.0, 530, 530), pt(5.5, 531, 529), pt(7.0, 530, 530)]  # f2 again; 1.5 s gap splits it
    ev = visits(points, FLOWERS, "honeybee_1")
    assert ev == [
        {"track_id": "honeybee_1", "flower_id": "f1", "type": "visit", "t_start": 0.0, "t_end": 1.0},
        {"track_id": "honeybee_1", "flower_id": "f2", "type": "visit", "t_start": 5.0, "t_end": 5.5},
    ]


def test_no_flowers_no_events():
    assert visits([pt(0, 10, 10), pt(1, 10, 10)], [], "b") == []
