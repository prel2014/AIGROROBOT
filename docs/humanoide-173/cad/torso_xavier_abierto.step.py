from interior_xavier import find_torso, interior, original


def gen_step():
    return interior(find_torso(original()), covers=False)

