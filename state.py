import json


def get(name):
    try:
        with open("state.json", "r") as f:
            state = json.load(f)
        return state.get(name)
    except:
        return None


def set(name, value):
    try:
        with open("state.json", "r") as f:
            state = json.load(f)
    except:
        state = {}

    state[name] = value

    with open("state.json", "w") as f:
        json.dump(state, f)

    return value
