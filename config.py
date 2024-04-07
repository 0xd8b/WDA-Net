
class global_var:
    value = 61

def set_value(value):
    global_var.value = value

def get_value():
    return global_var.value