from .misc_utils import base64_to_int, int_to_base64

class B64IDConverter:
    regex = "[A-z0-9-_]{1,}"

    def to_python(self, value):
        return base64_to_int(value)
    
    def to_url(self, value, id_length=4):
        return int_to_base64(value, id_length)