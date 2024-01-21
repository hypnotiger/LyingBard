# https://stackoverflow.com/a/28666223
def number_to_base(number: int, base: int) -> list[int]:
    if number == 0:
        return [0]
    digits = []
    while number:
        digits.append(number % base)
        number //= base
    return digits[::-1]

def number_to_base_str(number: int, base_chars: str) -> str:
    digits = number_to_base(number, len(base_chars))
    return "".join([base_chars[digit] for digit in digits])

from string import ascii_uppercase, ascii_lowercase, digits
BASE64_CHARS = ascii_uppercase + ascii_lowercase + digits + "-_"
def int_to_base64(number: int): return number_to_base_str(number, BASE64_CHARS)

def base_str_to_number(base_str: str, base_chars: str) -> str:
    base = len(base_chars)
    return sum([base_chars.index(char) * base**i for i, char in enumerate(reversed(base_str))])

def base64_to_int(b64: str): return base_str_to_number(b64, BASE64_CHARS)