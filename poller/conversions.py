import struct

def to_float32_be(registers):
    if registers is None or len(registers) != 2:
        return None
    try:
        b = struct.pack('>HH', registers[0], registers[1])
        return struct.unpack('>f', b)[0]
    except Exception:
        return None

def to_float64_be(registers):
    if registers is None or len(registers) != 4:
        return None
    try:
        b = struct.pack('>HHHH', registers[0], registers[1], registers[2], registers[3])
        return struct.unpack('>d', b)[0]
    except Exception:
        return None