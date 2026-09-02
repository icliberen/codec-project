"""Small, readable RLE + Huffman implementation for coefficient streams.
TR: Entropy kodlama katsayı payload'ını sıkıştırır. / EN: Entropy coding compresses coefficient payloads.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint values must be non-negative")
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
        shift += 7
        if shift > 63:
            raise ValueError("Invalid varint")
    raise ValueError("Truncated varint")


def _zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def _unzigzag(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def rle_encode(values: list[int]) -> bytes:
    """Encode zero runs followed by non-zero values, then a trailing run."""
    output = bytearray()
    zero_count = 0
    for raw in values:
        value = int(raw)
        if value == 0:
            zero_count += 1
            continue
        output.extend(_varint(zero_count))
        output.extend(_varint(_zigzag(value) + 1))
        zero_count = 0
    output.extend(_varint(zero_count))
    output.extend(b"\x00")  # value code 0 terminates the stream
    return bytes(output)


def rle_decode(data: bytes, count: int) -> list[int]:
    if count < 0:
        raise ValueError("RLE coefficient count must be non-negative")
    output: list[int] = []
    position = 0
    while position < len(data) and len(output) <= count:
        zeros, position = _read_varint(data, position)
        value_code, position = _read_varint(data, position)
        if len(output) + zeros > count:
            raise ValueError("RLE stream exceeds declared coefficient count")
        output.extend([0] * zeros)
        if value_code == 0:
            break
        output.append(_unzigzag(value_code - 1))
    if len(output) != count:
        raise ValueError(f"RLE stream has {len(output)} values, expected {count}")
    return output


@dataclass(frozen=True)
class HuffmanEncoded:
    data: bytes
    bit_length: int
    frequencies: list[int]


def _codes(frequencies: list[int]) -> dict[int, str]:
    symbols = [(frequency, symbol) for symbol, frequency in enumerate(frequencies) if frequency]
    if not symbols:
        return {}
    if len(symbols) == 1:
        return {symbols[0][1]: "0"}
    heap: list[tuple[int, int, object]] = []
    serial = 0
    for frequency, symbol in symbols:
        heap.append((frequency, serial, symbol))
        serial += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (left_frequency + right_frequency, serial, (left, right)))
        serial += 1
    root = heap[0][2]
    result: dict[int, str] = {}

    def visit(node: object, prefix: str) -> None:
        if isinstance(node, int):
            result[node] = prefix or "0"
            return
        left, right = node
        visit(left, prefix + "0")
        visit(right, prefix + "1")

    visit(root, "")
    return result


def huffman_encode(data: bytes) -> HuffmanEncoded:
    frequencies = [0] * 256
    for byte in data:
        frequencies[byte] += 1
    codes = _codes(frequencies)
    output = bytearray()
    current = 0
    used = 0
    bit_length = 0
    for byte in data:
        for bit in codes[byte]:
            current = (current << 1) | (bit == "1")
            used += 1
            bit_length += 1
            if used == 8:
                output.append(current)
                current = 0
                used = 0
    if used:
        output.append(current << (8 - used))
    return HuffmanEncoded(bytes(output), bit_length, frequencies)


def huffman_decode(encoded: HuffmanEncoded) -> bytes:
    if encoded.bit_length < 0 or encoded.bit_length > len(encoded.data) * 8:
        raise ValueError("Invalid Huffman bit length")
    if len(encoded.frequencies) != 256 or any(int(value) < 0 for value in encoded.frequencies):
        raise ValueError("Invalid Huffman frequency table")
    if not any(encoded.frequencies):
        return b""
    codes = _codes(encoded.frequencies)
    reverse = {code: symbol for symbol, code in codes.items()}
    output = bytearray()
    code = ""
    for index in range(encoded.bit_length):
        byte = encoded.data[index // 8]
        code += "1" if (byte >> (7 - (index % 8))) & 1 else "0"
        if code in reverse:
            output.append(reverse[code])
            code = ""
    if code:
        raise ValueError("Invalid Huffman padding or truncated stream")
    return bytes(output)


def encode_coefficients(values: list[int]) -> HuffmanEncoded:
    return huffman_encode(rle_encode(values))


def decode_coefficients(encoded: HuffmanEncoded, count: int) -> list[int]:
    return rle_decode(huffman_decode(encoded), count)
