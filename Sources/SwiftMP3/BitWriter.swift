// SwiftMP3
// BitWriter.swift

/// MSB-first bit writer.
struct BitWriter {
  private(set) var bytes: [UInt8] = []
  private var accumulator: UInt64 = 0
  private var pending = 0

  init(capacity: Int = 0) {
    bytes.reserveCapacity(capacity)
  }

  /// Total number of bits written.
  var bitCount: Int { bytes.count * 8 + pending }

  /// Writes the low `count` bits of `value` (at most 32).
  @inline(__always)
  mutating func write(_ value: UInt32, _ count: Int) {
    guard count > 0 else { return }
    accumulator = (accumulator << UInt64(count)) | UInt64(value & UInt32(truncatingIfNeeded: (UInt64(1) << UInt64(count)) - 1))
    pending += count
    while pending >= 8 {
      pending -= 8
      bytes.append(UInt8(truncatingIfNeeded: accumulator >> UInt64(pending)))
    }
  }

  @inline(__always)
  mutating func write(_ value: Int, _ count: Int) {
    write(UInt32(truncatingIfNeeded: value), count)
  }

  /// Pads with zero bits to the next byte boundary.
  mutating func alignToByte() {
    if pending > 0 {
      write(UInt32(0), 8 - pending)
    }
  }

  /// Removes and returns all complete bytes, keeping any partial byte.
  mutating func takeBytes() -> [UInt8] {
    let out = bytes
    bytes.removeAll(keepingCapacity: true)
    return out
  }
}
