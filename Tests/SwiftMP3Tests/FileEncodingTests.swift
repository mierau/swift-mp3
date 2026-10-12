import Foundation
import Testing
@testable import SwiftMP3

/// Writes a WAV file with the given format tag and bits per sample.
private func writeWAV(
  _ channels: [[Float]], sampleRate: Int, formatTag: UInt16, bits: Int, extensible: Bool = false
) throws -> URL {
  func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] { withUnsafeBytes(of: value.littleEndian, Array.init) }
  let blockAlign = channels.count * bits / 8
  var samples: [UInt8] = []
  for i in 0..<channels[0].count {
    for channel in channels {
      let x = Double(channel[i])
      switch (formatTag, bits) {
      case (1, 8): samples.append(UInt8(clamping: Int((x * 128).rounded()) + 128))
      case (1, 16): samples += le(Int16(clamping: Int((x * 32_768).rounded())))
      case (1, 24): samples += le(Int32(clamping: Int((x * 8_388_608).rounded()))).prefix(3)
      case (1, 32): samples += le(Int32(clamping: Int((x * 2_147_483_648).rounded())))
      case (3, 32): samples += le(Float(x).bitPattern)
      default: samples += le(x.bitPattern)
      }
    }
  }
  var fmt: [UInt8] = le(extensible ? UInt16(0xFFFE) : formatTag)
  fmt += le(UInt16(channels.count))
  fmt += le(UInt32(sampleRate))
  fmt += le(UInt32(sampleRate * blockAlign))
  fmt += le(UInt16(blockAlign))
  fmt += le(UInt16(bits))
  if extensible {
    // cbSize, valid bits, channel mask, then the subformat GUID.
    fmt += le(UInt16(22))
    fmt += le(UInt16(bits))
    fmt += le(UInt32(0))
    fmt += le(formatTag)
    let guid: [UInt8] = [0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]
    fmt += guid
  }
  var file: [UInt8] = Array("RIFF".utf8)
  file += le(UInt32(4 + 8 + fmt.count + 8 + 10 + 8 + samples.count))
  file += Array("WAVE".utf8)
  file += Array("fmt ".utf8)
  file += le(UInt32(fmt.count))
  file += fmt
  // An odd-sized chunk, padded to an even size.
  file += Array("LIST".utf8)
  file += le(UInt32(9))
  file += Array("INFOabcde\0".utf8)
  file += Array("data".utf8)
  file += le(UInt32(samples.count))
  file += samples
  let url = FileManager.default.temporaryDirectory.appendingPathComponent("wav_\(UUID().uuidString).wav")
  try Data(file).write(to: url)
  return url
}

private func tone(_ frequency: Double, rate: Int, count: Int, amplitude: Double = 0.5) -> [Float] {
  (0..<count).map { Float(sin(2 * Double.pi * frequency * Double($0) / Double(rate)) * amplitude) }
}

@Test(arguments: [(UInt16(1), 8, false), (1, 16, false), (1, 24, false), (1, 32, false), (3, 32, false), (3, 64, false), (1, 16, true), (3, 32, true)])
func wavReaderReadsFormats(formatTag: UInt16, bits: Int, extensible: Bool) throws {
  let left = tone(440, rate: 32_000, count: 40_000)
  let right = tone(660, rate: 32_000, count: 40_000)
  let url = try writeWAV([left, right], sampleRate: 32_000, formatTag: formatTag, bits: bits, extensible: extensible)
  defer { try? FileManager.default.removeItem(at: url) }

  var reader = try WAVReader(url: url)
  #expect(reader.sampleRate == 32_000)
  #expect(reader.channelCount == 2)
  #expect(reader.frameCount == 40_000)
  var read: [[Float]] = [[], []]
  while true {
    let chunk = try reader.read(frames: 7_000)
    if chunk[0].isEmpty { break }
    read[0] += chunk[0]
    read[1] += chunk[1]
  }
  let tolerance: Float = bits == 8 ? 1 / 64 : 1e-4
  #expect(read[0].count == 40_000)
  #expect(zip(read[0], left).allSatisfy { abs($0 - $1) <= tolerance })
  #expect(zip(read[1], right).allSatisfy { abs($0 - $1) <= tolerance })
}

@Test func wavReaderRejectsOtherFiles() throws {
  let url = FileManager.default.temporaryDirectory.appendingPathComponent("notwav_\(UUID().uuidString).wav")
  try Data("not a wav file at all".utf8).write(to: url)
  defer { try? FileManager.default.removeItem(at: url) }
  #expect(throws: MP3EncoderError.unsupportedAudioFile("not a WAV file")) { try WAVReader(url: url) }

  let adpcm = try writeWAV([[0, 0]], sampleRate: 8_000, formatTag: 2, bits: 16)
  defer { try? FileManager.default.removeItem(at: adpcm) }
  #expect(throws: MP3EncoderError.self) { try WAVReader(url: adpcm) }
}

@Test(.enabled(if: TestDecoder.isAvailable), arguments: [1, 2])
func encodesFileWithProgress(channels: Int) async throws {
  let rate = 48_000
  let input = (0..<channels).map { tone(440 * Double($0 + 1), rate: rate, count: rate * 2) }
  let source = try writeWAV(input, sampleRate: rate, formatTag: 1, bits: 16)
  let output = FileManager.default.temporaryDirectory.appendingPathComponent("file_\(UUID().uuidString).mp3")
  defer {
    try? FileManager.default.removeItem(at: source)
    try? FileManager.default.removeItem(at: output)
  }

  final class Fractions: @unchecked Sendable {
    var values: [Double] = []
  }
  let fractions = Fractions()
  // The configured input rate and mode are replaced by the file's.
  let encoder = try MP3Encoder(options: MP3EncoderOptions(sampleRate: 8_000, mode: .jointStereo))
  try await encoder.encode(contentsOf: source, to: output) { fractions.values.append($0) }

  #expect(fractions.values.last == 1)
  #expect(fractions.values == fractions.values.sorted())
  let decoded = try TestDecoder.decode(try Data(contentsOf: output))
  #expect(decoded.sampleRate == rate)
  #expect(decoded.channels.count == channels)
}

@Test func fileEncodeRejectsSurroundForStereo() async throws {
  let input = [[Float]](repeating: tone(440, rate: 44_100, count: 4_410), count: 6)
  let source = try writeWAV(input, sampleRate: 44_100, formatTag: 1, bits: 16)
  let output = FileManager.default.temporaryDirectory.appendingPathComponent("surround_\(UUID().uuidString).mp3")
  defer {
    try? FileManager.default.removeItem(at: source)
    try? FileManager.default.removeItem(at: output)
  }
  await #expect(throws: MP3EncoderError.self) {
    try await MP3Encoder(options: MP3EncoderOptions()).encode(contentsOf: source, to: output)
  }
  #expect(!FileManager.default.fileExists(atPath: output.path))

  // A mono MP3 averages them.
  try await MP3Encoder(options: MP3EncoderOptions(mode: .mono)).encode(contentsOf: source, to: output)
  #expect(FileManager.default.fileExists(atPath: output.path))
}
