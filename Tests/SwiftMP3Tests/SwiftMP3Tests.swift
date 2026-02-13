import Foundation
import Testing
@testable import SwiftMP3

@Test func encodeSilence() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Generate 1152 stereo samples of silence (one full frame)
  let silence = [Float](repeating: 0, count: 1152 * 2)
  let data = state.encode(samples:silence)

  // One full frame should produce output
  #expect(data.count > 0)

  // Verify MP3 sync word (0xFFF)
  #expect(data[0] == 0xFF)
  #expect(data[1] & 0xE0 == 0xE0)
}

@Test func encodeMono() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .mono)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  let silence = [Float](repeating: 0, count: 1152)
  let data = state.encode(samples:silence)

  #expect(data.count > 0)
  #expect(data[0] == 0xFF)
}

@Test func flushProducesFinalFrame() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Feed less than a full frame
  let partial = [Float](repeating: 0, count: 500)
  let appendData = state.encode(samples:partial)
  #expect(appendData.count == 0) // Not enough for a frame yet

  let flushData = state.flush()
  #expect(flushData.count > 0) // Flush should pad and produce a frame
}

@Test func xingHeader() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  let samples = [Float](repeating: 0, count: 1152 * 2)
  _ = state.encode(samples:samples)

  let xing = state.generateXingHeader()
  #expect(xing.count > 0)

  // Xing header should also start with MP3 sync word
  #expect(xing[0] == 0xFF)
  #expect(xing[1] & 0xE0 == 0xE0)
}

@Test func optionsDefaults() {
  let options = MP3EncoderOptions()
  #expect(options.sampleRate == 44_100)
  #expect(options.bitrateKbps == 128)
  #expect(options.vbr == false)
  #expect(options.mode == .stereo)
  #expect(options.quality == 5)
}

@Test func encodeSineWave() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Generate a 440Hz sine wave, one frame of stereo samples
  var samples = [Float]()
  for i in 0..<1152 {
    let t = Float(i) / 44_100.0
    let value = sin(2.0 * .pi * 440.0 * t) * 0.5
    samples.append(value) // Left
    samples.append(value) // Right
  }

  let data = state.encode(samples:samples)
  #expect(data.count > 0)
}

@Test func encodeAsyncStream() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  // Create an AsyncStream of sine wave chunks
  let source = AsyncStream<[Float]> { continuation in
    for chunkIndex in 0..<4 {
      var samples = [Float]()
      for i in 0..<1152 {
        let sampleIndex = chunkIndex * 1152 + i
        let t = Float(sampleIndex) / 44_100.0
        let value = sin(2.0 * .pi * 440.0 * t) * 0.5
        samples.append(value) // Left
        samples.append(value) // Right
      }
      continuation.yield(samples)
    }
    continuation.finish()
  }

  var totalBytes = 0
  for try await frame in encoder.encode(source) {
    #expect(frame.count > 0)
    // Each yielded chunk should start with the MP3 sync word
    #expect(frame[0] == 0xFF)
    #expect(frame[1] & 0xE0 == 0xE0)
    totalBytes += frame.count
  }

  #expect(totalBytes > 0)
}

@Test func encodeToFile() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  let url = FileManager.default.temporaryDirectory.appendingPathComponent("test_\(UUID().uuidString).mp3")
  defer { try? FileManager.default.removeItem(at: url) }

  // Create an AsyncStream of sine wave chunks
  let source = AsyncStream<[Float]> { continuation in
    for chunkIndex in 0..<4 {
      var samples = [Float]()
      for i in 0..<1152 {
        let sampleIndex = chunkIndex * 1152 + i
        let t = Float(sampleIndex) / 44_100.0
        let value = sin(2.0 * .pi * 440.0 * t) * 0.5
        samples.append(value)
        samples.append(value)
      }
      continuation.yield(samples)
    }
    continuation.finish()
  }

  try await encoder.encode(source, to: url)

  let data = try Data(contentsOf: url)
  #expect(data.count > 0)

  // File should start with MP3 sync word (Xing header frame)
  #expect(data[0] == 0xFF)
  #expect(data[1] & 0xE0 == 0xE0)

  // Verify Xing/Info tag is present in the first frame
  let infoTag = [UInt8]("Info".utf8)
  let xingTag = [UInt8]("Xing".utf8)
  let prefix = [UInt8](data.prefix(256))
  let containsTag = prefix.indices.contains { i in
    i + 4 <= prefix.count && (Array(prefix[i..<i+4]) == infoTag || Array(prefix[i..<i+4]) == xingTag)
  }
  #expect(containsTag)
}

@Test func encodeEmptyStream() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  let source = AsyncStream<[Float]> { continuation in
    continuation.finish()
  }

  var frameCount = 0
  for try await _ in encoder.encode(source) {
    frameCount += 1
  }

  #expect(frameCount == 0)
}

// MARK: - ID3 Tag Tests

@Test func id3TagGeneration() {
  let tag = ID3Tag(title: "Test Song", artist: "Test Artist", album: "Test Album")
  let options = MP3EncoderOptions(id3Tag: tag)
  let encoder = MP3Encoder(options: options)
  let session = encoder.newSession()

  let data = session.generateID3Tag()
  #expect(data.count > 0)

  // Should start with "ID3"
  #expect(data[0] == 0x49) // 'I'
  #expect(data[1] == 0x44) // 'D'
  #expect(data[2] == 0x33) // '3'

  // Version should be 2.3
  #expect(data[3] == 0x03)
  #expect(data[4] == 0x00)

  // Verify frame IDs are present in the tag data
  let bytes = [UInt8](data)
  let containsTIT2 = bytes.indices.contains { i in
    i + 4 <= bytes.count && bytes[i] == 0x54 && bytes[i+1] == 0x49 && bytes[i+2] == 0x54 && bytes[i+3] == 0x32
  }
  let containsTPE1 = bytes.indices.contains { i in
    i + 4 <= bytes.count && bytes[i] == 0x54 && bytes[i+1] == 0x50 && bytes[i+2] == 0x45 && bytes[i+3] == 0x31
  }
  let containsTALB = bytes.indices.contains { i in
    i + 4 <= bytes.count && bytes[i] == 0x54 && bytes[i+1] == 0x41 && bytes[i+2] == 0x4C && bytes[i+3] == 0x42
  }
  #expect(containsTIT2)
  #expect(containsTPE1)
  #expect(containsTALB)
}

@Test func encodeToFileWithID3() async throws {
  let tag = ID3Tag(title: "My Song", artist: "Artist", album: "Album")
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo, id3Tag: tag)
  let encoder = MP3Encoder(options: options)

  let url = FileManager.default.temporaryDirectory.appendingPathComponent("test_id3_\(UUID().uuidString).mp3")
  defer { try? FileManager.default.removeItem(at: url) }

  let source = AsyncStream<[Float]> { continuation in
    for chunkIndex in 0..<4 {
      var samples = [Float]()
      for i in 0..<1152 {
        let sampleIndex = chunkIndex * 1152 + i
        let t = Float(sampleIndex) / 44_100.0
        let value = sin(2.0 * .pi * 440.0 * t) * 0.5
        samples.append(value)
        samples.append(value)
      }
      continuation.yield(samples)
    }
    continuation.finish()
  }

  try await encoder.encode(source, to: url)

  let data = try Data(contentsOf: url)
  #expect(data.count > 0)

  // File should start with "ID3"
  #expect(data[0] == 0x49) // 'I'
  #expect(data[1] == 0x44) // 'D'
  #expect(data[2] == 0x33) // '3'

  // Parse the ID3 tag size to find where audio starts
  let tagSize = (Int(data[6]) << 21) | (Int(data[7]) << 14) | (Int(data[8]) << 7) | Int(data[9])
  let audioStart = 10 + tagSize

  // Xing/Info header should follow the ID3 tag
  let xingPrefix = [UInt8](data[audioStart..<min(audioStart + 256, data.count)])
  let infoTag = [UInt8]("Info".utf8)
  let xingTag = [UInt8]("Xing".utf8)
  let containsXing = xingPrefix.indices.contains { i in
    i + 4 <= xingPrefix.count && (Array(xingPrefix[i..<i+4]) == infoTag || Array(xingPrefix[i..<i+4]) == xingTag)
  }
  #expect(containsXing)
}

@Test func id3TagWithAlbumArt() {
  // Create a small fake JPEG (just the header bytes for testing)
  let fakeJPEG = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10] + Array(repeating: UInt8(0), count: 100))
  let tag = ID3Tag(title: "Art Track", albumArt: fakeJPEG, albumArtMIMEType: "image/jpeg")
  let data = ID3TagWriter.build(tag: tag)

  #expect(data.count > 0)
  #expect(data[0] == 0x49) // 'I'

  // Verify APIC frame is present
  let bytes = [UInt8](data)
  let containsAPIC = bytes.indices.contains { i in
    i + 4 <= bytes.count && bytes[i] == 0x41 && bytes[i+1] == 0x50 && bytes[i+2] == 0x49 && bytes[i+3] == 0x43
  }
  #expect(containsAPIC)

  // The tag should contain the image data
  #expect(data.count > fakeJPEG.count)
}

@Test func id3EmptyFields() {
  let tag = ID3Tag()
  let data = ID3TagWriter.build(tag: tag)
  #expect(data.isEmpty)

  // EncoderSession with no tag also returns empty
  let options = MP3EncoderOptions()
  let encoder = MP3Encoder(options: options)
  let session = encoder.newSession()
  #expect(session.generateID3Tag().isEmpty)
}
