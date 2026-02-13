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
