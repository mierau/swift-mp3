import Foundation
import Testing
import AVFoundation
import Accelerate
@testable import SwiftMP3

@Test func encodeSilence() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Generate 1152 stereo samples of silence (one full frame)
  let silence = [Float](repeating: 0, count: 1152 * 2)
  var data = state.encode(samples: silence)
  data.append(state.flush())

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
  var data = state.encode(samples: silence)
  data.append(state.flush())

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
  _ = state.encode(samples: samples)
  _ = state.flush()

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

  var data = state.encode(samples: samples)
  data.append(state.flush())
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

// MARK: - Bit Reservoir Tests

@Test func bitReservoirMainDataBegin() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Encode 12 frames of a low-amplitude sine wave — produces Huffman data
  // that is smaller than the slot, building up the reservoir
  var allData = Data()
  for frameIdx in 0..<12 {
    var samples = [Float]()
    for i in 0..<1152 {
      let sampleIndex = frameIdx * 1152 + i
      let t = Float(sampleIndex) / 44_100.0
      let value = sin(2.0 * .pi * 440.0 * t) * 0.1
      samples.append(value) // Left
      samples.append(value) // Right
    }
    allData.append(state.encode(samples: samples))
  }
  allData.append(state.flush())

  // Parse frames and check main_data_begin in later frames
  var offset = 0
  var frameIndex = 0
  var foundNonZeroMainDataBegin = false

  while offset + 6 < allData.count {
    guard allData[offset] == 0xFF && (allData[offset + 1] & 0xE0) == 0xE0 else {
      offset += 1
      continue
    }

    // main_data_begin is the first 9 bits of side info (byte 4)
    let sideInfoStart = offset + 4
    if sideInfoStart + 1 < allData.count {
      let mainDataBegin = (Int(allData[sideInfoStart]) << 1) | (Int(allData[sideInfoStart + 1]) >> 7)
      if frameIndex > 0 && mainDataBegin > 0 {
        foundNonZeroMainDataBegin = true
      }
    }

    // Calculate frame size to advance
    let bitrateIndex = (Int(allData[offset + 2]) >> 4) & 0x0F
    let sampleRateIndex = (Int(allData[offset + 2]) >> 2) & 0x03
    let paddingBit = (Int(allData[offset + 2]) >> 1) & 0x01
    let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    let sampleRateTable = [44100, 48000, 32000, 0]
    let bitrateValue = bitrateTable[bitrateIndex]
    let sampleRateValue = sampleRateTable[sampleRateIndex]
    guard bitrateValue > 0 && sampleRateValue > 0 else { break }
    let frameSize = (144 * bitrateValue * 1000) / sampleRateValue + paddingBit

    offset += frameSize
    frameIndex += 1
  }

  #expect(foundNonZeroMainDataBegin, "Later frames should have main_data_begin > 0 with reservoir")
}

@Test func framePadding() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Encode 100 frames — at 128kbps/44.1kHz we expect a mix of 417 and 418-byte frames
  var allData = Data()
  for _ in 0..<100 {
    let silence = [Float](repeating: 0, count: 1152 * 2)
    allData.append(state.encode(samples: silence))
  }
  allData.append(state.flush())

  // Parse frames and collect sizes
  var frameSizes = [Int]()
  var offset = 0

  while offset + 4 < allData.count {
    guard allData[offset] == 0xFF && (allData[offset + 1] & 0xE0) == 0xE0 else {
      offset += 1
      continue
    }

    let bitrateIndex = (Int(allData[offset + 2]) >> 4) & 0x0F
    let sampleRateIndex = (Int(allData[offset + 2]) >> 2) & 0x03
    let paddingBit = (Int(allData[offset + 2]) >> 1) & 0x01
    let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    let sampleRateTable = [44100, 48000, 32000, 0]
    let bitrateValue = bitrateTable[bitrateIndex]
    let sampleRateValue = sampleRateTable[sampleRateIndex]
    guard bitrateValue > 0 && sampleRateValue > 0 else { break }
    let frameSize = (144 * bitrateValue * 1000) / sampleRateValue + paddingBit

    frameSizes.append(frameSize)
    offset += frameSize
  }

  let has417 = frameSizes.contains(417)
  let has418 = frameSizes.contains(418)
  #expect(has417, "Should have 417-byte frames (no padding)")
  #expect(has418, "Should have 418-byte frames (with padding)")
}

// MARK: - One-Frame Delay Tests

@Test func oneFrameDelayFirstEncodeReturnsEmpty() {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // The very first full frame should return empty (buffered, not emitted)
  let samples = [Float](repeating: 0, count: 1152 * 2)
  let firstResult = state.encode(samples: samples)
  #expect(firstResult.isEmpty, "First encode should return empty due to one-frame buffering")

  // The second full frame should emit the first frame
  let secondResult = state.encode(samples: samples)
  #expect(secondResult.count > 0, "Second encode should emit the first buffered frame")
  #expect(secondResult[0] == 0xFF)
  #expect(secondResult[1] & 0xE0 == 0xE0)
}

@Test func flushEmitsBufferedFrameWithEmptyPCM() {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Feed exactly one full frame — pcmBuffer will be empty but frame is buffered
  let samples = [Float](repeating: 0, count: 1152 * 2)
  let encoded = state.encode(samples: samples)
  #expect(encoded.isEmpty)

  // flush() should emit the buffered frame even though pcmBuffer is empty
  let flushed = state.flush()
  #expect(flushed.count > 0)
  #expect(flushed[0] == 0xFF)
  #expect(flushed[1] & 0xE0 == 0xE0)
}

@Test func doubleFlushReturnsEmpty() {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  let samples = [Float](repeating: 0, count: 1152 * 2)
  _ = state.encode(samples: samples)

  let first = state.flush()
  #expect(first.count > 0)

  let second = state.flush()
  #expect(second.isEmpty, "Second flush should return empty — nothing left to emit")
}

@Test func frameCountAndByteCountAccuracy() {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  let frameTarget = 10
  var totalOutput = Data()
  for _ in 0..<frameTarget {
    let samples = [Float](repeating: 0, count: 1152 * 2)
    totalOutput.append(state.encode(samples: samples))
  }
  totalOutput.append(state.flush())

  #expect(state.encodedFrameCount == UInt32(frameTarget))
  #expect(state.encodedByteCount == UInt32(totalOutput.count))

  // At 128kbps/44.1kHz, each frame is 417 or 418 bytes
  let averageFrameSize = Double(totalOutput.count) / Double(frameTarget)
  #expect(averageFrameSize >= 417.0 && averageFrameSize <= 418.0,
    "Average frame size should be ~417-418 bytes, got \(averageFrameSize)")
}

@Test func finalFrameHasZeroMainDataBegin() {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Encode several frames of a sine wave to build reservoir state
  for frameIdx in 0..<6 {
    var samples = [Float]()
    for i in 0..<1152 {
      let t = Float(frameIdx * 1152 + i) / 44_100.0
      let value = sin(2.0 * .pi * 440.0 * t) * 0.3
      samples.append(value)
      samples.append(value)
    }
    _ = state.encode(samples: samples)
  }

  // flush() produces the last frame — feed partial samples to trigger isFinal path
  let partial = [Float](repeating: 0.1, count: 500)
  _ = state.encode(samples: partial)
  let flushed = state.flush()

  // The very last frame in the flushed data should have main_data_begin = 0
  // Find the last frame by scanning
  let bytes = [UInt8](flushed)
  var lastFrameOffset = -1
  var offset = 0
  while offset + 4 < bytes.count {
    if bytes[offset] == 0xFF && (bytes[offset + 1] & 0xE0) == 0xE0 {
      lastFrameOffset = offset
      // Advance by frame size
      let bitrateIndex = (Int(bytes[offset + 2]) >> 4) & 0x0F
      let sampleRateIndex = (Int(bytes[offset + 2]) >> 2) & 0x03
      let paddingBit = (Int(bytes[offset + 2]) >> 1) & 0x01
      let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
      let sampleRateTable = [44100, 48000, 32000, 0]
      let bv = bitrateTable[bitrateIndex]
      let sr = sampleRateTable[sampleRateIndex]
      guard bv > 0 && sr > 0 else { break }
      offset += (144 * bv * 1000) / sr + paddingBit
    } else {
      offset += 1
    }
  }

  #expect(lastFrameOffset >= 0, "Should find at least one frame")
  if lastFrameOffset >= 0 && lastFrameOffset + 5 < bytes.count {
    let mainDataBegin = (Int(bytes[lastFrameOffset + 4]) << 1) | (Int(bytes[lastFrameOffset + 5]) >> 7)
    #expect(mainDataBegin == 0, "Final frame should have main_data_begin = 0, got \(mainDataBegin)")
  }
}

@Test func monoReservoirWorks() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .mono)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  // Mono has 17-byte side info (vs 32 stereo), giving a larger main data slot
  var allData = Data()
  for frameIdx in 0..<8 {
    var samples = [Float]()
    for i in 0..<1152 {
      let t = Float(frameIdx * 1152 + i) / 44_100.0
      samples.append(sin(2.0 * .pi * 440.0 * t) * 0.2)
    }
    allData.append(state.encode(samples: samples))
  }
  allData.append(state.flush())

  #expect(allData.count > 0)
  #expect(state.encodedFrameCount == 8)

  // Verify frames are valid
  #expect(allData[0] == 0xFF)
  #expect(allData[1] & 0xE0 == 0xE0)

  // Check the channel mode bits indicate mono (bits 6-7 of byte 3 = 0b11)
  let modeBits = (Int(allData[3]) >> 6) & 0x03
  #expect(modeBits == 3, "Mode bits should be 0b11 for mono, got \(modeBits)")
}

@Test func multiFrameOutputContiguousFrames() {
  // Verify output is a sequence of contiguous valid frames with no gaps
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  var allData = Data()
  for frameIdx in 0..<20 {
    var samples = [Float]()
    for i in 0..<1152 {
      let t = Float(frameIdx * 1152 + i) / 44_100.0
      let value = sin(2.0 * .pi * 440.0 * t) * 0.5
      samples.append(value)
      samples.append(value)
    }
    allData.append(state.encode(samples: samples))
  }
  allData.append(state.flush())

  // Walk the output frame-by-frame — every byte should be accounted for
  var offset = 0
  var parsedFrames = 0
  let bytes = [UInt8](allData)

  while offset + 4 <= bytes.count {
    #expect(bytes[offset] == 0xFF, "Frame \(parsedFrames) at offset \(offset): bad sync byte 0")
    #expect(bytes[offset + 1] & 0xE0 == 0xE0, "Frame \(parsedFrames) at offset \(offset): bad sync byte 1")

    let bitrateIndex = (Int(bytes[offset + 2]) >> 4) & 0x0F
    let sampleRateIndex = (Int(bytes[offset + 2]) >> 2) & 0x03
    let paddingBit = (Int(bytes[offset + 2]) >> 1) & 0x01
    let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    let sampleRateTable = [44100, 48000, 32000, 0]
    let bv = bitrateTable[bitrateIndex]
    let sr = sampleRateTable[sampleRateIndex]

    #expect(bv > 0, "Frame \(parsedFrames): invalid bitrate index \(bitrateIndex)")
    #expect(sr > 0, "Frame \(parsedFrames): invalid sample rate index \(sampleRateIndex)")
    guard bv > 0 && sr > 0 else { break }

    let frameSize = (144 * bv * 1000) / sr + paddingBit
    offset += frameSize
    parsedFrames += 1
  }

  // All bytes should be consumed (no trailing garbage)
  #expect(offset == bytes.count, "Output has \(bytes.count - offset) trailing bytes after \(parsedFrames) frames")
  #expect(parsedFrames == 20, "Expected 20 frames, parsed \(parsedFrames)")
}

// MARK: - Helpers

/// Encodes a sine wave and returns complete MP3 file data with Xing header.
private func makeTestMP3(
  sampleRate: Int = 44_100,
  bitrateKbps: Int = 128,
  mode: MP3EncoderOptions.Mode = .stereo,
  amplitude: Float = 0.5,
  frameCount: Int = 20
) -> Data {
  let options = MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: bitrateKbps, mode: mode)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()
  let channels = mode == .mono ? 1 : 2

  var mp3Frames = Data()
  for frameIdx in 0..<frameCount {
    var samples = [Float]()
    for i in 0..<1152 {
      let sampleIndex = frameIdx * 1152 + i
      let t = Float(sampleIndex) / Float(sampleRate)
      let value = sin(2.0 * .pi * 440.0 * t) * amplitude
      for _ in 0..<channels {
        samples.append(value)
      }
    }
    mp3Frames.append(state.encode(samples: samples))
  }
  mp3Frames.append(state.flush())

  let xingHeader = state.generateXingHeader()
  return xingHeader + mp3Frames
}

/// Writes data to a temp .mp3 file and returns the URL.
private func writeTempMP3(_ data: Data) throws -> URL {
  let url = FileManager.default.temporaryDirectory
    .appendingPathComponent("test_\(UUID().uuidString).mp3")
  try data.write(to: url)
  return url
}

/// Decodes an MP3 file and returns the processing format and PCM buffer.
private func decodeMP3(at url: URL) throws -> (format: AVAudioFormat, buffer: AVAudioPCMBuffer) {
  let audioFile = try AVAudioFile(forReading: url)
  let format = audioFile.processingFormat
  let frameCount = AVAudioFrameCount(audioFile.length)
  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
  try audioFile.read(into: buffer)
  return (format, buffer)
}

// MARK: - AVFoundation Round-Trip Tests

@Test func avAudioFileCanDecodeOutput() throws {
  let mp3Data = makeTestMP3()
  let url = try writeTempMP3(mp3Data)
  defer { try? FileManager.default.removeItem(at: url) }

  let (format, buffer) = try decodeMP3(at: url)

  #expect(format.sampleRate == 44_100)
  #expect(format.channelCount == 2)
  #expect(buffer.frameLength > 0)
}

@Test func decodedSineWaveHasEnergy() throws {
  let mp3Data = makeTestMP3(amplitude: 0.5, frameCount: 10)
  let url = try writeTempMP3(mp3Data)
  defer { try? FileManager.default.removeItem(at: url) }

  let (_, buffer) = try decodeMP3(at: url)
  let sampleCount = Int(buffer.frameLength)
  let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: sampleCount))

  // Check peak amplitude — lossy compression, but 0.5 input should survive as > 0.05
  var peak: Float = 0
  vDSP_maxv(samples.map { abs($0) }, 1, &peak, vDSP_Length(sampleCount))
  #expect(peak > 0.05, "Decoded sine wave peak \(peak) is too low")

  // Check RMS energy
  var rms: Float = 0
  vDSP_rmsqv(samples, 1, &rms, vDSP_Length(sampleCount))
  #expect(rms > 0.01, "Decoded sine wave RMS \(rms) is too low")
}

@Test func decodedSilenceIsQuiet() throws {
  let mp3Data = makeTestMP3(amplitude: 0, frameCount: 10)
  let url = try writeTempMP3(mp3Data)
  defer { try? FileManager.default.removeItem(at: url) }

  let (_, buffer) = try decodeMP3(at: url)
  let sampleCount = Int(buffer.frameLength)
  let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: sampleCount))

  var peak: Float = 0
  vDSP_maxv(samples.map { abs($0) }, 1, &peak, vDSP_Length(sampleCount))
  #expect(peak < 0.05, "Decoded silence peak \(peak) is too high — expected near-zero")
}

@Test func decodedDurationIsCorrect() throws {
  let frameCount = 20
  let mp3Data = makeTestMP3(frameCount: frameCount)
  let url = try writeTempMP3(mp3Data)
  defer { try? FileManager.default.removeItem(at: url) }

  let audioFile = try AVAudioFile(forReading: url)
  let decodedFrames = Int(audioFile.length)

  // Expected: frameCount * 1152 samples, plus Xing header frame (1152 silent samples).
  // Allow tolerance for encoder/decoder delay (~1200 samples each direction).
  let expectedSamples = (frameCount + 1) * 1152
  let tolerance = 2400
  #expect(abs(decodedFrames - expectedSamples) < tolerance,
    "Decoded \(decodedFrames) samples, expected ~\(expectedSamples) ± \(tolerance)")
}

@Test func multipleConfigurationsDecodeSuccessfully() throws {
  let configs: [(sampleRate: Int, bitrate: Int, mode: MP3EncoderOptions.Mode, expectedChannels: UInt32)] = [
    (44_100, 128, .stereo, 2),
    (44_100, 128, .mono, 1),
    (48_000, 192, .stereo, 2),
    (32_000, 64, .stereo, 2),
    (44_100, 128, .jointStereo, 2),
  ]

  for config in configs {
    let mp3Data = makeTestMP3(
      sampleRate: config.sampleRate,
      bitrateKbps: config.bitrate,
      mode: config.mode,
      frameCount: 5
    )
    let url = try writeTempMP3(mp3Data)
    defer { try? FileManager.default.removeItem(at: url) }

    let (format, buffer) = try decodeMP3(at: url)

    #expect(format.sampleRate == Double(config.sampleRate),
      "\(config): sample rate mismatch")
    #expect(format.channelCount == config.expectedChannels,
      "\(config): channel count mismatch")
    #expect(buffer.frameLength > 0,
      "\(config): no decoded samples")
  }
}

@Test func decodedMonoHasOneChannel() throws {
  let mp3Data = makeTestMP3(mode: .mono, frameCount: 5)
  let url = try writeTempMP3(mp3Data)
  defer { try? FileManager.default.removeItem(at: url) }

  let (format, buffer) = try decodeMP3(at: url)
  #expect(format.channelCount == 1)

  let sampleCount = Int(buffer.frameLength)
  let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: sampleCount))

  var peak: Float = 0
  vDSP_maxv(samples.map { abs($0) }, 1, &peak, vDSP_Length(sampleCount))
  #expect(peak > 0.05, "Decoded mono sine wave should have energy")
}

// MARK: - Determinism and Invariant Tests

@Test func encodingIsDeterministic() {
  func encodeOnce() -> Data {
    let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
    let encoder = MP3Encoder(options: options)
    var state = encoder.newSession()

    var data = Data()
    for frameIdx in 0..<5 {
      var samples = [Float]()
      for i in 0..<1152 {
        let t = Float(frameIdx * 1152 + i) / 44_100.0
        let value = sin(2.0 * .pi * 440.0 * t) * 0.5
        samples.append(value)
        samples.append(value)
      }
      data.append(state.encode(samples: samples))
    }
    data.append(state.flush())
    return data
  }

  let first = encodeOnce()
  let second = encodeOnce()
  #expect(first == second, "Same input should produce identical output (\(first.count) vs \(second.count) bytes)")
}

@Test func paddingRatioMatchesTheory() {
  // At 128kbps/44.1kHz: remainder = (144 * 128 * 1000) % 44100 = 42300
  // Expected padded fraction = 42300 / 44100 ≈ 0.9592
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)
  var state = encoder.newSession()

  let total = 1000
  var allData = Data()
  for _ in 0..<total {
    let silence = [Float](repeating: 0, count: 1152 * 2)
    allData.append(state.encode(samples: silence))
  }
  allData.append(state.flush())

  // Count padded vs unpadded by inspecting the padding bit in each frame header
  var padded = 0
  var unpadded = 0
  var offset = 0
  let bytes = [UInt8](allData)

  while offset + 4 < bytes.count {
    guard bytes[offset] == 0xFF && (bytes[offset + 1] & 0xE0) == 0xE0 else {
      offset += 1
      continue
    }

    let paddingBit = (Int(bytes[offset + 2]) >> 1) & 0x01
    if paddingBit == 1 { padded += 1 } else { unpadded += 1 }

    let bitrateIndex = (Int(bytes[offset + 2]) >> 4) & 0x0F
    let sampleRateIndex = (Int(bytes[offset + 2]) >> 2) & 0x03
    let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    let sampleRateTable = [44100, 48000, 32000, 0]
    let bv = bitrateTable[bitrateIndex]
    let sr = sampleRateTable[sampleRateIndex]
    guard bv > 0 && sr > 0 else { break }
    offset += (144 * bv * 1000) / sr + paddingBit
  }

  let counted = padded + unpadded
  let ratio = Double(padded) / Double(counted)
  // Theoretical: 42300/44100 ≈ 0.9592, allow ±2%
  #expect(ratio > 0.93 && ratio < 0.98,
    "Padding ratio should be ~0.96, got \(String(format: "%.4f", ratio)) (\(padded)/\(counted))")
}
