import Foundation
import Testing
@testable import SwiftMP3

@Test func encodeSilence() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  let silence = [Float](repeating: 0, count: 1152)
  var data = state.encode(samples: silence)
  data.append(state.flush())

  #expect(data.count > 0)
  #expect(data[0] == 0xFF)
}

@Test func flushProducesFinalFrame() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)

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
  let encoder = try MP3Encoder(options: options)

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
  let encoder = try MP3Encoder(options: options)

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

@Test func id3TagGeneration() throws {
  let tag = ID3Tag(title: "Test Song", artist: "Test Artist", album: "Test Album")
  let options = MP3EncoderOptions(id3Tag: tag)
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)

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

@Test func id3EmptyFields() throws {
  let tag = ID3Tag()
  let data = ID3TagWriter.build(tag: tag)
  #expect(data.isEmpty)

  // EncoderSession with no tag also returns empty
  let options = MP3EncoderOptions()
  let encoder = try MP3Encoder(options: options)
  let session = encoder.newSession()
  #expect(session.generateID3Tag().isEmpty)
}

// MARK: - Bit Reservoir Tests

@Test func bitReservoirMainDataBegin() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
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
  let encoder = try MP3Encoder(options: options)
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

// MARK: - Latency Tests

@Test func firstEncodeIsBufferedUntilEnoughInput() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  // The filterbank, MDCT overlap, and block-switching lookahead delay output by about a frame.
  let samples = [Float](repeating: 0, count: 1152 * 2)
  #expect(state.encode(samples: samples).isEmpty)

  var output = Data()
  for _ in 0..<3 {
    output.append(state.encode(samples: samples))
  }
  #expect(output.count > 0)
  #expect(output[0] == 0xFF)
  #expect(output[1] & 0xE0 == 0xE0)
}

@Test func flushEmitsBufferedFrames() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  let samples = [Float](repeating: 0, count: 1152 * 2)
  let encoded = state.encode(samples: samples)
  #expect(encoded.isEmpty)

  let flushed = state.flush()
  #expect(flushed.count > 0)
  #expect(flushed[0] == 0xFF)
  #expect(flushed[1] & 0xE0 == 0xE0)
}

@Test func doubleFlushReturnsEmpty() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  let samples = [Float](repeating: 0, count: 1152 * 2)
  _ = state.encode(samples: samples)

  let first = state.flush()
  #expect(first.count > 0)

  let second = state.flush()
  #expect(second.isEmpty, "Second flush should return empty — nothing left to emit")
}

@Test func frameCountAndByteCountAccuracy() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  let frameTarget = 10
  var totalOutput = Data()
  for _ in 0..<frameTarget {
    let samples = [Float](repeating: 0, count: 1152 * 2)
    totalOutput.append(state.encode(samples: samples))
  }
  totalOutput.append(state.flush())

  // One extra frame carries the tail delayed by the filterbank.
  #expect(state.encodedFrameCount == frameTarget + 1)
  #expect(state.encodedByteCount == totalOutput.count)

  // At 128kbps/44.1kHz, each frame is 417 or 418 bytes
  let averageFrameSize = Double(totalOutput.count) / Double(frameTarget + 1)
  #expect(averageFrameSize >= 417.0 && averageFrameSize <= 418.0,
    "Average frame size should be ~417-418 bytes, got \(averageFrameSize)")
}

@Test func mainDataBeginStaysWithinEarlierFrames() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
  var state = encoder.newSession()

  // A quiet sine followed by a loud chirp fills the reservoir, then drains it.
  var output = Data()
  for frameIdx in 0..<12 {
    var samples = [Float]()
    for i in 0..<1152 {
      let t = Float(frameIdx * 1152 + i) / 44_100.0
      let value = frameIdx < 6 ? sin(2.0 * .pi * 440.0 * t) * 0.05 : sin(2.0 * .pi * 3000.0 * t * t) * 0.8
      samples.append(value)
      samples.append(-value)
    }
    output.append(state.encode(samples: samples))
  }
  output.append(state.flush())

  let bytes = [UInt8](output)
  let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
  var offset = 0
  var available = 0
  var frames = 0
  while offset + 6 <= bytes.count {
    #expect(bytes[offset] == 0xFF && bytes[offset + 1] & 0xE0 == 0xE0)
    let bitrate = bitrateTable[Int(bytes[offset + 2]) >> 4]
    let size = 144_000 * bitrate / 44_100 + (Int(bytes[offset + 2]) >> 1) & 0x01
    let mainDataBegin = (Int(bytes[offset + 4]) << 1) | (Int(bytes[offset + 5]) >> 7)
    #expect(mainDataBegin <= available, "Frame \(frames) points \(mainDataBegin) bytes back, only \(available) available")
    available = min(available + size - 4 - 32, 511)
    offset += size
    frames += 1
  }
  #expect(offset == bytes.count)
  #expect(frames == 13)
}

@Test func monoReservoirWorks() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .mono)
  let encoder = try MP3Encoder(options: options)
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
  #expect(state.encodedFrameCount == 9)

  // Verify frames are valid
  #expect(allData[0] == 0xFF)
  #expect(allData[1] & 0xE0 == 0xE0)

  // Check the channel mode bits indicate mono (bits 6-7 of byte 3 = 0b11)
  let modeBits = (Int(allData[3]) >> 6) & 0x03
  #expect(modeBits == 3, "Mode bits should be 0b11 for mono, got \(modeBits)")
}

@Test func multiFrameOutputContiguousFrames() throws {
  // Verify output is a sequence of contiguous valid frames with no gaps
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
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
  #expect(parsedFrames == 21, "Expected 21 frames, parsed \(parsedFrames)")
}

// MARK: - Helpers

/// Encodes a sine wave and returns complete MP3 file data with Xing header.
func makeTestMP3(
  sampleRate: Int = 44_100,
  bitrateKbps: Int = 128,
  mode: MP3EncoderOptions.Mode = .stereo,
  amplitude: Float = 0.5,
  frameCount: Int = 20
) throws -> Data {
  let options = MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: bitrateKbps, mode: mode)
  let encoder = try MP3Encoder(options: options)
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

/// Walks the frames of an MP3 stream and returns each frame's bitrate in kbps.
private func frameBitrates(_ data: Data) -> [Int] {
  let bytes = [UInt8](data)
  let bitrateTable = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
  let sampleRateTable = [44100, 48000, 32000, 0]
  var result = [Int]()
  var offset = 0
  while offset + 4 <= bytes.count, bytes[offset] == 0xFF, bytes[offset + 1] & 0xE0 == 0xE0 {
    let bitrate = bitrateTable[Int(bytes[offset + 2]) >> 4]
    let sampleRate = sampleRateTable[(Int(bytes[offset + 2]) >> 2) & 0x03]
    guard bitrate > 0, sampleRate > 0 else { break }
    result.append(bitrate)
    offset += 144_000 * bitrate / sampleRate + (Int(bytes[offset + 2]) >> 1) & 0x01
  }
  return offset == bytes.count ? result : []
}

@Test func vbrSpendsFewerBitsOnSimpleSignals() throws {
  func encode(vbr: Bool) throws -> Data {
    let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 192, vbr: vbr, mode: .jointStereo)
    var state = try MP3Encoder(options: options).newSession()
    var samples = [Float]()
    for i in 0..<(44_100 * 2) {
      let value = sin(2 * .pi * 440 * Float(i) / 44_100) * 0.5
      samples.append(value)
      samples.append(value)
    }
    var data = state.encode(samples: samples)
    data.append(state.flush())
    return data
  }
  let cbr = try encode(vbr: false)
  let vbr = try encode(vbr: true)
  let bitrates = frameBitrates(vbr)
  #expect(!bitrates.isEmpty, "VBR output should be a contiguous run of valid frames")
  #expect(Set(bitrates).count > 1 || bitrates.first.map { $0 < 192 } == true)
  #expect(vbr.count * 2 < cbr.count, "VBR \(vbr.count) bytes vs CBR \(cbr.count) bytes")
}

@Test func vbrAverageStaysAtOrBelowTarget() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, vbr: true, mode: .jointStereo)
  var state = try MP3Encoder(options: options).newSession()
  // Uncorrelated noise needs every bit it can get.
  var generator = SystemRandomNumberGenerator()
  var data = Data()
  for _ in 0..<100 {
    let samples = (0..<(1152 * 2)).map { _ in Float.random(in: -0.5...0.5, using: &generator) }
    data.append(state.encode(samples: samples))
  }
  data.append(state.flush())
  let bitrates = frameBitrates(data)
  #expect(bitrates.count == 101)
  let average = Double(bitrates.reduce(0, +)) / Double(max(bitrates.count, 1))
  #expect(average <= 128 * 1.1, "Average bitrate \(average) kbps")
}

// MARK: - Determinism and Invariant Tests

@Test func encodingIsDeterministic() throws {
  func encodeOnce() throws -> Data {
    let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
    let encoder = try MP3Encoder(options: options)
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

  let first = try encodeOnce()
  let second = try encodeOnce()
  #expect(first == second, "Same input should produce identical output (\(first.count) vs \(second.count) bytes)")
}

@Test func paddingRatioMatchesTheory() throws {
  // At 128kbps/44.1kHz: remainder = (144 * 128 * 1000) % 44100 = 42300
  // Expected padded fraction = 42300 / 44100 ≈ 0.9592
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = try MP3Encoder(options: options)
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
