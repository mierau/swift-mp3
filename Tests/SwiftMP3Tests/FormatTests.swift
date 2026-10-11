import Foundation
import Testing
@testable import SwiftMP3
#if canImport(AVFAudio)
import AVFAudio
#endif

/// One parsed frame header.
struct FrameHeader {
  var version: Int
  var bitrate: Int
  var sampleRate: Int
  var padding: Bool
  var length: Int
}

/// Parses every frame header in `data`; returns `nil` if the frames don't cover it exactly.
func parseFrames(_ data: Data) -> [FrameHeader]? {
  let bytes = [UInt8](data)
  let rates = [3: [44_100, 48_000, 32_000], 2: [22_050, 24_000, 16_000], 0: [11_025, 12_000, 8_000]]
  let mpeg1Bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
  let lsfBitrates = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
  var frames = [FrameHeader]()
  var offset = 0
  while offset + 4 <= bytes.count {
    guard bytes[offset] == 0xFF, bytes[offset + 1] & 0xE0 == 0xE0 else { return nil }
    let version = Int(bytes[offset + 1] >> 3) & 3
    let bitrateIndex = Int(bytes[offset + 2] >> 4)
    let rateIndex = Int(bytes[offset + 2] >> 2) & 3
    guard let table = rates[version], rateIndex < 3, (1...14).contains(bitrateIndex) else { return nil }
    let bitrate = (version == 3 ? mpeg1Bitrates : lsfBitrates)[bitrateIndex]
    let padding = bytes[offset + 2] & 2 != 0
    let length = (version == 3 ? 144_000 : 72_000) * bitrate / table[rateIndex] + (padding ? 1 : 0)
    frames.append(FrameHeader(version: version, bitrate: bitrate, sampleRate: table[rateIndex], padding: padding, length: length))
    offset += length
  }
  return offset == bytes.count ? frames : nil
}

// MARK: - Option validation

@Test func rejectsUnsupportedSampleRate() {
  #expect(throws: MP3EncoderError.unsupportedSampleRate(0)) {
    try MP3Encoder(options: MP3EncoderOptions(sampleRate: 0))
  }
  #expect(throws: MP3EncoderError.unsupportedSampleRate(44_000)) {
    try MP3Encoder(options: MP3EncoderOptions(sampleRate: 48_000, outputSampleRate: 44_000))
  }
}

@Test(arguments: [
  (44_100, 44_100), (8_000, 8_000), (44_000, 44_100), (37_800, 44_100), (7_000, 8_000),
  (88_200, 44_100), (96_000, 48_000), (176_400, 44_100), (192_000, 48_000), (50_000, 48_000),
])
func picksOutputSampleRate(input: Int, output: Int) throws {
  let session = try MP3Encoder(options: MP3EncoderOptions(sampleRate: input, bitrateKbps: 64)).newSession()
  #expect(session.sampleRate == input)
  #expect(session.outputSampleRate == output)
}

@Test func checksBitrateAgainstOutputRate() throws {
  // 96 kHz input encodes at 48 kHz, where 320 kbps is allowed but 144 kbps isn't.
  _ = try MP3Encoder(options: MP3EncoderOptions(sampleRate: 96_000, bitrateKbps: 320))
  #expect(throws: MP3EncoderError.unsupportedBitrate(144, sampleRate: 48_000)) {
    try MP3Encoder(options: MP3EncoderOptions(sampleRate: 96_000, bitrateKbps: 144))
  }
}

@Test func rejectsBitrateOutsideVersion() {
  #expect(throws: MP3EncoderError.unsupportedBitrate(129, sampleRate: 44_100)) {
    try MP3Encoder(options: MP3EncoderOptions(bitrateKbps: 129))
  }
  // 320 kbps is MPEG-1 only; 8 kbps is MPEG-2 and 2.5 only.
  #expect(throws: MP3EncoderError.unsupportedBitrate(320, sampleRate: 22_050)) {
    try MP3Encoder(options: MP3EncoderOptions(sampleRate: 22_050, bitrateKbps: 320))
  }
  #expect(throws: MP3EncoderError.unsupportedBitrate(8, sampleRate: 48_000)) {
    try MP3Encoder(options: MP3EncoderOptions(sampleRate: 48_000, bitrateKbps: 8))
  }
}

@Test func rejectsQualityOutOfRange() {
  #expect(throws: MP3EncoderError.invalidQuality(10)) {
    try MP3Encoder(options: MP3EncoderOptions(quality: 10))
  }
  #expect(throws: MP3EncoderError.invalidQuality(-1)) {
    try MP3Encoder(options: MP3EncoderOptions(quality: -1))
  }
}

@Test func errorsDescribeTheProblem() {
  let error = MP3EncoderError.unsupportedBitrate(320, sampleRate: 22_050)
  #expect(error.description.contains("144"))
}

// MARK: - MPEG-2 and 2.5

@Test(arguments: [
  (22_050, 2, 64), (24_000, 2, 160), (16_000, 2, 8),
  (11_025, 0, 32), (12_000, 0, 64), (8_000, 0, 8),
])
func lowSampleRatesUseMPEG2Frames(sampleRate: Int, version: Int, bitrate: Int) throws {
  for mode in [MP3EncoderOptions.Mode.mono, .jointStereo] {
    let channels = mode == .mono ? 1 : 2
    let input = musicLikeSignal(sampleRate: sampleRate, channels: channels, seconds: 1)
    let file = try encodeFile(input, options: MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: bitrate, mode: mode))
    let frames = try #require(parseFrames(file))
    #expect(frames.allSatisfy { $0.version == version && $0.sampleRate == sampleRate })
    #expect(frames.dropFirst().allSatisfy { $0.bitrate == bitrate })
    // 576 samples per frame, plus the Info frame and the encoder delay.
    let samples = input.count / channels
    #expect(frames.count - 1 >= (samples + 528) / 576)
    #expect(frames.count - 1 <= (samples + 528) / 576 + 3)
  }
}

@Test func lsfVBRVariesBitrate() throws {
  let input = musicLikeSignal(sampleRate: 22_050, channels: 2, seconds: 2)
  let file = try encodeFile(input, options: MP3EncoderOptions(sampleRate: 22_050, bitrateKbps: 96, vbr: true, mode: .jointStereo))
  let frames = try #require(parseFrames(file))
  let average = Double(frames.dropFirst().map(\.bitrate).reduce(0, +)) / Double(frames.count - 1)
  #expect(Set(frames.dropFirst().map(\.bitrate)).count > 1)
  #expect(average <= 96 * 1.1)
}

@Test func lsfScalefactorCompressRoundTrips() {
  for table in 0..<3 {
    for compress in 0..<512 {
      let decoded = LSFScalefactors.decode(compress)
      guard decoded.table == table else { continue }
      #expect(LSFScalefactors.compress(table: table, slen: decoded.slen) == compress)
    }
  }
}

// MARK: - LAME tag

@Test(arguments: [44_100, 22_050])
func infoFrameRecordsDelayAndPadding(sampleRate: Int) throws {
  let samples = 10_000
  var session = try MP3Encoder(options: MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: 64, mode: .mono)).newSession()
  var audio = session.encode(samples: [Float](repeating: 0.1, count: samples))
  audio.append(session.flush())
  let info = [UInt8](session.generateXingHeader())

  let sideInfo = sampleRate == 44_100 ? 17 : 9
  let lame = 4 + sideInfo + 120
  #expect(Array(info[(4 + sideInfo)..<(8 + sideInfo)]) == Array("Info".utf8))
  #expect(Array(info[lame..<(lame + 4)]) == Array("LAME".utf8))

  let delay = Int(info[lame + 21]) << 4 | Int(info[lame + 22]) >> 4
  let padding = Int(info[lame + 22] & 0x0F) << 8 | Int(info[lame + 23])
  let samplesPerFrame = sampleRate == 44_100 ? 1152 : 576
  #expect(delay == 528)
  #expect(delay + samples + padding == session.encodedFrameCount * samplesPerFrame)

  // Music length covers the Info frame and the audio; the CRCs match LAME's.
  let musicLength = info[(lame + 28)..<(lame + 32)].reduce(0) { $0 << 8 | Int($1) }
  #expect(musicLength == info.count + audio.count)
  let musicCRC = Int(info[lame + 32]) << 8 | Int(info[lame + 33])
  #expect(musicCRC == Int(lameCRC([UInt8](audio))))
  let tagCRC = Int(info[lame + 34]) << 8 | Int(info[lame + 35])
  #expect(tagCRC == Int(lameCRC(Array(info[0..<(lame + 34)]))))
}

/// CRC-16 as LAME computes it (polynomial 0x8005, reflected, initialized to 0).
private func lameCRC(_ bytes: [UInt8]) -> UInt16 {
  var crc: UInt16 = 0
  for byte in bytes {
    crc ^= UInt16(byte)
    for _ in 0..<8 {
      crc = crc & 1 != 0 ? (crc >> 1) ^ 0xA001 : crc >> 1
    }
  }
  return crc
}

// MARK: - Input formats

private func encodeAll(_ options: MP3EncoderOptions, _ body: (inout EncoderSession) throws -> Data) throws -> Data {
  var session = try MP3Encoder(options: options).newSession()
  var data = try body(&session)
  data.append(session.flush())
  return data
}

@Test func int16InputMatchesFloatInput() throws {
  let ints = (0..<8_000).map { Int16(truncatingIfNeeded: Int(sin(Double($0) * 0.05) * 20_000)) }
  let floats = ints.map { Float($0) / 32_768 }
  let options = MP3EncoderOptions(mode: .stereo)
  let fromInts = try encodeAll(options) { $0.encode(samples: ints) }
  let fromFloats = try encodeAll(options) { $0.encode(samples: floats) }
  #expect(fromInts == fromFloats)
}

@Test func planarInputMatchesInterleavedInput() throws {
  let left = (0..<5_000).map { Float(sin(Double($0) * 0.03)) * 0.5 }
  let right = (0..<5_000).map { Float(sin(Double($0) * 0.07)) * 0.3 }
  let interleaved = zip(left, right).flatMap { [$0, $1] }
  let options = MP3EncoderOptions(mode: .jointStereo)
  let planar = try encodeAll(options) { $0.encode(planar: [left, right]) }
  let packed = try encodeAll(options) { $0.encode(samples: interleaved) }
  #expect(planar == packed)
}

@Test func oddChunksMatchWholeInput() throws {
  // Stereo input split mid-frame (between left and right) encodes the same as one call.
  let input = (0..<9_001).map { Float(sin(Double($0) * 0.02)) * 0.4 }
  let options = MP3EncoderOptions(mode: .stereo)
  let whole = try encodeAll(options) { $0.encode(samples: input) }
  let chunked = try encodeAll(options) { session in
    var data = Data()
    var start = 0
    for size in [1, 333, 1_000, 2_001] where start < input.count {
      data.append(session.encode(samples: Array(input[start..<min(start + size, input.count)])))
      start += size
    }
    data.append(session.encode(samples: Array(input[start...])))
    return data
  }
  #expect(whole == chunked)
}

@Test func monoEncoderAveragesPlanarChannels() throws {
  let left = (0..<5_000).map { Float(sin(Double($0) * 0.03)) * 0.5 }
  let right = (0..<5_000).map { Float(sin(Double($0) * 0.07)) * 0.3 }
  let options = MP3EncoderOptions(mode: .mono)
  let mixed = try encodeAll(options) { $0.encode(planar: [left, right]) }
  let average = try encodeAll(options) { $0.encode(samples: zip(left, right).map { ($0 + $1) * 0.5 }) }
  #expect(mixed == average)
}

@Test func stereoEncoderPlaysMonoInBothChannels() throws {
  let mono = (0..<5_000).map { Float(sin(Double($0) * 0.03)) * 0.5 }
  let options = MP3EncoderOptions(mode: .stereo)
  let upmixed = try encodeAll(options) { $0.encode(planar: [mono]) }
  let duplicated = try encodeAll(options) { $0.encode(samples: mono.flatMap { [$0, $0] }) }
  #expect(upmixed == duplicated)
}

// MARK: - Resampling

/// A sine of `frequency` Hz sampled at `rate`, `count` samples long, interleaved over `channels`.
private func sine(_ frequency: Double, rate: Int, count: Int, channels: Int = 1) -> [Float] {
  (0..<count).flatMap { i in
    Array(repeating: Float(0.5 * sin(2 * .pi * frequency * Double(i) / Double(rate))), count: channels)
  }
}

@Test(arguments: [(96_000, 48_000), (44_100, 48_000), (37_800, 44_100), (48_000, 22_050), (8_000, 11_025)])
func resamplerIsAccurateAndExact(input: Int, output: Int) {
  let count = input / 2 + 123
  let samples = sine(1_000, rate: input, count: count, channels: 2)
  var resampler = Resampler(inputRate: input, outputRate: output, channels: 2)
  var result = [Float]()
  samples.withUnsafeBufferPointer { resampler.process($0, into: &result) }
  resampler.finish(into: &result)

  // Exactly ceil(count * output / input) frames, with no delay.
  let expectedFrames = (count * output + input - 1) / input
  #expect(result.count == expectedFrames * 2)
  let ideal = sine(1_000, rate: output, count: expectedFrames)
  var signal = 0.0
  var noise = 0.0
  for i in 200..<(expectedFrames - 200) {
    signal += Double(ideal[i] * ideal[i])
    noise += Double((result[2 * i] - ideal[i]) * (result[2 * i] - ideal[i]))
    noise += Double((result[2 * i + 1] - ideal[i]) * (result[2 * i + 1] - ideal[i]))
  }
  let snr = 10 * log10(2 * signal / max(noise, 1e-30))
  #expect(snr > 70, "SNR \(snr) dB")
}

@Test func resamplerHandlesOddChunks() {
  let samples = sine(440, rate: 96_000, count: 20_001, channels: 2)
  func run(_ chunks: [Int]) -> [Float] {
    var resampler = Resampler(inputRate: 96_000, outputRate: 44_100, channels: 2)
    var result = [Float]()
    var start = 0
    for size in chunks + [samples.count] where start < samples.count {
      let end = min(start + size, samples.count)
      samples[start..<end].withUnsafeBufferPointer { resampler.process($0, into: &result) }
      start = end
    }
    resampler.finish(into: &result)
    return result
  }
  #expect(run([1, 3, 1_000, 7, 9_999]) == run([]))
}

@Test func resampledInputEncodesAtOutputRate() throws {
  let input = sine(1_000, rate: 96_000, count: 96_000, channels: 2)
  let file = try encodeFile(input, options: MP3EncoderOptions(sampleRate: 96_000, bitrateKbps: 192))
  let frames = try #require(parseFrames(file))
  #expect(frames.allSatisfy { $0.sampleRate == 48_000 })
}

#if canImport(AVFAudio)
@Test func audioBufferInputMatchesFloatInput() throws {
  let frames = 4_000
  let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false)!
  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
  buffer.frameLength = AVAudioFrameCount(frames)
  var interleaved = [Float]()
  for i in 0..<frames {
    let l = Float(sin(Double(i) * 0.03)) * 0.5
    let r = Float(sin(Double(i) * 0.05)) * 0.3
    buffer.floatChannelData![0][i] = l
    buffer.floatChannelData![1][i] = r
    interleaved += [l, r]
  }
  let options = MP3EncoderOptions(mode: .stereo)
  let fromBuffer = try encodeAll(options) { try $0.encode(buffer: buffer) }
  let fromArray = try encodeAll(options) { $0.encode(samples: interleaved) }
  #expect(fromBuffer == fromArray)
}

@Test func audioBufferWithWrongRateThrows() throws {
  let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2)!
  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16)!
  var session = try MP3Encoder(options: MP3EncoderOptions()).newSession()
  #expect(throws: MP3EncoderError.self) { try session.encode(buffer: buffer) }
}

@Test func monoEncoderAveragesStereoBuffer() throws {
  let frames = 4_000
  let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: true)!
  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
  buffer.frameLength = AVAudioFrameCount(frames)
  var mono = [Float]()
  for i in 0..<frames {
    let l = Float(sin(Double(i) * 0.03)) * 0.5
    let r = Float(sin(Double(i) * 0.05)) * 0.3
    buffer.floatChannelData![0][2 * i] = l
    buffer.floatChannelData![0][2 * i + 1] = r
    mono.append((l + r) * 0.5)
  }
  let options = MP3EncoderOptions(mode: .mono)
  let fromBuffer = try encodeAll(options) { try $0.encode(buffer: buffer) }
  let fromArray = try encodeAll(options) { $0.encode(samples: mono) }
  #expect(fromBuffer == fromArray)
}
#endif

// MARK: - ReplayGain

@Test(arguments: [48_000, 44_100, 22_050])
func measuresLoudnessOfReferenceTone(sampleRate: Int) throws {
  // EBU Tech 3341 case 1: a 1 kHz stereo sine at -23 dBFS measures -23 LUFS.
  let amplitude = Float(pow(10, -23.0 / 20))
  var input = [Float]()
  for i in 0..<(sampleRate * 5) {
    let value = amplitude * Float(sin(2 * Double.pi * 1_000 * Double(i) / Double(sampleRate)))
    input += [value, value]
  }
  var session = try MP3Encoder(options: MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: 64)).newSession()
  _ = session.encode(samples: input)
  _ = session.flush()
  let gain = try #require(session.replayGain)
  #expect(abs(gain - 5) < 0.1, "Gain \(gain) dB")
  #expect(abs(session.peakAmplitude - amplitude) < 1e-4)

  // The LAME tag holds the peak (1.0 = 2^23) and the radio gain in tenths of a dB.
  let header = [UInt8](session.generateXingHeader())
  let tag = try #require(header.indices.first { header[$0...].starts(with: Array("LAMESwMP3".utf8)) })
  let peak = header[(tag + 11)..<(tag + 15)].reduce(0) { $0 << 8 | Int($1) }
  #expect(abs(Double(peak) / Double(1 << 23) - Double(amplitude)) < 1e-4)
  let radio = Int(header[tag + 15]) << 8 | Int(header[tag + 16])
  #expect(radio >> 13 == 1)
  #expect(radio & 0x3FF == 50)
}

@Test func silenceHasNoReplayGain() throws {
  var session = try MP3Encoder(options: MP3EncoderOptions()).newSession()
  _ = session.encode(samples: [Float](repeating: 0, count: 88_200))
  _ = session.flush()
  #expect(session.replayGain == nil)
}
