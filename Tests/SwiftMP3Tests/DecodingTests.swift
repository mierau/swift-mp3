import Foundation
import Testing
@testable import SwiftMP3
#if canImport(AVFoundation)
import AVFoundation
#endif

// Round-trip tests decode the encoder's output with AVFoundation where it exists, and
// with ffmpeg elsewhere. Gapless tests need ffmpeg, which honors the LAME tag.

/// Decoded audio, one array per channel.
struct DecodedAudio {
  var sampleRate: Int
  var channels: [[Float]]
}

enum TestDecoder {
  static let ffmpeg: String? = {
    #if os(macOS) || os(Linux)
    let candidates = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]
    return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    #else
    return nil
    #endif
  }()

  static var isAvailable: Bool {
    #if canImport(AVFoundation)
    return true
    #else
    return ffmpeg != nil
    #endif
  }

  /// Decodes with the platform decoder.
  static func decode(_ data: Data) throws -> DecodedAudio {
    #if canImport(AVFoundation)
    return try decodeWithAVFoundation(data)
    #else
    return try decodeWithFFmpeg(data)
    #endif
  }

  #if canImport(AVFoundation)
  static func decodeWithAVFoundation(_ data: Data) throws -> DecodedAudio {
    let url = try writeTempMP3(data)
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    let channels = (0..<Int(format.channelCount)).map {
      Array(UnsafeBufferPointer(start: buffer.floatChannelData![$0], count: Int(buffer.frameLength)))
    }
    return DecodedAudio(sampleRate: Int(format.sampleRate), channels: channels)
  }
  #endif

  /// Decodes with ffmpeg to 32-bit float WAV and parses the result.
  static func decodeWithFFmpeg(_ data: Data) throws -> DecodedAudio {
    #if os(macOS) || os(Linux)
    let url = try writeTempMP3(data)
    defer { try? FileManager.default.removeItem(at: url) }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: ffmpeg!)
    process.arguments = ["-v", "error", "-i", url.path, "-c:a", "pcm_f32le", "-f", "wav", "-"]
    let pipe = Pipe()
    process.standardOutput = pipe
    try process.run()
    let wav = [UInt8](pipe.fileHandleForReading.readDataToEndOfFile())
    process.waitUntilExit()
    return try parseWAV(wav)
    #else
    throw DecodeError.unavailable
    #endif
  }

  enum DecodeError: Error {
    case unavailable
    case badWAV
  }

  private static func parseWAV(_ bytes: [UInt8]) throws -> DecodedAudio {
    func u32(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 | Int(bytes[i + 2]) << 16 | Int(bytes[i + 3]) << 24 }
    func u16(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 }
    guard bytes.count >= 12, Array(bytes[0..<4]) == Array("RIFF".utf8) else { throw DecodeError.badWAV }
    var offset = 12
    var channelCount = 0
    var sampleRate = 0
    while offset + 8 <= bytes.count {
      let id = String(decoding: bytes[offset..<(offset + 4)], as: UTF8.self)
      var size = u32(offset + 4)
      let body = offset + 8
      if id == "fmt " {
        channelCount = u16(body + 2)
        sampleRate = u32(body + 4)
      } else if id == "data" {
        // Streamed output has no length; take the rest.
        if size == 0 || size == 0xFFFF_FFFF || body + size > bytes.count { size = bytes.count - body }
        guard channelCount > 0 else { throw DecodeError.badWAV }
        let count = size / 4 / channelCount
        var channels = [[Float]](repeating: [Float](repeating: 0, count: count), count: channelCount)
        bytes.withUnsafeBytes { raw in
          for i in 0..<count {
            for ch in 0..<channelCount {
              channels[ch][i] = raw.loadUnaligned(fromByteOffset: body + (i * channelCount + ch) * 4, as: Float.self)
            }
          }
        }
        return DecodedAudio(sampleRate: sampleRate, channels: channels)
      }
      offset = body + size + (size & 1)
    }
    throw DecodeError.badWAV
  }
}

func writeTempMP3(_ data: Data) throws -> URL {
  let url = FileManager.default.temporaryDirectory
    .appendingPathComponent("test_\(UUID().uuidString).mp3")
  try data.write(to: url)
  return url
}

private func peak(_ samples: [Float]) -> Float {
  samples.reduce(0) { max($0, abs($1)) }
}

private func rms(_ samples: [Float]) -> Float {
  (samples.reduce(0) { $0 + $1 * $1 } / Float(max(samples.count, 1))).squareRoot()
}

/// Encodes interleaved PCM into a complete file with its Info frame.
func encodeFile(_ input: [Float], options: MP3EncoderOptions) throws -> Data {
  var session = try MP3Encoder(options: options).newSession()
  var data = session.encode(samples: input)
  data.append(session.flush())
  return session.generateXingHeader() + data
}

/// Encodes `input` (interleaved), decodes it, aligns the result, and returns the
/// signal-to-noise ratio in dB and the output/input level ratio of the first channel.
private func roundTrip(
  _ input: [Float],
  channels: Int,
  options: MP3EncoderOptions
) throws -> (snr: Double, gain: Double) {
  let decoded = try TestDecoder.decode(try encodeFile(input, options: options)).channels[0]
  let reference = stride(from: 0, to: input.count, by: channels).map { input[$0] }

  // Find any decoder delay by cross-correlating a window from the middle.
  let window = Array(reference[8192..<16384])
  var bestLag = 0
  var bestScore = -Double.infinity
  for lag in 0..<4096 where 8192 + lag + window.count <= decoded.count {
    var score = 0.0
    for i in Swift.stride(from: 0, to: window.count, by: 4) {
      score += Double(window[i]) * Double(decoded[8192 + lag + i])
    }
    if score > bestScore {
      bestScore = score
      bestLag = lag
    }
  }
  var signal = 0.0
  var noise = 0.0
  var output = 0.0
  for i in 4096..<(reference.count - 4096) where i + bestLag < decoded.count {
    let a = Double(reference[i])
    let b = Double(decoded[i + bestLag])
    signal += a * a
    noise += (a - b) * (a - b)
    output += b * b
  }
  return (10 * log10(signal / max(noise, 1e-20)), (output / max(signal, 1e-20)).squareRoot())
}

/// A few seconds of a chord with vibrato and a decaying pluck, which exercises tonal
/// masking, block switching, and the bit reservoir.
func musicLikeSignal(sampleRate: Int, channels: Int, seconds: Double = 3) -> [Float] {
  let count = Int(Double(sampleRate) * seconds)
  var output = [Float]()
  output.reserveCapacity(count * channels)
  let notes: [Float] = [220, 277.18, 329.63, 440, 659.25]
  let pluck = min(1760, Float(sampleRate) / 6)
  for i in 0..<count {
    let t = Float(i) / Float(sampleRate)
    var value: Float = 0
    for (k, f) in notes.enumerated() {
      value += sin(2 * .pi * f * t + 0.3 * sin(2 * .pi * 5 * t)) * 0.08 / Float(k + 1).squareRoot()
    }
    let pluckTime = t.truncatingRemainder(dividingBy: 0.5)
    value += sin(2 * .pi * pluck * t) * 0.2 * exp(-pluckTime * 12)
    for ch in 0..<channels {
      output.append(ch == 0 ? value : value * 0.8 + sin(2 * .pi * 330 * t) * 0.05)
    }
  }
  return output
}

// MARK: - Round trips

@Test(.enabled(if: TestDecoder.isAvailable))
func decoderReadsOutput() throws {
  let decoded = try TestDecoder.decode(try makeTestMP3())
  #expect(decoded.sampleRate == 44_100)
  #expect(decoded.channels.count == 2)
  #expect(decoded.channels[0].count > 0)
}

@Test(.enabled(if: TestDecoder.isAvailable))
func decodedSineWaveHasEnergy() throws {
  let samples = try TestDecoder.decode(try makeTestMP3(amplitude: 0.5, frameCount: 10)).channels[0]

  // A 0.5 amplitude sine should come back at about the same level.
  #expect(peak(samples) > 0.45, "Decoded sine wave peak \(peak(samples)) is too low")
  #expect(peak(samples) < 0.55, "Decoded sine wave peak \(peak(samples)) is too high")
  #expect(rms(samples) > 0.2, "Decoded sine wave RMS \(rms(samples)) is too low")
}

@Test(.enabled(if: TestDecoder.isAvailable))
func decodedSilenceIsQuiet() throws {
  let samples = try TestDecoder.decode(try makeTestMP3(amplitude: 0, frameCount: 10)).channels[0]
  #expect(peak(samples) < 0.001, "Decoded silence peak \(peak(samples)) is too high — expected near-zero")
}

@Test(.enabled(if: TestDecoder.isAvailable))
func decodedDurationCoversInput() throws {
  let frameCount = 20
  let decoded = try TestDecoder.decode(try makeTestMP3(frameCount: frameCount)).channels[0].count

  // All input, plus at most the delay and padding if the decoder doesn't trim them.
  let inputSamples = frameCount * 1152
  #expect(decoded >= inputSamples, "Decoded \(decoded) samples, input was \(inputSamples)")
  #expect(decoded <= inputSamples + 3 * 1152, "Decoded \(decoded) samples, input was \(inputSamples)")
}

@Test(.enabled(if: TestDecoder.isAvailable))
func sineRoundTripIsAccurate() throws {
  var input = [Float]()
  for i in 0..<(44_100 * 2) {
    let value = sin(2 * .pi * 1000 * Float(i) / 44_100) * 0.5
    input.append(value)
    input.append(value)
  }
  let result = try roundTrip(input, channels: 2, options: MP3EncoderOptions(bitrateKbps: 128, mode: .jointStereo))
  #expect(result.snr > 40, "SNR \(result.snr) dB")
  #expect(abs(result.gain - 1) < 0.02, "Gain \(result.gain)")
}

@Test(.enabled(if: TestDecoder.isAvailable), arguments: [
  (44_100, 128, MP3EncoderOptions.Mode.jointStereo, false),
  (44_100, 128, .stereo, false),
  (44_100, 96, .mono, false),
  (48_000, 192, .jointStereo, false),
  (32_000, 64, .jointStereo, false),
  (44_100, 128, .jointStereo, true),
  (24_000, 64, .jointStereo, false),
  (22_050, 64, .stereo, false),
  (16_000, 32, .mono, false),
  (22_050, 96, .jointStereo, true),
  (12_000, 32, .jointStereo, false),
  (11_025, 24, .mono, false),
  (8_000, 16, .mono, false),
])
func musicRoundTripQuality(sampleRate: Int, bitrate: Int, mode: MP3EncoderOptions.Mode, vbr: Bool) throws {
  let channels = mode == .mono ? 1 : 2
  let input = musicLikeSignal(sampleRate: sampleRate, channels: channels, seconds: sampleRate < 16_000 ? 4 : 3)
  let options = MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: bitrate, vbr: vbr, mode: mode)
  let result = try roundTrip(input, channels: channels, options: options)
  #expect(result.snr > 12, "SNR \(result.snr) dB")
  #expect(abs(result.gain - 1) < 0.05, "Gain \(result.gain)")
}

@Test(.enabled(if: TestDecoder.isAvailable))
func multipleConfigurationsDecodeSuccessfully() throws {
  let configs: [(sampleRate: Int, bitrate: Int, mode: MP3EncoderOptions.Mode, expectedChannels: Int)] = [
    (44_100, 128, .stereo, 2),
    (44_100, 128, .mono, 1),
    (48_000, 192, .stereo, 2),
    (32_000, 64, .stereo, 2),
    (44_100, 128, .jointStereo, 2),
    (22_050, 64, .stereo, 2),
    (16_000, 48, .mono, 1),
    (8_000, 8, .mono, 1),
  ]

  for config in configs {
    let mp3Data = try makeTestMP3(
      sampleRate: config.sampleRate,
      bitrateKbps: config.bitrate,
      mode: config.mode,
      frameCount: 5
    )
    let decoded = try TestDecoder.decode(mp3Data)
    #expect(decoded.sampleRate == config.sampleRate, "\(config): sample rate mismatch")
    #expect(decoded.channels.count == config.expectedChannels, "\(config): channel count mismatch")
    #expect(decoded.channels[0].count > 0, "\(config): no decoded samples")
  }
}

@Test(.enabled(if: TestDecoder.isAvailable))
func decodedMonoHasOneChannel() throws {
  let decoded = try TestDecoder.decode(try makeTestMP3(mode: .mono, frameCount: 5))
  #expect(decoded.channels.count == 1)
  #expect(peak(decoded.channels[0]) > 0.45, "Decoded mono sine wave should have energy")
}

// MARK: - Gapless playback

/// A small seeded generator for repeatable noise.
struct SplitMix64: RandomNumberGenerator {
  var state: UInt64
  init(seed: UInt64) { state = seed }
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}

@Test(.enabled(if: TestDecoder.ffmpeg != nil), arguments: [
  (44_100, 128, MP3EncoderOptions.Mode.jointStereo, false),
  (48_000, 160, .mono, true),
  (22_050, 64, .stereo, false),
  (8_000, 16, .mono, false),
])
func gaplessDecodeMatchesInput(sampleRate: Int, bitrate: Int, mode: MP3EncoderOptions.Mode, vbr: Bool) throws {
  let channels = mode == .mono ? 1 : 2
  // Music plus noise, so only the true alignment correlates well, at an odd length that
  // doesn't fill the last frame.
  var generator = SplitMix64(seed: 1)
  let input = musicLikeSignal(sampleRate: sampleRate, channels: channels, seconds: 2)
    .dropLast(777 * channels)
    .map { $0 + Float.random(in: -0.05...0.05, using: &generator) }
  let options = MP3EncoderOptions(sampleRate: sampleRate, bitrateKbps: bitrate, vbr: vbr, mode: mode)
  let decoded = try TestDecoder.decodeWithFFmpeg(try encodeFile(input, options: options)).channels[0]
  let reference = stride(from: 0, to: input.count, by: channels).map { input[$0] }
  #expect(decoded.count == reference.count)

  // The trimmed output lines up with the input sample for sample.
  func correlation(_ lag: Int) -> Double {
    var sum = 0.0
    for i in stride(from: 2_000, to: reference.count - 2_000, by: 2) {
      sum += Double(reference[i]) * Double(decoded[i + lag])
    }
    return sum
  }
  let best = (-64...64).max { correlation($0) < correlation($1) }
  #expect(best == 0)
}

@Test(.enabled(if: TestDecoder.ffmpeg != nil))
func resampledEncodeIsGapless() throws {
  // 96 kHz input plays back at 48 kHz with exactly half as many samples, lined up.
  var generator = SplitMix64(seed: 2)
  let reference = musicLikeSignal(sampleRate: 48_000, channels: 1, seconds: 2)
    .map { $0 + Float.random(in: -0.02...0.02, using: &generator) }
  // Upsample by inserting the midpoint, which the encoder's resampler then removes.
  var input = [Float]()
  for i in reference.indices {
    input.append(reference[i])
    input.append(i + 1 < reference.count ? (reference[i] + reference[i + 1]) / 2 : reference[i] / 2)
  }
  let options = MP3EncoderOptions(sampleRate: 96_000, bitrateKbps: 160, mode: .mono)
  let decoded = try TestDecoder.decodeWithFFmpeg(try encodeFile(input, options: options))
  #expect(decoded.sampleRate == 48_000)
  #expect(decoded.channels[0].count == reference.count)
  func correlation(_ lag: Int) -> Double {
    var sum = 0.0
    for i in stride(from: 2_000, to: reference.count - 2_000, by: 2) {
      sum += Double(reference[i]) * Double(decoded.channels[0][i + lag])
    }
    return sum
  }
  #expect((-32...32).max { correlation($0) < correlation($1) } == 0)
}

/// Interleaved stereo with a cluster of tones in 10.5-11.5 kHz, inside the intensity
/// stereo region and below the lowpass at 64 kbps, at `left` and `right` times the same level.
private func pannedTones(sampleRate: Int, seconds: Double, left: Float, right: Float) -> [Float] {
  var random = SplitMix64(seed: 7)
  var tones: [(frequency: Float, phase: Float)] = []
  for k in 0..<24 {
    let phase = Float(random.next() % 1_000) / 1_000 * 2 * Float.pi
    tones.append((10_500 + Float(k) * 1_000 / 24, phase))
  }
  var output = [Float]()
  for i in 0..<Int(Double(sampleRate) * seconds) {
    let t = Float(i) / Float(sampleRate)
    var value: Float = 0
    for tone in tones { value += sin(2 * .pi * tone.frequency * t + tone.phase) * 0.03 }
    output.append(value * left)
    output.append(value * right)
  }
  return output
}

/// Frames whose mode extension turns on intensity stereo.
private func intensityFrames(_ data: Data) -> (intensity: Int, total: Int) {
  let bytes = [UInt8](data)
  var intensity = 0
  var total = 0
  var offset = 0
  for frame in parseFrames(data) ?? [] {
    if bytes[offset + 3] >> 6 == 0b01 && (bytes[offset + 3] >> 4) & 1 == 1 { intensity += 1 }
    total += 1
    offset += frame.length
  }
  return (intensity, total)
}

@Test(.enabled(if: TestDecoder.isAvailable), arguments: [64, 96])
func intensityStereoKeepsPanning(bitrate: Int) throws {
  let input = pannedTones(sampleRate: 44_100, seconds: 2, left: 1, right: 0.33)
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: bitrate, mode: .jointStereo)
  let data = try encodeFile(input, options: options)
  let frames = intensityFrames(data)
  #expect(frames.intensity > frames.total / 2)

  // The level difference between channels survives within the 15° steps of the
  // intensity position, and the total level within 1 dB.
  let decoded = try TestDecoder.decode(data).channels
  func energy(_ x: [Float]) -> Double {
    x[4_410..<70_560].reduce(0) { $0 + Double($1) * Double($1) }
  }
  let original = (0..<2).map { ch in energy(stride(from: ch, to: input.count, by: 2).map { input[$0] }) }
  let coded = decoded.map(energy)
  let originalDifference = 10 * log10(original[0] / original[1])
  let codedDifference = 10 * log10(coded[0] / coded[1])
  #expect(abs(codedDifference - originalDifference) < 2.5)
  #expect(abs(10 * log10((coded[0] + coded[1]) / (original[0] + original[1]))) < 1)
}

@Test(arguments: [(128, MP3EncoderOptions.Mode.jointStereo), (64, .stereo)])
func intensityStereoIsOffAboveLowBitratesAndOutsideJointStereo(bitrate: Int, mode: MP3EncoderOptions.Mode) throws {
  let input = pannedTones(sampleRate: 44_100, seconds: 1, left: 1, right: 0.33)
  let data = try encodeFile(input, options: MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: bitrate, mode: mode))
  #expect(intensityFrames(data).intensity == 0)
}
