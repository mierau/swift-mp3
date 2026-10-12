// SwiftMP3
// MP3Encoder.swift

import Foundation

/// Configuration options for the MP3 encoder.
public struct MP3EncoderOptions: Sendable, Hashable, Codable {
  /// The channel mode for the encoded audio.
  public enum Mode: String, Sendable, Hashable, Codable, CaseIterable {
    /// A single audio channel.
    case mono
    /// Two independently encoded channels.
    case stereo
    /// Two channels encoded as mid/side on frames where that takes fewer bits.
    case jointStereo
  }

  /// Sample rate of the input in Hz. Any positive rate works: one MP3 can't store is
  /// resampled to ``outputSampleRate``.
  public var sampleRate: Int
  /// Sample rate of the MP3 in Hz, or `nil` to use ``sampleRate`` when MP3 supports it and
  /// the nearest rate that keeps the input's bandwidth when it doesn't (44100 or 48000
  /// above 48 kHz). 32000, 44100, and 48000 encode MPEG-1; 16000, 22050, and 24000 encode
  /// MPEG-2; 8000, 11025, and 12000 encode MPEG-2.5.
  public var outputSampleRate: Int?
  /// Bitrate in kbps for CBR, or the maximum average bitrate for VBR. MPEG-1 allows 32,
  /// 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, and 320. MPEG-2 and 2.5 allow
  /// 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, and 160.
  public var bitrateKbps: Int
  /// Whether to vary the bitrate frame by frame. Each frame gets the bits its masking
  /// threshold calls for, and the average stays at or below ``bitrateKbps``.
  public var vbr: Bool
  /// Channel mode (mono, stereo, or joint stereo).
  public var mode: Mode
  /// Encoding quality from 0 (best, slowest) to 9 (fastest).
  public var quality: Int
  /// Whether to include CRC error protection in each frame.
  public var crcProtected: Bool
  /// Whether to set the original bit in the frame header.
  public var original: Bool
  /// Whether to set the copyright bit in the frame header.
  public var copyright: Bool
  /// ID3v2.3 metadata tag to embed at the start of the file.
  public var id3Tag: ID3Tag?

  /// Creates a new set of encoder options.
  /// - Parameters:
  ///   - sampleRate: Sample rate of the input in Hz. Defaults to 44100.
  ///   - outputSampleRate: Sample rate of the MP3 in Hz, or `nil` to pick one from
  ///     `sampleRate`. Defaults to `nil`.
  ///   - bitrateKbps: Bitrate in kbps; one of the bitrates for the sample rate's MPEG
  ///     version. Defaults to 128.
  ///   - vbr: Vary the bitrate per frame, averaging at most `bitrateKbps`. Defaults to `false`.
  ///   - mode: Channel mode. Defaults to `.jointStereo`, which switches each frame to
  ///     mid/side coding when that needs fewer bits.
  ///   - quality: Quality level from 0 (best) to 9 (fastest). Defaults to 5.
  ///   - crcProtected: Enable CRC protection. Defaults to `false`.
  ///   - original: Set the original flag. Defaults to `true`.
  ///   - copyright: Set the copyright flag. Defaults to `false`.
  ///   - id3Tag: ID3v2.3 metadata to embed. Defaults to `nil`.
  public init(
    sampleRate: Int = 44_100,
    outputSampleRate: Int? = nil,
    bitrateKbps: Int = 128,
    vbr: Bool = false,
    mode: Mode = .jointStereo,
    quality: Int = 5,
    crcProtected: Bool = false,
    original: Bool = true,
    copyright: Bool = false,
    id3Tag: ID3Tag? = nil
  ) {
    self.sampleRate = sampleRate
    self.outputSampleRate = outputSampleRate
    self.bitrateKbps = bitrateKbps
    self.vbr = vbr
    self.mode = mode
    self.quality = quality
    self.crcProtected = crcProtected
    self.original = original
    self.copyright = copyright
    self.id3Tag = id3Tag
  }

  /// The sample rates an MP3 can store, in Hz, from lowest to highest.
  public static let supportedOutputSampleRates: [Int] = MPEGVersion.allSampleRates.sorted()

  /// The bitrates allowed at an MP3 sample rate, in kbps, from lowest to highest, or an
  /// empty array if MP3 can't store `outputSampleRate`.
  public static func supportedBitrates(outputSampleRate: Int) -> [Int] {
    MPEGVersion(sampleRate: outputSampleRate).map { Array($0.bitrates.dropFirst()) } ?? []
  }

  /// Number of interleaved channels expected in the input.
  var channelCount: Int { mode == .mono ? 1 : 2 }

  /// The rate the MP3 is encoded at: ``outputSampleRate``, or the one picked for ``sampleRate``.
  var encodedSampleRate: Int {
    if let outputSampleRate { return outputSampleRate }
    let rates = MPEGVersion.allSampleRates.sorted()
    if rates.contains(sampleRate) { return sampleRate }
    if sampleRate > 48_000 { return sampleRate % 11_025 == 0 ? 44_100 : 48_000 }
    return rates.first { $0 >= sampleRate } ?? 48_000
  }
}

/// An error from configuring the encoder or passing it input.
public enum MP3EncoderError: Error, Equatable, Sendable, CustomStringConvertible {
  /// The input sample rate isn't positive, or the output sample rate isn't one MP3 supports.
  case unsupportedSampleRate(Int)
  /// The bitrate isn't available at the sample rate's MPEG version.
  case unsupportedBitrate(Int, sampleRate: Int)
  /// The quality isn't in 0...9.
  case invalidQuality(Int)
  /// An audio buffer's sample rate, channel count, or sample format doesn't match the encoder.
  case unsupportedBufferFormat(String)
  /// An audio file can't be read, or its channels don't suit the encoder.
  case unsupportedAudioFile(String)

  public var description: String {
    switch self {
    case .unsupportedSampleRate(let rate):
      return "Unsupported sample rate \(rate) Hz; expected one of \(MP3EncoderOptions.supportedOutputSampleRates)"
    case .unsupportedBitrate(let kbps, let rate):
      let allowed = MP3EncoderOptions.supportedBitrates(outputSampleRate: rate)
      return "Unsupported bitrate \(kbps) kbps at \(rate) Hz; expected one of \(allowed)"
    case .invalidQuality(let quality):
      return "Invalid quality \(quality); expected 0 through 9"
    case .unsupportedBufferFormat(let reason):
      return "Unsupported buffer format: \(reason)"
    case .unsupportedAudioFile(let reason):
      return "Unsupported audio file: \(reason)"
    }
  }
}

/// MPEG Layer III (MP3) encoder.
///
/// A stateless, `Sendable` value type that holds encoding configuration. Create an
/// ``EncoderSession`` via ``newSession()`` for synchronous encoding, or use the async
/// ``encode(_:)`` and ``encode(_:to:)`` methods for streaming and file output.
///
/// Usage:
/// ```swift
/// let encoder = try MP3Encoder(options: MP3EncoderOptions())
/// var session = encoder.newSession()
/// var mp3Data = session.encode(samples: pcmSamples)
/// mp3Data.append(session.flush())
/// let finalData = session.generateXingHeader() + mp3Data
/// ```
public struct MP3Encoder: Sendable {
  /// The encoding configuration.
  public let options: MP3EncoderOptions

  /// Creates a new MP3 encoder with the given options.
  /// - Parameter options: Configuration for sample rate, bitrate, channel mode, etc.
  /// - Throws: ``MP3EncoderError`` if the sample rate, bitrate, or quality isn't supported.
  ///   The bitrate is checked against the output sample rate.
  public init(options: MP3EncoderOptions) throws {
    guard options.sampleRate > 0 else {
      throw MP3EncoderError.unsupportedSampleRate(options.sampleRate)
    }
    let rate = options.encodedSampleRate
    guard let version = MPEGVersion(sampleRate: rate) else {
      throw MP3EncoderError.unsupportedSampleRate(rate)
    }
    guard options.bitrateKbps > 0, version.bitrates.contains(options.bitrateKbps) else {
      throw MP3EncoderError.unsupportedBitrate(options.bitrateKbps, sampleRate: rate)
    }
    guard (0...9).contains(options.quality) else {
      throw MP3EncoderError.invalidQuality(options.quality)
    }
    self.options = options
  }

  /// Creates a mutable encoding session for synchronous use.
  public func newSession() -> EncoderSession {
    EncoderSession(options: self.options)
  }

  /// Encodes a whole clip in memory and returns a complete MP3 file: the ID3 tag (if
  /// ``MP3EncoderOptions/id3Tag`` is set, with ReplayGain), the Info frame for seeking
  /// and gapless playback, and the audio.
  ///
  /// - Parameter samples: Interleaved PCM float samples (L, R, L, R, ... for stereo),
  ///   normalized to [-1.0, 1.0].
  public func encode(samples: [Float]) -> Data {
    encodeFile { $0.encode(samples: samples) }
  }

  /// Encodes interleaved 16-bit samples in memory, like the float-sample `encode(samples:)`.
  public func encode(samples: [Int16]) -> Data {
    encodeFile { $0.encode(samples: samples) }
  }

  /// Encodes one array of samples per channel in memory, like the float-sample `encode(samples:)`.
  /// See ``EncoderSession/encode(planar:)`` for how channel counts are matched.
  public func encode(planar channels: [[Float]]) -> Data {
    encodeFile { $0.encode(planar: channels) }
  }

  /// Runs a session over `feed` and assembles the file: ID3 tag, Info frame, then audio.
  func encodeFile(_ feed: (inout EncoderSession) throws -> Data) rethrows -> Data {
    var session = newSession()
    var audio = try feed(&session)
    audio.append(session.flush())
    var file = session.generateID3Tag()
    file.append(session.generateXingHeader())
    file.append(audio)
    return file
  }

  /// Streaming encode: yields MP3 frames as `Data` chunks. No Xing header is included.
  ///
  /// - Parameter input: An async sequence of interleaved PCM float sample buffers.
  /// - Returns: An `AsyncThrowingStream` that yields encoded MP3 frame data.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S
  ) -> AsyncThrowingStream<Data, Error> where S.Element == [Float] {
    stream(input) { session, samples in session.encode(samples: samples) }
  }

  /// Streaming encode of interleaved 16-bit samples, like the float-sample `encode(_:)`.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S
  ) -> AsyncThrowingStream<Data, Error> where S.Element == [Int16] {
    stream(input) { session, samples in session.encode(samples: samples) }
  }

  /// File encode: writes MP3 frames incrementally to disk with a Xing header.
  ///
  /// Creates the file at `url`, writes encoded frames as they are produced, then
  /// seeks back to the beginning to write the Xing header for accurate seeking. If
  /// encoding fails or is cancelled, the partly written file is removed.
  ///
  /// - Parameters:
  ///   - input: An async sequence of interleaved PCM float sample buffers.
  ///   - url: The file URL to write the MP3 output to.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S, to url: URL
  ) async throws where S.Element == [Float] {
    try await write(input, to: url) { session, samples in session.encode(samples: samples) }
  }

  /// File encode of interleaved 16-bit samples, like the float-sample `encode(_:to:)`.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S, to url: URL
  ) async throws where S.Element == [Int16] {
    try await write(input, to: url) { session, samples in session.encode(samples: samples) }
  }

  /// Encodes each element of `input` with `feed`, yielding the MP3 frames.
  func stream<S: AsyncSequence & Sendable>(
    _ input: S,
    feed: @escaping @Sendable (inout EncoderSession, S.Element) throws -> Data
  ) -> AsyncThrowingStream<Data, Error> {
    let options = self.options
    return AsyncThrowingStream { continuation in
      let task = Task {
        var state = EncoderSession(options: options)
        do {
          for try await element in input {
            try Task.checkCancellation()
            let data = try feed(&state, element)
            if !data.isEmpty {
              continuation.yield(data)
            }
          }
          let final = state.flush()
          if !final.isEmpty {
            continuation.yield(final)
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  /// Encodes each element of `input` with `feed` into a file at `url`, removing the file
  /// if anything throws.
  func write<S: AsyncSequence>(
    _ input: S,
    to url: URL,
    feed: (inout EncoderSession, S.Element) throws -> Data
  ) async throws {
    var state = EncoderSession(options: self.options)

    // Leave room in the ID3 tag for the ReplayGain frames, known only at the end.
    let id3Data = options.id3Tag.map {
      let size = max(ID3TagWriter.build(tag: $0).count, 10) + ID3TagWriter.replayGainFramesSize
      return ID3TagWriter.build(tag: $0, minimumSize: size)
    } ?? Data()
    let xingFrameSize = state.xingFrameSize

    // Create the file: [ID3 tag] + [Xing placeholder]
    var fileHeader = id3Data
    fileHeader.append(Data(count: xingFrameSize))
    try fileHeader.write(to: url)

    let handle: FileHandle
    do {
      handle = try FileHandle(forWritingTo: url)
    } catch {
      try? FileManager.default.removeItem(at: url)
      throw error
    }
    do {
      try handle.seek(toOffset: UInt64(id3Data.count + xingFrameSize))

      for try await element in input {
        try Task.checkCancellation()
        let data = try feed(&state, element)
        if !data.isEmpty {
          try handle.write(contentsOf: data)
        }
      }
      try Task.checkCancellation()

      let finalData = state.flush()
      if !finalData.isEmpty {
        try handle.write(contentsOf: finalData)
      }

      // Rewrite the ID3 tag with the ReplayGain, then the real Xing header after it.
      if let tag = options.id3Tag {
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: ID3TagWriter.build(tag: tag, replayGain: state.id3ReplayGain, minimumSize: id3Data.count))
      }
      let xingHeader = state.generateXingHeader()
      try handle.seek(toOffset: UInt64(id3Data.count))
      try handle.write(contentsOf: xingHeader)
      try handle.close()
    } catch {
      try? handle.close()
      try? FileManager.default.removeItem(at: url)
      throw error
    }
  }
}
