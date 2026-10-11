// SwiftMP3
// MP3Encoder.swift

import Foundation

/// Configuration options for the MP3 encoder.
public struct MP3EncoderOptions: Sendable, Equatable {
  /// The channel mode for the encoded audio.
  public enum Mode: String, Sendable, Equatable {
    /// A single audio channel.
    case mono
    /// Two independently encoded channels.
    case stereo
    /// Two channels encoded as mid/side on frames where that takes fewer bits.
    case jointStereo
  }

  /// Sample rate in Hz. 32000, 44100, and 48000 encode MPEG-1; 16000, 22050, and 24000
  /// encode MPEG-2; 8000, 11025, and 12000 encode MPEG-2.5.
  public var sampleRate: Int
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
  ///   - sampleRate: Sample rate in Hz. Defaults to 44100.
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
    self.bitrateKbps = bitrateKbps
    self.vbr = vbr
    self.mode = mode
    self.quality = quality
    self.crcProtected = crcProtected
    self.original = original
    self.copyright = copyright
    self.id3Tag = id3Tag
  }

  /// Number of interleaved channels expected in the input.
  var channelCount: Int { mode == .mono ? 1 : 2 }
}

/// An error from configuring the encoder or passing it input.
public enum MP3EncoderError: Error, Equatable, Sendable, CustomStringConvertible {
  /// The sample rate isn't one MP3 supports.
  case unsupportedSampleRate(Int)
  /// The bitrate isn't available at the sample rate's MPEG version.
  case unsupportedBitrate(Int, sampleRate: Int)
  /// The quality isn't in 0...9.
  case invalidQuality(Int)
  /// An audio buffer's sample rate, channel count, or sample format doesn't match the encoder.
  case unsupportedBufferFormat(String)

  public var description: String {
    switch self {
    case .unsupportedSampleRate(let rate):
      return "Unsupported sample rate \(rate) Hz; expected one of \(MPEGVersion.allSampleRates.sorted())"
    case .unsupportedBitrate(let kbps, let rate):
      let allowed = MPEGVersion(sampleRate: rate).map { Array($0.bitrates.dropFirst()) } ?? []
      return "Unsupported bitrate \(kbps) kbps at \(rate) Hz; expected one of \(allowed)"
    case .invalidQuality(let quality):
      return "Invalid quality \(quality); expected 0 through 9"
    case .unsupportedBufferFormat(let reason):
      return "Unsupported buffer format: \(reason)"
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
  public init(options: MP3EncoderOptions) throws {
    guard let version = MPEGVersion(sampleRate: options.sampleRate) else {
      throw MP3EncoderError.unsupportedSampleRate(options.sampleRate)
    }
    guard options.bitrateKbps > 0, version.bitrates.contains(options.bitrateKbps) else {
      throw MP3EncoderError.unsupportedBitrate(options.bitrateKbps, sampleRate: options.sampleRate)
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

  /// Streaming encode: yields MP3 frames as `Data` chunks. No Xing header is included.
  ///
  /// - Parameter input: An async sequence of interleaved PCM float sample buffers.
  /// - Returns: An `AsyncThrowingStream` that yields encoded MP3 frame data.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S
  ) -> AsyncThrowingStream<Data, Error> where S.Element == [Float] {
    let options = self.options
    return AsyncThrowingStream { continuation in
      let task = Task {
        var state = EncoderSession(options: options)
        do {
          for try await samples in input {
            try Task.checkCancellation()
            let data = state.encode(samples: samples)
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

  /// File encode: writes MP3 frames incrementally to disk with a Xing header.
  ///
  /// Creates the file at `url`, writes encoded frames as they are produced, then
  /// seeks back to the beginning to write the Xing header for accurate seeking.
  ///
  /// - Parameters:
  ///   - input: An async sequence of interleaved PCM float sample buffers.
  ///   - url: The file URL to write the MP3 output to.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S, to url: URL
  ) async throws where S.Element == [Float] {
    var state = EncoderSession(options: self.options)

    let id3Data = state.generateID3Tag()
    let xingFrameSize = state.xingFrameSize

    // Create the file: [ID3 tag] + [Xing placeholder]
    var fileHeader = id3Data
    fileHeader.append(Data(count: xingFrameSize))
    try fileHeader.write(to: url)

    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(id3Data.count + xingFrameSize))

    for try await samples in input {
      try Task.checkCancellation()
      let data = state.encode(samples: samples)
      if !data.isEmpty {
        try handle.write(contentsOf: data)
      }
    }

    let finalData = state.flush()
    if !finalData.isEmpty {
      try handle.write(contentsOf: finalData)
    }

    // Write the real Xing header after the ID3 tag
    let xingHeader = state.generateXingHeader()
    try handle.seek(toOffset: UInt64(id3Data.count))
    try handle.write(contentsOf: xingHeader)
  }
}
