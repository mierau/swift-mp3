// SwiftMP3
// AudioFileReader.swift
//
// Reads an audio file in chunks for MP3Encoder.encode(contentsOf:to:progress:). Apple
// platforms read any format Core Audio does; elsewhere, WAV files are read directly.

import Foundation
#if canImport(AVFAudio)
import AVFAudio
#endif

/// An audio file opened for reading, one chunk of planar float samples at a time.
protocol AudioFileReader {
  var sampleRate: Int { get }
  var channelCount: Int { get }
  /// Sample frames in the file, or `nil` if the file doesn't say.
  var frameCount: Int? { get }
  /// Reads up to `frames` sample frames, one array per channel; empty arrays at the end.
  mutating func read(frames: Int) throws -> [[Float]]
}

/// Opens `url` with the platform's reader.
func openAudioFile(_ url: URL) throws -> any AudioFileReader {
  #if canImport(AVFAudio)
  return try AVAudioFileReader(url: url)
  #else
  return try WAVReader(url: url)
  #endif
}

#if canImport(AVFAudio)
/// Reads any file `AVAudioFile` does, as 32-bit float.
struct AVAudioFileReader: AudioFileReader {
  private let file: AVAudioFile
  let sampleRate: Int
  let channelCount: Int
  let frameCount: Int?

  init(url: URL) throws {
    file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    sampleRate = Int(file.processingFormat.sampleRate)
    channelCount = Int(file.processingFormat.channelCount)
    frameCount = Int(file.length)
  }

  mutating func read(frames: Int) throws -> [[Float]] {
    guard file.framePosition < file.length else { return Array(repeating: [], count: channelCount) }
    guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frames)) else {
      throw MP3EncoderError.unsupportedAudioFile("can't allocate a buffer for \(file.processingFormat)")
    }
    try file.read(into: buffer, frameCount: AVAudioFrameCount(frames))
    let count = Int(buffer.frameLength)
    return (0..<channelCount).map { Array(UnsafeBufferPointer(start: buffer.floatChannelData![$0], count: count)) }
  }
}
#endif

/// Reads uncompressed WAV files: 8, 16, 24, and 32-bit integer and 32 and 64-bit float
/// samples, including `WAVE_FORMAT_EXTENSIBLE`.
struct WAVReader: AudioFileReader {
  private enum Encoding {
    case unsigned8, int16, int24, int32, float32, float64
  }

  private let handle: FileHandle
  private let encoding: Encoding
  private let blockAlign: Int
  let sampleRate: Int
  let channelCount: Int
  let frameCount: Int?
  /// Bytes of sample data not yet read, or `nil` to read to the end of the file.
  private var remaining: Int?

  init(url: URL) throws {
    handle = try FileHandle(forReadingFrom: url)
    var format: (tag: Int, channels: Int, rate: Int, blockAlign: Int, bits: Int)?
    do {
      let riff = try Self.read(handle, 12)
      guard riff.count == 12, riff[0..<4] == Array("RIFF".utf8)[...], riff[8..<12] == Array("WAVE".utf8)[...] else {
        throw MP3EncoderError.unsupportedAudioFile("not a WAV file")
      }
      while true {
        let header = try Self.read(handle, 8)
        guard header.count == 8 else { throw MP3EncoderError.unsupportedAudioFile("no data chunk") }
        let id = String(decoding: header[0..<4], as: UTF8.self)
        let size = Self.uint32(header, 4)
        if id == "data" {
          guard let format else { throw MP3EncoderError.unsupportedAudioFile("data chunk before fmt chunk") }
          // A streaming writer may leave the size at 0 or 0xFFFFFFFF.
          remaining = size == 0 || size == 0xFFFF_FFFF ? nil : size
          sampleRate = format.rate
          channelCount = format.channels
          blockAlign = format.blockAlign
          frameCount = remaining.map { $0 / format.blockAlign }
          break
        }
        let body = try Self.read(handle, size + size % 2)
        if id == "fmt " {
          guard body.count >= 16 else { throw MP3EncoderError.unsupportedAudioFile("short fmt chunk") }
          var tag = Self.uint16(body, 0)
          // WAVE_FORMAT_EXTENSIBLE keeps the real format in the first two bytes of its GUID.
          if tag == 0xFFFE, body.count >= 26 { tag = Self.uint16(body, 24) }
          format = (tag, Self.uint16(body, 2), Self.uint32(body, 4), Self.uint16(body, 12), Self.uint16(body, 14))
        }
      }
      guard let format, format.channels > 0, format.rate > 0 else {
        throw MP3EncoderError.unsupportedAudioFile("no channels or sample rate")
      }
      switch (format.tag, format.bits) {
      case (1, 8): encoding = .unsigned8
      case (1, 16): encoding = .int16
      case (1, 24): encoding = .int24
      case (1, 32): encoding = .int32
      case (3, 32): encoding = .float32
      case (3, 64): encoding = .float64
      default:
        throw MP3EncoderError.unsupportedAudioFile("WAV format \(format.tag) with \(format.bits)-bit samples")
      }
      guard format.blockAlign == format.channels * format.bits / 8 else {
        throw MP3EncoderError.unsupportedAudioFile("block size \(format.blockAlign) doesn't fit \(format.channels) channels")
      }
    } catch {
      try? handle.close()
      throw error
    }
  }

  mutating func read(frames: Int) throws -> [[Float]] {
    var bytes = frames * blockAlign
    if let remaining { bytes = min(bytes, remaining) }
    var data = try Self.read(handle, bytes)
    data.removeLast(data.count % blockAlign)
    remaining = remaining.map { $0 - data.count }
    let count = data.count / blockAlign
    let channels = channelCount
    var output = [[Float]](repeating: [Float](repeating: 0, count: count), count: channels)
    data.withUnsafeBytes { raw in
      for ch in 0..<channels {
        output[ch].withUnsafeMutableBufferPointer { out in
          for i in 0..<count {
            let offset = i * blockAlign
            switch encoding {
            case .unsigned8:
              out[i] = (Float(raw[offset + ch]) - 128) * (1 / 128)
            case .int16:
              out[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: offset + ch * 2, as: Int16.self))) * (1 / 32_768)
            case .int24:
              let p = offset + ch * 3
              let value = Int32(raw[p]) << 8 | Int32(raw[p + 1]) << 16 | Int32(raw[p + 2]) << 24
              out[i] = Float(value >> 8) * (1 / 8_388_608)
            case .int32:
              out[i] = Float(Int32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + ch * 4, as: Int32.self))) * (1 / 2_147_483_648)
            case .float32:
              let bits = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + ch * 4, as: UInt32.self))
              out[i] = Float(bitPattern: bits)
            case .float64:
              let bits = UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset + ch * 8, as: UInt64.self))
              out[i] = Float(Double(bitPattern: bits))
            }
          }
        }
      }
    }
    if count == 0 { try? handle.close() }
    return output
  }

  private static func read(_ handle: FileHandle, _ count: Int) throws -> [UInt8] {
    guard count > 0 else { return [] }
    return [UInt8](try handle.read(upToCount: count) ?? Data())
  }

  private static func uint16(_ bytes: [UInt8], _ i: Int) -> Int {
    Int(bytes[i]) | Int(bytes[i + 1]) << 8
  }

  private static func uint32(_ bytes: [UInt8], _ i: Int) -> Int {
    Int(bytes[i]) | Int(bytes[i + 1]) << 8 | Int(bytes[i + 2]) << 16 | Int(bytes[i + 3]) << 24
  }
}

extension MP3Encoder {
  /// File-to-file encode: reads the audio file at `source` and writes an MP3 to `url`, with
  /// a Xing header, as ``encode(_:to:)`` does. If encoding fails or is cancelled, the
  /// partly written file is removed.
  ///
  /// The file's sample rate replaces ``MP3EncoderOptions/sampleRate``, and a mono file
  /// makes a mono MP3. Apple platforms read any format `AVAudioFile` does (WAV, AIFF, CAF,
  /// FLAC, M4A, ...); other platforms read uncompressed WAV.
  ///
  /// - Parameters:
  ///   - source: The audio file to encode.
  ///   - url: The file URL to write the MP3 output to.
  ///   - progress: Called after each chunk with the fraction of the file encoded so far,
  ///     from 0 to 1. Not called if the file doesn't record its length.
  /// - Throws: ``MP3EncoderError/unsupportedAudioFile(_:)`` if the file can't be read or
  ///   has more than two channels for a stereo MP3, or ``MP3EncoderError`` if the
  ///   options don't suit the file's sample rate.
  public func encode(contentsOf source: URL, to url: URL, progress: @Sendable (Double) -> Void = { _ in }) async throws {
    let reader = try openAudioFile(source)
    var options = self.options
    options.sampleRate = reader.sampleRate
    if reader.channelCount == 1 {
      options.mode = .mono
    } else if options.mode != .mono && reader.channelCount > 2 {
      throw MP3EncoderError.unsupportedAudioFile("\(reader.channelCount) channels; a stereo MP3 takes 1 or 2")
    }
    let encoder = try MP3Encoder(options: options)

    let total = reader.frameCount
    var done = 0
    try await encoder.write(AudioFileChunks(reader: reader), to: url) { session, chunk in
      done += chunk[0].count
      let data = session.encode(planar: chunk)
      if let total, total > 0 { progress(min(1, Double(done) / Double(total))) }
      return data
    }
  }
}

/// The chunks of an audio file, as an `AsyncSequence` for ``MP3Encoder/write(_:to:feed:)``.
private struct AudioFileChunks: AsyncSequence, AsyncIteratorProtocol {
  typealias Element = [[Float]]
  var reader: any AudioFileReader

  mutating func next() async throws -> [[Float]]? {
    let chunk = try reader.read(frames: 16_384)
    return chunk[0].isEmpty ? nil : chunk
  }

  func makeAsyncIterator() -> AudioFileChunks { self }
}
