// SwiftMP3
// AVFoundationSupport.swift

#if canImport(AVFAudio)
import AVFAudio
import Foundation

extension EncoderSession {
  /// Appends the samples in an audio buffer and encodes any complete frames.
  ///
  /// The buffer must match the encoder's input sample rate (``sampleRate``) and hold 32-bit
  /// float or 16-bit integer samples, interleaved or not. A mono encoder averages a buffer
  /// with any number of channels; a stereo encoder takes one or two channels and plays a
  /// single channel in both. Use `AVAudioConverter` to convert other formats first.
  ///
  /// - Throws: ``MP3EncoderError/unsupportedBufferFormat(_:)`` if the buffer's format doesn't fit.
  public mutating func encode(buffer: AVAudioPCMBuffer) throws -> Data {
    let format = buffer.format
    guard format.sampleRate == Double(sampleRate) else {
      throw MP3EncoderError.unsupportedBufferFormat("sample rate \(format.sampleRate) Hz, expected \(sampleRate) Hz")
    }
    let bufferChannels = Int(format.channelCount)
    guard bufferChannels > 0, channelCount == 1 || bufferChannels <= 2 else {
      throw MP3EncoderError.unsupportedBufferFormat("\(format.channelCount) channels, expected 1 or 2")
    }
    let frames = Int(buffer.frameLength)
    let matches = bufferChannels == channelCount
    switch format.commonFormat {
    case .pcmFormatFloat32:
      guard let data = buffer.floatChannelData else { break }
      if matches && (format.isInterleaved || bufferChannels == 1) {
        return encode(samples: UnsafeBufferPointer(start: data[0], count: frames * bufferChannels))
      }
      return encode(planar: Self.planar(data, channels: bufferChannels, frames: frames, interleaved: format.isInterleaved) { $0 })
    case .pcmFormatInt16:
      guard let data = buffer.int16ChannelData else { break }
      if matches && (format.isInterleaved || bufferChannels == 1) {
        return encode(samples: UnsafeBufferPointer(start: data[0], count: frames * bufferChannels))
      }
      return encode(planar: Self.planar(data, channels: bufferChannels, frames: frames, interleaved: format.isInterleaved) {
        Float($0) * (1 / 32_768)
      })
    default:
      break
    }
    throw MP3EncoderError.unsupportedBufferFormat("\(format.commonFormat), expected 32-bit float or 16-bit integer samples")
  }

  /// Copies a buffer's samples into one float array per channel.
  private static func planar<T>(
    _ data: UnsafePointer<UnsafeMutablePointer<T>>,
    channels: Int,
    frames: Int,
    interleaved: Bool,
    _ convert: (T) -> Float
  ) -> [[Float]] {
    (0..<channels).map { ch in
      if interleaved {
        return (0..<frames).map { convert(data[0][$0 * channels + ch]) }
      }
      return UnsafeBufferPointer(start: data[ch], count: frames).map(convert)
    }
  }
}

extension MP3Encoder {
  /// Encodes an audio buffer in memory and returns a complete MP3 file, like
  /// the float-sample `encode(samples:)`.
  ///
  /// - Throws: ``MP3EncoderError/unsupportedBufferFormat(_:)`` if the buffer doesn't suit
  ///   ``EncoderSession/encode(buffer:)``.
  public func encode(buffer: AVAudioPCMBuffer) throws -> Data {
    try encodeFile { try $0.encode(buffer: buffer) }
  }

  /// File encode of audio buffers: writes MP3 frames to disk with a Xing header, as the
  /// float-sample `encode(_:to:)` does. If encoding fails or is cancelled, the partly
  /// written file is removed.
  ///
  /// Each buffer must suit ``EncoderSession/encode(buffer:)``; encoding stops with
  /// ``MP3EncoderError/unsupportedBufferFormat(_:)`` at the first one that doesn't.
  public func encode<S: AsyncSequence>(
    _ input: S, to url: URL
  ) async throws where S.Element == AVAudioPCMBuffer {
    try await write(input, to: url) { session, buffer in try session.encode(buffer: buffer) }
  }
}
#endif
