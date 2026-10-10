// SwiftMP3
// AVFoundationSupport.swift

#if canImport(AVFAudio)
import AVFAudio
import Foundation

extension EncoderSession {
  /// Appends the samples in an audio buffer and encodes any complete frames.
  ///
  /// The buffer must match the encoder's sample rate and channel count, and hold 32-bit
  /// float or 16-bit integer samples, interleaved or not. Use `AVAudioConverter` to
  /// convert other formats first.
  ///
  /// - Throws: ``MP3EncoderError/unsupportedBufferFormat(_:)`` if the buffer's format doesn't fit.
  public mutating func encode(buffer: AVAudioPCMBuffer) throws -> Data {
    let format = buffer.format
    guard format.sampleRate == Double(sampleRate) else {
      throw MP3EncoderError.unsupportedBufferFormat("sample rate \(format.sampleRate) Hz, expected \(sampleRate) Hz")
    }
    guard Int(format.channelCount) == channelCount else {
      throw MP3EncoderError.unsupportedBufferFormat("\(format.channelCount) channels, expected \(channelCount)")
    }
    let frames = Int(buffer.frameLength)
    let channels = channelCount
    switch format.commonFormat {
    case .pcmFormatFloat32:
      guard let data = buffer.floatChannelData else { break }
      if format.isInterleaved || channels == 1 {
        return encode(samples: UnsafeBufferPointer(start: data[0], count: frames * channels))
      }
      return encode(planar: (0..<channels).map { Array(UnsafeBufferPointer(start: data[$0], count: frames)) })
    case .pcmFormatInt16:
      guard let data = buffer.int16ChannelData else { break }
      if format.isInterleaved || channels == 1 {
        return encode(samples: UnsafeBufferPointer(start: data[0], count: frames * channels))
      }
      let planar = (0..<channels).map { ch in
        UnsafeBufferPointer(start: data[ch], count: frames).map { Float($0) * (1 / 32_768) }
      }
      return encode(planar: planar)
    default:
      break
    }
    throw MP3EncoderError.unsupportedBufferFormat("\(format.commonFormat), expected 32-bit float or 16-bit integer samples")
  }
}
#endif
