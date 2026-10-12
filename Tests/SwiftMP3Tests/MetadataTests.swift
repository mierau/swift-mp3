import Foundation
import Testing
@testable import SwiftMP3

/// The frames of an ID3v2.3 tag at the start of `data`, by ID, and the tag's total size.
private func parseID3(_ data: Data) -> (frames: [(id: String, content: [UInt8])], size: Int)? {
  let bytes = [UInt8](data)
  guard bytes.count >= 10, bytes[0..<3] == [0x49, 0x44, 0x33] else { return nil }
  let size = 10 + (Int(bytes[6]) << 21 | Int(bytes[7]) << 14 | Int(bytes[8]) << 7 | Int(bytes[9]))
  var frames: [(String, [UInt8])] = []
  var i = 10
  while i + 10 <= size, bytes[i] != 0 {
    let id = String(decoding: bytes[i..<i + 4], as: UTF8.self)
    let length = bytes[i + 4..<i + 8].reduce(0) { $0 << 8 | Int($1) }
    frames.append((id, Array(bytes[i + 10..<i + 10 + length])))
    i += 10 + length
  }
  return (frames, size)
}

/// The Latin-1 text of a text frame.
private func text(_ content: [UInt8]) -> String {
  String(decoding: content.dropFirst(), as: UTF8.self)
}

/// The value of a TXXX frame with the given description.
private func userText(_ frames: [(id: String, content: [UInt8])], _ description: String) -> String? {
  for frame in frames where frame.id == "TXXX" {
    let parts = frame.content.dropFirst().split(separator: 0, maxSplits: 1, omittingEmptySubsequences: false)
    if parts.count == 2, String(decoding: parts[0], as: UTF8.self) == description {
      return String(decoding: parts[1], as: UTF8.self)
    }
  }
  return nil
}

@Test func id3WritesExtendedFrames() throws {
  let tag = ID3Tag(
    title: "Song", artist: "Guest", album: "Album", albumArtist: "Band", composer: "Writer",
    track: 3, trackTotal: 12, disc: 2, discTotal: 2, lyrics: "La la la"
  )
  let parsed = try #require(parseID3(ID3TagWriter.build(tag: tag)))
  let frames = Dictionary(parsed.frames.map { ($0.id, $0.content) }, uniquingKeysWith: { a, _ in a })
  #expect(text(try #require(frames["TPE2"])) == "Band")
  #expect(text(try #require(frames["TCOM"])) == "Writer")
  #expect(text(try #require(frames["TPOS"])) == "2/2")
  #expect(text(try #require(frames["TRCK"])) == "3/12")
  // USLT: encoding, language, empty description, then the lyrics.
  let expected: [UInt8] = [0, 0x65, 0x6E, 0x67, 0] + Array("La la la".utf8)
  #expect(try #require(frames["USLT"]) == expected)
}

@Test func id3PaddingKeepsSizeAndRewritesInPlace() throws {
  let tag = ID3Tag(title: "Song")
  let reserved = ID3TagWriter.build(tag: tag).count + ID3TagWriter.replayGainFramesSize
  let placeholder = ID3TagWriter.build(tag: tag, minimumSize: reserved)
  #expect(placeholder.count == reserved)
  #expect(try #require(parseID3(placeholder)).frames.map(\.id) == ["TIT2"])

  for gain in [ID3TagWriter.ReplayGain(gain: -99.99, peak: 99.999999), .init(gain: 3.2, peak: 0.5)] {
    let final = ID3TagWriter.build(tag: tag, replayGain: gain, minimumSize: reserved)
    #expect(final.count == reserved)
  }
}

@Test func sessionID3TagCarriesReplayGainAfterFlush() throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, mode: .mono, id3Tag: ID3Tag(title: "Tone"))
  var session = try MP3Encoder(options: options).newSession()
  let samples = (0..<44_100).map { i -> Float in
    let phase: Double = 2 * Double.pi * 1_000 * Double(i) / 44_100
    return Float(sin(phase) * 0.5)
  }
  _ = session.encode(samples: samples)
  #expect(userText(try #require(parseID3(session.generateID3Tag())).frames, "REPLAYGAIN_TRACK_GAIN") == nil)
  _ = session.flush()

  let frames = try #require(parseID3(session.generateID3Tag())).frames
  let gain = try #require(userText(frames, "REPLAYGAIN_TRACK_GAIN"))
  let peak = try #require(userText(frames, "REPLAYGAIN_TRACK_PEAK"))
  #expect(gain == String(format: "%.2f dB", session.replayGain!))
  #expect(gain.hasSuffix(" dB"))
  #expect(abs(Double(peak)! - 0.5) < 0.001)
}

@Test func fileEncodeWritesReplayGainToID3() async throws {
  let tag = ID3Tag(title: "Tone", albumArtist: "Band")
  let options = MP3EncoderOptions(sampleRate: 44_100, mode: .mono, id3Tag: tag)
  let encoder = try MP3Encoder(options: options)
  let url = FileManager.default.temporaryDirectory.appendingPathComponent("rg_\(UUID().uuidString).mp3")
  defer { try? FileManager.default.removeItem(at: url) }

  let source = AsyncStream<[Float]> { continuation in
    for chunk in 0..<10 {
      continuation.yield((0..<4_410).map { i -> Float in
        let phase: Double = 2 * Double.pi * 440 * Double(chunk * 4_410 + i) / 44_100
        return Float(sin(phase) * 0.25)
      })
    }
    continuation.finish()
  }
  try await encoder.encode(source, to: url)

  let data = try Data(contentsOf: url)
  let parsed = try #require(parseID3(data))
  #expect(parsed.frames.map(\.id) == ["TIT2", "TPE2", "TXXX", "TXXX"])
  #expect(userText(parsed.frames, "REPLAYGAIN_TRACK_GAIN") != nil)
  #expect(abs(Double(try #require(userText(parsed.frames, "REPLAYGAIN_TRACK_PEAK")))! - 0.25) < 0.001)

  // The Info frame still follows the tag.
  let audio = [UInt8](data[parsed.size...])
  #expect(audio[0] == 0xFF)
  #expect(audio.indices.dropLast(4).contains { Array(audio[$0..<$0 + 4]) == Array("Info".utf8) })
}
