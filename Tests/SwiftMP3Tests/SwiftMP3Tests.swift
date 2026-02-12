import Foundation
import Testing
@testable import SwiftMP3

@Test func encodeSilence() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  // Generate 1152 stereo samples of silence (one full frame)
  let silence = [Float](repeating: 0, count: 1152 * 2)
  let data = encoder.appendSamples(silence)

  // One full frame should produce output
  #expect(data.count > 0)

  // Verify MP3 sync word (0xFFF)
  #expect(data[0] == 0xFF)
  #expect(data[1] & 0xE0 == 0xE0)
}

@Test func encodeMono() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .mono)
  let encoder = MP3Encoder(options: options)

  let silence = [Float](repeating: 0, count: 1152)
  let data = encoder.appendSamples(silence)

  #expect(data.count > 0)
  #expect(data[0] == 0xFF)
}

@Test func flushProducesFinalFrame() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  // Feed less than a full frame
  let partial = [Float](repeating: 0, count: 500)
  let appendData = encoder.appendSamples(partial)
  #expect(appendData.count == 0) // Not enough for a frame yet

  let flushData = encoder.flush()
  #expect(flushData.count > 0) // Flush should pad and produce a frame
}

@Test func xingHeader() async throws {
  let options = MP3EncoderOptions(sampleRate: 44_100, bitrateKbps: 128, mode: .stereo)
  let encoder = MP3Encoder(options: options)

  let samples = [Float](repeating: 0, count: 1152 * 2)
  _ = encoder.appendSamples(samples)

  let xing = encoder.makeXingHeader()
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

  // Generate a 440Hz sine wave, one frame of stereo samples
  var samples = [Float]()
  for i in 0..<1152 {
    let t = Float(i) / 44_100.0
    let value = sin(2.0 * .pi * 440.0 * t) * 0.5
    samples.append(value) // Left
    samples.append(value) // Right
  }

  let data = encoder.appendSamples(samples)
  #expect(data.count > 0)
}
