// SwiftMP3
// SwiftMP3Benchmark/main.swift
//
// Measures encode speed on synthetic signals. Run in release mode:
//
//   swift run -c release SwiftMP3Benchmark [--seconds N] [--runs N] [--hashes]
//
// `--hashes` also prints a hash of each encoded file, to check that a change leaves the
// output byte-for-byte the same.

import Foundation
import SwiftMP3

struct Signal {
  var name: String
  var samples: [Float]
}

struct Configuration {
  var name: String
  var options: MP3EncoderOptions
}

let arguments = CommandLine.arguments
func value(of flag: String, default fallback: Int) -> Int {
  guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return fallback }
  return Int(arguments[index + 1]) ?? fallback
}
let seconds = value(of: "--seconds", default: 30)
let runs = value(of: "--runs", default: 5)
let printHashes = arguments.contains("--hashes")
let writeDirectory: String? = arguments.firstIndex(of: "--write").map { arguments[$0 + 1] }

let sampleRate = 44_100
let frames = seconds * sampleRate

/// A deterministic pseudo-random generator, so every run encodes the same input.
struct Random {
  var state: UInt64 = 0x9E37_79B9_7F4A_7C15
  mutating func next() -> Float {
    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return Float(Int64(bitPattern: state) >> 40) / Float(1 << 23)
  }
}

func makeSignals() -> [Signal] {
  var random = Random()

  // Chords of decaying harmonic notes over a little noise, with a kick every half second.
  var music = [Float](repeating: 0, count: frames * 2)
  let notes: [Double] = [220, 277.18, 329.63, 440, 554.37, 659.25]
  for i in 0..<frames {
    let t = Double(i) / Double(sampleRate)
    let beat = t.truncatingRemainder(dividingBy: 0.5)
    var left = 0.0
    var right = 0.0
    for (n, f) in notes.enumerated() {
      let envelope = exp(-3 * t.truncatingRemainder(dividingBy: 2))
      var tone = 0.0
      for h in 1...6 {
        tone += sin(2 * .pi * f * Double(h) * t) / Double(h * h)
      }
      let pan = Double(n) / Double(notes.count - 1)
      left += tone * envelope * (1 - pan) * 0.12
      right += tone * envelope * pan * 0.12
    }
    let kick = sin(2 * .pi * 60 * beat) * exp(-30 * beat) * 0.5
    let noise = Double(random.next()) * 0.02
    music[2 * i] = Float(left + kick + noise)
    music[2 * i + 1] = Float(right + kick + noise)
  }

  // A sine sweeping from 20 Hz to 20 kHz.
  var sweep = [Float](repeating: 0, count: frames * 2)
  let duration = Double(frames) / Double(sampleRate)
  let k = log(1000.0) / duration
  for i in 0..<frames {
    let t = Double(i) / Double(sampleRate)
    let v = Float(0.5 * sin(2 * .pi * 20 * (exp(k * t) - 1) / k))
    sweep[2 * i] = v
    sweep[2 * i + 1] = v
  }

  var noise = [Float](repeating: 0, count: frames * 2)
  for i in noise.indices {
    noise[i] = random.next() * 0.5
  }

  return [
    Signal(name: "Music-like", samples: music),
    Signal(name: "Sine sweep", samples: sweep),
    Signal(name: "White noise", samples: noise),
  ]
}

let configurations = [
  Configuration(name: "CBR 128 q5", options: MP3EncoderOptions(bitrateKbps: 128, mode: .jointStereo, quality: 5)),
  Configuration(name: "CBR 320 q2", options: MP3EncoderOptions(bitrateKbps: 320, mode: .jointStereo, quality: 2)),
  Configuration(name: "VBR 256 q0", options: MP3EncoderOptions(bitrateKbps: 256, vbr: true, mode: .jointStereo, quality: 0)),
]

func encode(_ samples: [Float], options: MP3EncoderOptions) throws -> Data {
  var session = try MP3Encoder(options: options).newSession()
  var data = Data()
  // Feed it in chunks, the way a caller streaming audio would.
  let chunk = 4096 * 2
  var start = 0
  while start < samples.count {
    let end = min(start + chunk, samples.count)
    data.append(samples.withUnsafeBufferPointer { session.encode(samples: UnsafeBufferPointer(rebasing: $0[start..<end])) })
    start = end
  }
  data.append(session.flush())
  return session.generateXingHeader() + data
}

/// FNV-1a, enough to tell whether two encodes are identical.
func hash(_ data: Data) -> String {
  var h: UInt64 = 0xCBF2_9CE4_8422_2325
  for byte in data {
    h = (h ^ UInt64(byte)) &* 0x0000_0100_0000_01B3
  }
  return String(h, radix: 16)
}

func now() -> Double {
  Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}

let signals = makeSignals()
if let writeDirectory {
  // Write each signal as a 16-bit WAV file, to encode with other encoders for comparison.
  for signal in signals {
    var wav = Data()
    func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { wav.append(contentsOf: $0) } }
    let bytes = signal.samples.count * 2
    wav.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + bytes))
    wav.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(1)); append(UInt16(2))
    append(UInt32(sampleRate)); append(UInt32(sampleRate * 4)); append(UInt16(4)); append(UInt16(16))
    wav.append(contentsOf: Array("data".utf8)); append(UInt32(bytes))
    for s in signal.samples { append(Int16(max(-1, min(1, s)) * 32767)) }
    let name = signal.name.lowercased().replacingOccurrences(of: " ", with: "-")
    try wav.write(to: URL(fileURLWithPath: writeDirectory).appendingPathComponent("\(name).wav"))
  }
  exit(0)
}
print("Encoding \(seconds) s of 44.1 kHz stereo, best of \(runs) runs, in x realtime")
print("")
let nameWidth = 12
let columnWidth = 12
func pad(_ s: String, _ width: Int) -> String {
  s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}
print(pad("Signal", nameWidth) + configurations.map { pad($0.name, columnWidth) }.joined())
var hashes: [String] = []
for signal in signals {
  var row = pad(signal.name, nameWidth)
  for configuration in configurations {
    var best = Double.infinity
    var output = Data()
    for _ in 0..<runs {
      let start = now()
      output = try encode(signal.samples, options: configuration.options)
      best = min(best, now() - start)
    }
    row += pad(String(format: "%.0fx", Double(seconds) / best), columnWidth)
    hashes.append("\(signal.name), \(configuration.name): \(hash(output)) (\(output.count) bytes)")
  }
  print(row)
}
if printHashes {
  print("")
  hashes.forEach { print($0) }
}
