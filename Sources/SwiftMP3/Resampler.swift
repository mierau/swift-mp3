// SwiftMP3
// Resampler.swift
//
// Streaming sample rate conversion for input at rates MP3 can't store, with a
// Kaiser-windowed sinc kernel.

import Foundation

/// Converts interleaved audio from one sample rate to another, a block at a time.
///
/// Output sample `j` sits at input time `j * inputRate / outputRate`, so the output has no
/// delay and ``finish(into:)`` brings the total to `ceil(inputFrames * outputRate / inputRate)`
/// frames. Gapless trimming in the encoder relies on both.
struct Resampler {
  let inputRate: Int
  let outputRate: Int
  let channels: Int

  /// Input samples per channel on each side of the kernel's center.
  private let halfWidth: Int
  /// The kernel at `table[i]` for `x = i / resolution` input samples from its center.
  private let table: [Float]
  private static let resolution = 512

  /// Input per channel, starting `halfWidth` samples of silence before the first sample.
  private var history: [[Float]]
  /// Index in `history` of input sample 0 of the next output's window position.
  private var discarded = 0
  private var inputFrames = 0
  /// Samples of an incomplete interleaved frame.
  private var partial: [Float] = []
  private var outputFrames = 0

  init(inputRate: Int, outputRate: Int, channels: Int) {
    self.inputRate = inputRate
    self.outputRate = outputRate
    self.channels = channels
    // Pass up to 95% of the lower Nyquist frequency; 16 zero crossings each side.
    let cutoff = 0.95 * min(1, Double(outputRate) / Double(inputRate))
    let crossings = 16.0
    let width = crossings / cutoff
    halfWidth = Int(width.rounded(.up))
    let beta = 8.0
    func besselI0(_ x: Double) -> Double {
      var sum = 1.0
      var term = 1.0
      for k in 1..<32 {
        term *= (x / (2 * Double(k))) * (x / (2 * Double(k)))
        sum += term
      }
      return sum
    }
    let count = halfWidth * Self.resolution + 2
    table = (0..<count).map { i in
      let x = Double(i) / Double(Self.resolution)
      guard x < width else { return 0 }
      let u = cutoff * x
      let sinc = u == 0 ? 1 : sin(.pi * u) / (.pi * u)
      let r = x / width
      return Float(cutoff * sinc * besselI0(beta * (1 - r * r).squareRoot()) / besselI0(beta))
    }
    history = Array(repeating: [Float](repeating: 0, count: halfWidth), count: channels)
  }

  /// Takes interleaved input and appends every output frame it completes to `output`.
  /// A trailing partial frame waits for the rest of its samples.
  mutating func process(_ input: UnsafeBufferPointer<Float>, into output: inout [Float]) {
    var index = 0
    while !partial.isEmpty && index < input.count {
      partial.append(input[index])
      index += 1
      if partial.count == channels {
        for ch in 0..<channels { history[ch].append(partial[ch]) }
        partial.removeAll(keepingCapacity: true)
        inputFrames += 1
      }
    }
    let frames = (input.count - index) / channels
    for ch in 0..<channels {
      history[ch].reserveCapacity(history[ch].count + frames)
      var i = index + ch
      for _ in 0..<frames {
        history[ch].append(input[i])
        i += channels
      }
    }
    partial.append(contentsOf: input[(index + frames * channels)...])
    inputFrames += frames
    produce(limit: nil, into: &output)
  }

  /// Ends the input and appends the remaining output. A trailing partial frame is
  /// completed with silence.
  mutating func finish(into output: inout [Float]) {
    if !partial.isEmpty {
      partial.append(contentsOf: repeatElement(0, count: channels - partial.count))
      for ch in 0..<channels { history[ch].append(partial[ch]) }
      partial.removeAll()
      inputFrames += 1
    }
    for ch in 0..<channels {
      history[ch].append(contentsOf: repeatElement(0, count: halfWidth + 1))
    }
    let total = (Int64(inputFrames) * Int64(outputRate) + Int64(inputRate) - 1) / Int64(inputRate)
    produce(limit: Int(total), into: &output)
  }

  private mutating func produce(limit: Int?, into output: inout [Float]) {
    let available = history[0].count
    let resolution = Float(Self.resolution)
    while limit.map({ outputFrames < $0 }) ?? true {
      // Center of the kernel, in input samples (exact, as a fraction of `outputRate`).
      let position = Int64(outputFrames) * Int64(inputRate)
      let whole = Int(position / Int64(outputRate))
      let fraction = Float(position % Int64(outputRate)) / Float(outputRate)
      // `history` index of input sample `whole`, and the window's last sample.
      let center = whole + halfWidth - discarded
      guard center + halfWidth + 1 < available else { break }
      for ch in 0..<channels {
        var sum: Float = 0
        history[ch].withUnsafeBufferPointer { x in
          table.withUnsafeBufferPointer { h in
            for k in -halfWidth...(halfWidth + 1) {
              let distance = abs(Float(k) - fraction) * resolution
              let index = Int(distance)
              guard index + 1 < h.count else { continue }
              let t = distance - Float(index)
              let weight = h[index] + (h[index + 1] - h[index]) * t
              sum += x[center + k] * weight
            }
          }
        }
        output.append(sum)
      }
      outputFrames += 1
    }
    // Drop input no later output can reach.
    let position = Int64(outputFrames) * Int64(inputRate)
    let next = Int(position / Int64(outputRate)) + halfWidth - discarded
    let drop = next - halfWidth
    if drop > 4096 {
      for ch in 0..<channels { history[ch].removeFirst(drop) }
      discarded += drop
    }
  }
}
