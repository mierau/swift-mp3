// SwiftMP3
// Filterbank.swift
//
// Polyphase analysis filterbank and MDCT (ISO/IEC 11172-3, 2.4.3.2 and Annex C.1.5.3).

import Foundation

/// Polyphase analysis filterbank state for one channel.
///
/// Splits PCM into 32 subbands, producing 18 subband samples per subband for each
/// 576-sample granule.
struct PolyphaseAnalyzer {
  /// The last 480 input samples followed by the current granule (1056 samples).
  private var buffer = [Float](repeating: 0, count: 480 + MPEG.samplesPerGranule)

  /// Filters one granule of PCM.
  ///
  /// - Parameters:
  ///   - input: 576 PCM samples, read with the given stride.
  ///   - stride: Distance between consecutive samples in `input` (2 for interleaved stereo).
  ///   - scale: Gain applied to each input sample.
  ///   - output: Receives 18 × 32 subband samples, indexed `slot * 32 + subband`.
  mutating func analyze(
    _ input: UnsafePointer<Float>,
    stride: Int,
    scale: Float,
    into output: UnsafeMutablePointer<Float>
  ) {
    buffer.withUnsafeMutableBufferPointer { buf in
      let base = buf.baseAddress!
      // Shift history and append the new granule.
      base.update(from: base + MPEG.samplesPerGranule, count: 480)
      for i in 0..<MPEG.samplesPerGranule {
        base[480 + i] = input[i * stride] * scale
      }

      Tables.reversedWindow.withUnsafeBufferPointer { window in
        Tables.matrix.withUnsafeBufferPointer { matrix in
          var y = [Float](repeating: 0, count: 64)
          var a = [Float](repeating: 0, count: 32)
          y.withUnsafeMutableBufferPointer { y in
            a.withUnsafeMutableBufferPointer { a in
              for slot in 0..<18 {
                Self.filter(
                  base + slot * 32,
                  window: window.baseAddress!,
                  matrix: matrix.baseAddress!,
                  y: y.baseAddress!,
                  a: a.baseAddress!,
                  output: output + slot * 32
                )
                // Compensate for frequency inversion in odd subbands at odd time slots.
                if slot & 1 == 1 {
                  for k in Swift.stride(from: 1, to: 32, by: 2) {
                    output[slot * 32 + k] = -output[slot * 32 + k]
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /// Computes 32 subband samples from the 512 samples starting at `x` (oldest first).
  @inline(__always)
  private static func filter(
    _ x: UnsafePointer<Float>,
    window: UnsafePointer<Float>,
    matrix: UnsafePointer<Float>,
    y: UnsafeMutablePointer<Float>,
    a: UnsafeMutablePointer<Float>,
    output: UnsafeMutablePointer<Float>
  ) {
    // Window and partial sums. yr[r] = sum_k w[r + 64k] * x[r + 64k]; y[j] = yr[63 - j].
    let rawX = UnsafeRawPointer(x)
    let rawW = UnsafeRawPointer(window)
    for chunk in 0..<4 {
      var acc = SIMD16<Float>()
      for k in 0..<8 {
        let offset = (chunk * 16 + k * 64) * MemoryLayout<Float>.stride
        let xv = rawX.loadUnaligned(fromByteOffset: offset, as: SIMD16<Float>.self)
        let wv = rawW.loadUnaligned(fromByteOffset: offset, as: SIMD16<Float>.self)
        acc += xv * wv
      }
      for i in 0..<16 {
        y[63 - (chunk * 16 + i)] = acc[i]
      }
    }

    // Fold the 64-point matrixing into a 32-point cosine transform using the symmetry
    // of cos((2k+1)(j-16)pi/64).
    a[0] = y[16]
    for m in 1...16 {
      a[m] = y[16 - m] + y[16 + m]
    }
    for m in 17..<32 {
      a[m] = y[16 + m] - y[80 - m]
    }

    let rawA = UnsafeRawPointer(a)
    let a0 = rawA.loadUnaligned(as: SIMD16<Float>.self)
    let a1 = rawA.loadUnaligned(fromByteOffset: 16 * MemoryLayout<Float>.stride, as: SIMD16<Float>.self)
    let rawM = UnsafeRawPointer(matrix)
    for k in 0..<32 {
      let row = k * 32 * MemoryLayout<Float>.stride
      let m0 = rawM.loadUnaligned(fromByteOffset: row, as: SIMD16<Float>.self)
      let m1 = rawM.loadUnaligned(fromByteOffset: row + 16 * MemoryLayout<Float>.stride, as: SIMD16<Float>.self)
      output[k] = (m0 * a0 + m1 * a1).sum()
    }
  }

  private enum Tables {
    /// Analysis window C[i] (ISO/IEC 11172-3 Table C.1), stored reversed.
    static let reversedWindow: [Float] = Array(analysisWindow.reversed())

    /// cos((2k + 1) * m * pi / 64) for k, m in 0..<32.
    static let matrix: [Float] = {
      var m = [Float](repeating: 0, count: 32 * 32)
      for k in 0..<32 {
        for i in 0..<32 {
          m[k * 32 + i] = Float(cos(Double((2 * k + 1) * i) * Double.pi / 64))
        }
      }
      return m
    }()

    static let analysisWindow: [Float] = [
    0.000000000, -0.000000477, -0.000000477, -0.000000477, -0.000000477, -0.000000477, -0.000000477, -0.000000954,
    -0.000000954, -0.000000954, -0.000000954, -0.000001431, -0.000001431, -0.000001907, -0.000001907, -0.000002384,
    -0.000002384, -0.000002861, -0.000003338, -0.000003338, -0.000003815, -0.000004292, -0.000004768, -0.000005245,
    -0.000006199, -0.000006676, -0.000007629, -0.000008106, -0.000009060, -0.000010014, -0.000011444, -0.000012398,
    -0.000013828, -0.000014782, -0.000016689, -0.000018120, -0.000019550, -0.000021458, -0.000023365, -0.000025272,
    -0.000027657, -0.000030041, -0.000032425, -0.000034809, -0.000037670, -0.000040531, -0.000043392, -0.000046253,
    -0.000049591, -0.000052929, -0.000055790, -0.000059605, -0.000062943, -0.000066280, -0.000070095, -0.000073433,
    -0.000076771, -0.000080585, -0.000083923, -0.000087261, -0.000090599, -0.000093460, -0.000096321, -0.000099182,
    0.000101566, 0.000103951, 0.000105858, 0.000107288, 0.000108242, 0.000108719, 0.000108719, 0.000108242,
    0.000106812, 0.000105381, 0.000102520, 0.000099182, 0.000095367, 0.000090122, 0.000084400, 0.000077724,
    0.000069618, 0.000060558, 0.000050545, 0.000039577, 0.000027180, 0.000013828, -0.000000954, -0.000017166,
    -0.000034332, -0.000052929, -0.000072956, -0.000093937, -0.000116348, -0.000140190, -0.000165462, -0.000191212,
    -0.000218868, -0.000247478, -0.000277042, -0.000307560, -0.000339031, -0.000371456, -0.000404358, -0.000438213,
    -0.000472546, -0.000507355, -0.000542164, -0.000576973, -0.000611782, -0.000646591, -0.000680923, -0.000714302,
    -0.000747204, -0.000779152, -0.000809669, -0.000838757, -0.000866413, -0.000891685, -0.000915051, -0.000935555,
    -0.000954151, -0.000968933, -0.000980854, -0.000989437, -0.000994205, -0.000995159, -0.000991821, -0.000983715,
    0.000971317, 0.000953674, 0.000930786, 0.000902653, 0.000868797, 0.000829220, 0.000783920, 0.000731945,
    0.000674248, 0.000610352, 0.000539303, 0.000462532, 0.000378609, 0.000288486, 0.000191689, 0.000088215,
    -0.000021458, -0.000137329, -0.000259876, -0.000388145, -0.000522137, -0.000661850, -0.000806808, -0.000956535,
    -0.001111031, -0.001269817, -0.001432419, -0.001597881, -0.001766682, -0.001937389, -0.002110004, -0.002283096,
    -0.002457142, -0.002630711, -0.002803326, -0.002974033, -0.003141880, -0.003306866, -0.003467083, -0.003622532,
    -0.003771782, -0.003914356, -0.004048824, -0.004174709, -0.004290581, -0.004395962, -0.004489899, -0.004570484,
    -0.004638195, -0.004691124, -0.004728317, -0.004748821, -0.004752159, -0.004737377, -0.004703045, -0.004649162,
    -0.004573822, -0.004477024, -0.004357815, -0.004215240, -0.004049301, -0.003858566, -0.003643036, -0.003401756,
    0.003134727, 0.002841473, 0.002521515, 0.002174854, 0.001800537, 0.001399517, 0.000971317, 0.000515938,
    0.000033379, -0.000475883, -0.001011848, -0.001573563, -0.002161503, -0.002774239, -0.003411293, -0.004072189,
    -0.004756451, -0.005462170, -0.006189346, -0.006937027, -0.007703304, -0.008487225, -0.009287834, -0.010103703,
    -0.010933399, -0.011775017, -0.012627602, -0.013489246, -0.014358521, -0.015233517, -0.016112804, -0.016994476,
    -0.017876148, -0.018756866, -0.019634247, -0.020506859, -0.021372318, -0.022228718, -0.023074150, -0.023907185,
    -0.024725437, -0.025527000, -0.026310921, -0.027073860, -0.027815342, -0.028532982, -0.029224873, -0.029890060,
    -0.030526638, -0.031132698, -0.031706810, -0.032248020, -0.032754898, -0.033225536, -0.033659935, -0.034055710,
    -0.034412861, -0.034730434, -0.035007000, -0.035242081, -0.035435200, -0.035586357, -0.035694122, -0.035758972,
    0.035780907, 0.035758972, 0.035694122, 0.035586357, 0.035435200, 0.035242081, 0.035007000, 0.034730434,
    0.034412861, 0.034055710, 0.033659935, 0.033225536, 0.032754898, 0.032248020, 0.031706810, 0.031132698,
    0.030526638, 0.029890060, 0.029224873, 0.028532982, 0.027815342, 0.027073860, 0.026310921, 0.025527000,
    0.024725437, 0.023907185, 0.023074150, 0.022228718, 0.021372318, 0.020506859, 0.019634247, 0.018756866,
    0.017876148, 0.016994476, 0.016112804, 0.015233517, 0.014358521, 0.013489246, 0.012627602, 0.011775017,
    0.010933399, 0.010103703, 0.009287834, 0.008487225, 0.007703304, 0.006937027, 0.006189346, 0.005462170,
    0.004756451, 0.004072189, 0.003411293, 0.002774239, 0.002161503, 0.001573563, 0.001011848, 0.000475883,
    -0.000033379, -0.000515938, -0.000971317, -0.001399517, -0.001800537, -0.002174854, -0.002521515, -0.002841473,
    0.003134727, 0.003401756, 0.003643036, 0.003858566, 0.004049301, 0.004215240, 0.004357815, 0.004477024,
    0.004573822, 0.004649162, 0.004703045, 0.004737377, 0.004752159, 0.004748821, 0.004728317, 0.004691124,
    0.004638195, 0.004570484, 0.004489899, 0.004395962, 0.004290581, 0.004174709, 0.004048824, 0.003914356,
    0.003771782, 0.003622532, 0.003467083, 0.003306866, 0.003141880, 0.002974033, 0.002803326, 0.002630711,
    0.002457142, 0.002283096, 0.002110004, 0.001937389, 0.001766682, 0.001597881, 0.001432419, 0.001269817,
    0.001111031, 0.000956535, 0.000806808, 0.000661850, 0.000522137, 0.000388145, 0.000259876, 0.000137329,
    0.000021458, -0.000088215, -0.000191689, -0.000288486, -0.000378609, -0.000462532, -0.000539303, -0.000610352,
    -0.000674248, -0.000731945, -0.000783920, -0.000829220, -0.000868797, -0.000902653, -0.000930786, -0.000953674,
    0.000971317, 0.000983715, 0.000991821, 0.000995159, 0.000994205, 0.000989437, 0.000980854, 0.000968933,
    0.000954151, 0.000935555, 0.000915051, 0.000891685, 0.000866413, 0.000838757, 0.000809669, 0.000779152,
    0.000747204, 0.000714302, 0.000680923, 0.000646591, 0.000611782, 0.000576973, 0.000542164, 0.000507355,
    0.000472546, 0.000438213, 0.000404358, 0.000371456, 0.000339031, 0.000307560, 0.000277042, 0.000247478,
    0.000218868, 0.000191212, 0.000165462, 0.000140190, 0.000116348, 0.000093937, 0.000072956, 0.000052929,
    0.000034332, 0.000017166, 0.000000954, -0.000013828, -0.000027180, -0.000039577, -0.000050545, -0.000060558,
    -0.000069618, -0.000077724, -0.000084400, -0.000090122, -0.000095367, -0.000099182, -0.000102520, -0.000105381,
    -0.000106812, -0.000108242, -0.000108719, -0.000108719, -0.000108242, -0.000107288, -0.000105858, -0.000103951,
    0.000101566, 0.000099182, 0.000096321, 0.000093460, 0.000090599, 0.000087261, 0.000083923, 0.000080585,
    0.000076771, 0.000073433, 0.000070095, 0.000066280, 0.000062943, 0.000059605, 0.000055790, 0.000052929,
    0.000049591, 0.000046253, 0.000043392, 0.000040531, 0.000037670, 0.000034809, 0.000032425, 0.000030041,
    0.000027657, 0.000025272, 0.000023365, 0.000021458, 0.000019550, 0.000018120, 0.000016689, 0.000014782,
    0.000013828, 0.000012398, 0.000011444, 0.000010014, 0.000009060, 0.000008106, 0.000007629, 0.000006676,
    0.000006199, 0.000005245, 0.000004768, 0.000004292, 0.000003815, 0.000003338, 0.000003338, 0.000002861,
    0.000002384, 0.000002384, 0.000001907, 0.000001907, 0.000001431, 0.000001431, 0.000000954, 0.000000954,
    0.000000954, 0.000000954, 0.000000477, 0.000000477, 0.000000477, 0.000000477, 0.000000477, 0.000000477,
    ]
  }
}

/// MDCT for one granule of one channel.
enum MDCT {
  /// Transforms subband samples into 576 spectral lines.
  ///
  /// - Parameters:
  ///   - previous: The previous granule's subband samples (`slot * 32 + subband`).
  ///   - current: This granule's subband samples.
  ///   - blockType: Window shape.
  ///   - shortOrder: For short blocks, maps `window * 192 + line` to its position in
  ///     scalefactor-band order.
  ///   - output: Receives 576 coefficients. Long blocks are in frequency order;
  ///     short blocks are in scalefactor band, window, frequency order.
  static func transform(
    previous: UnsafePointer<Float>,
    current: UnsafePointer<Float>,
    blockType: BlockType,
    shortOrder: UnsafePointer<Int>,
    output: UnsafeMutablePointer<Float>
  ) {
    var input = [Float](repeating: 0, count: 36)
    input.withUnsafeMutableBufferPointer { inBuf in
      let x = inBuf.baseAddress!
      for sb in 0..<32 {
        for t in 0..<18 {
          x[t] = previous[t * 32 + sb]
          x[18 + t] = current[t * 32 + sb]
        }
        if blockType == .short {
          Tables.short.withUnsafeBufferPointer { table in
            for w in 0..<3 {
              let segment = x + 6 + 6 * w
              for m in 0..<6 {
                var sum: Float = 0
                let row = table.baseAddress! + m * 12
                for n in 0..<12 {
                  sum += row[n] * segment[n]
                }
                output[shortOrder[w * 192 + sb * 6 + m]] = sum
              }
            }
          }
        } else {
          let table: [Float]
          switch blockType {
          case .start: table = Tables.start
          case .stop: table = Tables.stop
          default: table = Tables.normal
          }
          table.withUnsafeBufferPointer { table in
            let rawX = UnsafeRawPointer(x)
            let x0 = rawX.loadUnaligned(as: SIMD16<Float>.self)
            let x1 = rawX.loadUnaligned(fromByteOffset: 64, as: SIMD16<Float>.self)
            let x2 = rawX.loadUnaligned(fromByteOffset: 128, as: SIMD4<Float>.self)
            let rawT = UnsafeRawPointer(table.baseAddress!)
            for m in 0..<18 {
              let row = m * 36 * MemoryLayout<Float>.stride
              let t0 = rawT.loadUnaligned(fromByteOffset: row, as: SIMD16<Float>.self)
              let t1 = rawT.loadUnaligned(fromByteOffset: row + 64, as: SIMD16<Float>.self)
              let t2 = rawT.loadUnaligned(fromByteOffset: row + 128, as: SIMD4<Float>.self)
              output[sb * 18 + m] = (t0 * x0 + t1 * x1).sum() + (t2 * x2).sum()
            }
          }
        }
      }
    }

    if blockType != .short {
      // Aliasing reduction butterflies between adjacent subbands (the inverse of the
      // decoder's butterfly).
      for sb in 1..<32 {
        let lower = output + sb * 18
        let upper = output + sb * 18 - 1
        for i in 0..<8 {
          let bu = upper[-i]
          let bd = lower[i]
          upper[-i] = bu * Tables.cs[i] + bd * Tables.ca[i]
          lower[i] = bd * Tables.cs[i] - bu * Tables.ca[i]
        }
      }
    }
  }

  private enum Tables {
    static let cs: [Float] = coefficients.map { Float(1 / (1 + $0 * $0).squareRoot()) }
    static let ca: [Float] = coefficients.map { Float($0 / (1 + $0 * $0).squareRoot()) }
    static let coefficients: [Double] = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037]

    static let normal = longTable { i in sin(Double.pi / 36 * (Double(i) + 0.5)) }
    static let start = longTable { i in
      switch i {
      case 0..<18: return sin(Double.pi / 36 * (Double(i) + 0.5))
      case 18..<24: return 1
      case 24..<30: return sin(Double.pi / 12 * (Double(i - 18) + 0.5))
      default: return 0
      }
    }
    static let stop = longTable { i in
      switch i {
      case 0..<6: return 0
      case 6..<12: return sin(Double.pi / 12 * (Double(i - 6) + 0.5))
      case 12..<18: return 1
      default: return sin(Double.pi / 36 * (Double(i) + 0.5))
      }
    }

    /// Windowed long MDCT basis, 18 × 36, including the 1/9 normalization.
    static func longTable(window: (Int) -> Double) -> [Float] {
      var table = [Float](repeating: 0, count: 18 * 36)
      for m in 0..<18 {
        for n in 0..<36 {
          let basis = cos(Double.pi / 72 * Double(2 * n + 19) * Double(2 * m + 1))
          table[m * 36 + n] = Float(window(n) * basis / 9)
        }
      }
      return table
    }

    /// Windowed short MDCT basis, 6 × 12, including the 1/3 normalization.
    static let short: [Float] = {
      var table = [Float](repeating: 0, count: 6 * 12)
      for m in 0..<6 {
        for n in 0..<12 {
          let window = sin(Double.pi / 12 * (Double(n) + 0.5))
          let basis = cos(Double.pi / 24 * Double(2 * n + 7) * Double(2 * m + 1))
          table[m * 12 + n] = Float(window * basis / 3)
        }
      }
      return table
    }()
  }
}
