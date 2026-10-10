// SwiftMP3
// Quantizer.swift
//
// Rate/distortion loop for one granule of one channel: chooses scalefactors and a global
// gain so that quantization noise stays under the masking threshold where the bit budget
// allows (ISO/IEC 11172-3 Annex C.1.5.4), then selects Huffman tables.

import Foundation

/// Side information and quantized values for one granule of one channel.
struct GranuleInfo {
  var blockType: BlockType = .normal
  var part2Length = 0
  var part23Length = 0
  var bigValues = 0
  var count1 = 0
  var globalGain = 0
  var scalefacCompress = 0
  var tableSelect = [0, 0, 0]
  var region0Count = 0
  var region1Count = 0
  var count1TableSelect = 0
  var preflag = 0
  var scalefacScale = 0
  /// Long blocks: 22 entries (band 21 is always 0). Short blocks: 39 entries,
  /// indexed `band * 3 + window` (band 12 is always 0).
  var scalefactors: [Int] = []
  /// Quantized magnitudes, 576 entries (signs come from the spectrum).
  var quantized: [Int32] = []

  var isShort: Bool { blockType == .short }

  /// The big-values region boundaries `(region1Start, region2Start)` in lines.
  func regionBoundaries(longBands: [Int]) -> (Int, Int) {
    if blockType != .normal {
      return (36, 576)
    }
    return (longBands[region0Count + 1], longBands[region0Count + region1Count + 2])
  }
}

/// Quantization bands for one block type, in coding order.
private struct BandLayout {
  let count: Int
  let start: [Int]
  let end: [Int]
  /// Largest scalefactor each band can hold (0 for bands without a scalefactor).
  let limit: [Int]
  /// Pre-emphasis added to each band's scalefactor when `preflag` is set.
  let pretab: [Int]

  init(long bands: [Int]) {
    count = 22
    start = Array(bands[0..<22])
    end = Array(bands[1...22])
    limit = (0..<22).map { $0 < 11 ? 15 : ($0 < 21 ? 7 : 0) }
    pretab = MPEG.pretab
  }

  init(short bands: [Int]) {
    count = 39
    var start = [Int]()
    var end = [Int]()
    for band in 0..<13 {
      let width = bands[band + 1] - bands[band]
      for w in 0..<3 {
        start.append(bands[band] * 3 + w * width)
        end.append(bands[band] * 3 + (w + 1) * width)
      }
    }
    self.start = start
    self.end = end
    limit = (0..<39).map { $0 < 18 ? 15 : ($0 < 36 ? 7 : 0) }
    pretab = Array(repeating: 0, count: 39)
  }
}

/// Quantization noise relative to the allowed noise, summed over bands.
private struct Distortion {
  var overCount = 0
  var overNoise: Float = 0
  var maxNoise: Float = -.infinity

  func isBetter(than other: Distortion) -> Bool {
    if overCount == 0 && other.overCount == 0 {
      return maxNoise < other.maxNoise
    }
    if overNoise != other.overNoise {
      return overNoise < other.overNoise
    }
    return maxNoise < other.maxNoise
  }
}

/// Per-sample-rate quantizer with reusable scratch buffers.
struct Quantizer {
  static let maxQuantized: Int32 = 8206

  let longBands: [Int]
  private let longLayout: BandLayout
  private let shortLayout: BandLayout

  /// Number of outer-loop iterations (noise shaping effort).
  private let iterations: Int
  /// Whether to search for the best big-values region split.
  private let searchRegions: Bool

  /// |x|^(3/4), then |x|, for the granule being quantized.
  private var floats = [Float](repeating: 0, count: 576 * 2)
  private var ix = [Int32](repeating: 0, count: 576)

  init(sampleRateIndex: Int, quality: Int) {
    longBands = MPEG.longBands(sampleRateIndex: sampleRateIndex)
    longLayout = BandLayout(long: longBands)
    shortLayout = BandLayout(short: MPEG.shortBands(sampleRateIndex: sampleRateIndex))
    switch quality {
    case 0...2: iterations = 40
    case 3...5: iterations = 24
    case 6...7: iterations = 10
    default: iterations = 0
    }
    searchRegions = quality <= 5
  }

  /// Quantizes one granule within `maxBits` (part 2 + part 3).
  mutating func quantize(
    spectrum: UnsafePointer<Float>,
    blockType: BlockType,
    allowedNoise: [Float],
    maxBits: Int
  ) -> GranuleInfo {
    let layout = blockType == .short ? shortLayout : longLayout
    let longBands = self.longBands
    let iterations = self.iterations
    let searchRegions = self.searchRegions
    return floats.withUnsafeMutableBufferPointer { floats in
      ix.withUnsafeMutableBufferPointer { ix in
        var work = GranuleQuantization(
          xr34: floats.baseAddress!,
          absXR: floats.baseAddress! + 576,
          ix: ix.baseAddress!,
          layout: layout,
          longBands: longBands,
          allowedNoise: allowedNoise
        )
        return work.run(
          spectrum: spectrum,
          blockType: blockType,
          maxBits: maxBits,
          iterations: iterations,
          searchRegions: searchRegions
        )
      }
    }
  }
}

extension Quantizer {
  /// Quantizes one granule as coarsely as the allowed noise permits, ignoring any bit
  /// budget. Returns `nil` if the result does not fit in a granule (4095 bits).
  mutating func quantizeTransparent(
    spectrum: UnsafePointer<Float>,
    blockType: BlockType,
    allowedNoise: [Float]
  ) -> GranuleInfo? {
    let layout = blockType == .short ? shortLayout : longLayout
    let longBands = self.longBands
    let searchRegions = self.searchRegions
    return floats.withUnsafeMutableBufferPointer { floats in
      ix.withUnsafeMutableBufferPointer { ix in
        var work = GranuleQuantization(
          xr34: floats.baseAddress!,
          absXR: floats.baseAddress! + 576,
          ix: ix.baseAddress!,
          layout: layout,
          longBands: longBands,
          allowedNoise: allowedNoise
        )
        return work.runTransparent(spectrum: spectrum, blockType: blockType, searchRegions: searchRegions)
      }
    }
  }
}

/// The working state of one granule's quantization, over borrowed scratch buffers.
private struct GranuleQuantization {
  let xr34: UnsafeMutablePointer<Float>
  let absXR: UnsafeMutablePointer<Float>
  let ix: UnsafeMutablePointer<Int32>
  let layout: BandLayout
  let longBands: [Int]
  let allowedNoise: [Float]
  let huffman = HuffmanTables.flat
  let tables = QuantTables.shared

  /// Noise per band for the most recent ``distortion(_:)`` call.
  var noise: [Float]
  /// Lines past this index are zero.
  var lastLine = 0

  init(
    xr34: UnsafeMutablePointer<Float>,
    absXR: UnsafeMutablePointer<Float>,
    ix: UnsafeMutablePointer<Int32>,
    layout: BandLayout,
    longBands: [Int],
    allowedNoise: [Float]
  ) {
    self.xr34 = xr34
    self.absXR = absXR
    self.ix = ix
    self.layout = layout
    self.longBands = longBands
    self.allowedNoise = allowedNoise
    self.noise = [Float](repeating: 0, count: layout.count)
  }

  mutating func run(
    spectrum: UnsafePointer<Float>,
    blockType: BlockType,
    maxBits: Int,
    iterations: Int,
    searchRegions: Bool
  ) -> GranuleInfo {
    var info = GranuleInfo()
    info.blockType = blockType
    info.scalefactors = [Int](repeating: 0, count: layout.count)

    var peak: Float = 0
    for i in 0..<576 {
      let a = abs(spectrum[i])
      absXR[i] = a
      let p = (a * a.squareRoot()).squareRoot()
      xr34[i] = p
      if p > 1e-9 { lastLine = i + 1 }
      peak = max(peak, p)
    }

    let budget = min(maxBits, 4095)
    if peak <= 1e-9 || budget <= 0 {
      ix.update(repeating: 0, count: 576)
      info.globalGain = 210
      info.part23Length = countBits(&info)
      info.quantized = Array(UnsafeBufferPointer(start: ix, count: 576))
      return info
    }

    info.globalGain = minimumGain(info)
    searchGain(&info, budget: budget, from: info.globalGain, coarse: true)

    var bestInfo = info
    var best = distortion(info)

    if iterations > 0 && best.overCount > 0 {
      var current = info
      for _ in 0..<iterations {
        // Amplify every band whose noise exceeds the allowed noise. `noise` holds the
        // distortion of `current`.
        var amplified = false
        var exceeded = false
        var allAmplified = true
        for band in 0..<layout.count where layout.limit[band] > 0 {
          if noise[band] > allowedNoise[band] {
            current.scalefactors[band] += 1
            amplified = true
            if current.scalefactors[band] > layout.limit[band] { exceeded = true }
          }
          if current.scalefactors[band] == 0 { allAmplified = false }
        }
        guard amplified, !allAmplified else { break }
        if exceeded {
          guard current.scalefacScale == 0 else { break }
          current.scalefacScale = 1
          for band in 0..<layout.count {
            current.scalefactors[band] = (current.scalefactors[band] + 1) >> 1
          }
        }
        current.part2Length = part2Bits(&current)
        guard current.part2Length < budget else { break }

        let start = max(current.globalGain, minimumGain(current))
        guard searchGain(&current, budget: budget, from: start, coarse: false) else { break }
        let result = distortion(current)
        if result.isBetter(than: best) {
          best = result
          bestInfo = current
        }
        if result.overCount == 0 { break }
      }
    }

    info = bestInfo
    quantizeAll(info)
    tryPreflag(&info)
    info.part2Length = part2Bits(&info)
    var huffmanBits = countBits(&info)
    if searchRegions && info.blockType == .normal {
      huffmanBits = optimizeRegions(&info, total: huffmanBits)
    }
    info.part23Length = info.part2Length + huffmanBits
    info.quantized = Array(UnsafeBufferPointer(start: ix, count: 576))
    return info
  }

  /// Picks, for every band, the coarsest step whose noise stays under the allowed noise,
  /// then expresses those steps with a global gain and scalefactors.
  mutating func runTransparent(spectrum: UnsafePointer<Float>, blockType: BlockType, searchRegions: Bool) -> GranuleInfo? {
    var peak: Float = 0
    for i in 0..<576 {
      let a = abs(spectrum[i])
      absXR[i] = a
      let p = (a * a.squareRoot()).squareRoot()
      xr34[i] = p
      if p > 1e-9 { lastLine = i + 1 }
      peak = max(peak, p)
    }
    var base = GranuleInfo()
    base.blockType = blockType
    base.scalefactors = [Int](repeating: 0, count: layout.count)
    guard peak > 1e-9 else {
      ix.update(repeating: 0, count: 576)
      base.globalGain = 210
      base.part23Length = countBits(&base)
      base.quantized = Array(UnsafeBufferPointer(start: ix, count: 576))
      return base
    }

    // Required step exponent per band; `nil` where the band may be dropped entirely.
    var required = [Int?](repeating: nil, count: layout.count)
    var floor = [Int](repeating: Int.min, count: layout.count)
    for band in 0..<layout.count {
      let start = layout.start[band]
      let end = layout.end[band]
      var m: Float = 0
      var energy: Float = 0
      for i in start..<end {
        m = max(m, xr34[i])
        energy += absXR[i] * absXR[i]
      }
      guard m > 1e-9, energy > allowedNoise[band] else { continue }
      let lowest = Int((log2(m / 8206) / 0.1875).rounded(.up)) + 1
      floor[band] = lowest
      var low = lowest
      var high = lowest + 256
      if bandNoise(band, exponent: low) > allowedNoise[band] {
        required[band] = low
        continue
      }
      while high - low > 1 {
        let mid = (low + high) / 2
        if bandNoise(band, exponent: mid) <= allowedNoise[band] { low = mid } else { high = mid }
      }
      required[band] = low
    }

    var best: GranuleInfo?
    var bestBits = Int.max
    for scale in 0...1 {
      let shift = scale == 1 ? 4 : 2
      var gain = Int.min
      var ceiling = Int.max
      for band in 0..<layout.count {
        guard let e = required[band] else { continue }
        gain = max(gain, e)
        ceiling = min(ceiling, e + shift * layout.limit[band])
      }
      gain = min(max(min(gain, ceiling) + 210, 0), 255)
      var info = base
      info.scalefacScale = scale
      info.globalGain = gain
      for band in 0..<layout.count {
        guard let e = required[band] else { continue }
        var sf = min(max((gain - 210 - e + shift - 1) / shift, 0), layout.limit[band])
        while sf > 0 && gain - 210 - shift * sf < floor[band] { sf -= 1 }
        info.scalefactors[band] = sf
      }
      // A band that would still overflow forces a coarser global gain.
      while !quantizeAll(info) && info.globalGain < 255 { info.globalGain += 1 }
      tryPreflag(&info)
      info.part2Length = part2Bits(&info)
      var huffmanBits = countBits(&info)
      if searchRegions && info.blockType == .normal {
        huffmanBits = optimizeRegions(&info, total: huffmanBits)
      }
      info.part23Length = info.part2Length + huffmanBits
      if info.part23Length < bestBits {
        bestBits = info.part23Length
        info.quantized = Array(UnsafeBufferPointer(start: ix, count: 576))
        best = info
      }
    }
    guard bestBits <= 4095 else { return nil }
    return best
  }

  /// Quantization noise of one band at step exponent `e`.
  private func bandNoise(_ band: Int, exponent e: Int) -> Float {
    let inverse = tables.inverseStep(e)
    let scale = tables.step(e)
    let threshold = tables.roundingThreshold
    let pow43 = tables.pow43
    var sum: Float = 0
    for i in layout.start[band]..<layout.end[band] {
      let v = xr34[i] * inverse
      var q = Int(v)
      if v - Float(q) >= threshold[q] { q += 1 }
      let d = absXR[i] - pow43[q] * scale
      sum += d * d
    }
    return sum
  }

  // MARK: - Gain search

  /// Quantizer step exponent for a band, in units of 2^(1/4).
  @inline(__always)
  func stepExponent(_ info: GranuleInfo, band: Int) -> Int {
    let shift = info.scalefacScale == 1 ? 4 : 2
    let pre = info.preflag == 1 ? layout.pretab[band] : 0
    return info.globalGain - 210 - shift * (info.scalefactors[band] + pre)
  }

  /// Lowest global gain at which no value exceeds ``Quantizer/maxQuantized``.
  func minimumGain(_ info: GranuleInfo) -> Int {
    var gain = 0
    var probe = info
    probe.globalGain = 210
    for band in 0..<layout.count {
      var m: Float = 0
      for i in layout.start[band]..<layout.end[band] { m = max(m, xr34[i]) }
      guard m > 1e-9 else { continue }
      // m * 2^(-3/16 · e) must stay below 8206 after rounding.
      let needed = Int((log2(m / 8206) / 0.1875).rounded(.up)) + 1
      gain = max(gain, needed - stepExponent(probe, band: band) + 210)
    }
    return min(max(gain, 0), 255)
  }

  /// Finds the smallest global gain from `start` whose bit count fits `budget`, leaving
  /// `ix` quantized at that gain. Returns `false` if nothing fits.
  @discardableResult
  mutating func searchGain(_ info: inout GranuleInfo, budget: Int, from start: Int, coarse: Bool) -> Bool {
    let huffBudget = budget - info.part2Length
    guard huffBudget >= 0 else { return false }

    func fits(_ gain: Int) -> Bool {
      info.globalGain = gain
      guard quantizeAll(info) else { return false }
      return countBits(&info) <= huffBudget
    }

    if fits(start) { return true }
    // Gallop upward, then bisect.
    var low = start
    var step = coarse ? 16 : 1
    var high = min(start + step, 255)
    while !fits(high) {
      if high == 255 { return false }
      low = high
      step *= 2
      high = min(high + step, 255)
    }
    var highFits = true
    while high - low > 1 {
      let mid = (low + high) / 2
      if fits(mid) {
        high = mid
        highFits = true
      } else {
        low = mid
        highFits = false
      }
    }
    if !highFits { _ = fits(high) }
    return true
  }

  /// Quantizes all bands into `ix`. Returns `false` if any value overflows.
  @discardableResult
  func quantizeAll(_ info: GranuleInfo) -> Bool {
    var ok = true
    let threshold = tables.roundingThreshold
    for band in 0..<layout.count {
      let start = layout.start[band]
      let end = layout.end[band]
      if start >= lastLine {
        for i in start..<end { ix[i] = 0 }
        continue
      }
      let step = tables.inverseStep(stepExponent(info, band: band))
      for i in start..<end {
        let v = xr34[i] * step
        if v >= 8206 {
          ok = false
          ix[i] = Quantizer.maxQuantized
          continue
        }
        var q = Int32(v)
        if v - Float(q) >= threshold[Int(q)] { q += 1 }
        ix[i] = q
      }
    }
    return ok
  }

  // MARK: - Distortion

  /// Measures noise per band (stored in `noise`) for the quantization in `ix`.
  mutating func distortion(_ info: GranuleInfo) -> Distortion {
    var result = Distortion()
    let pow43 = tables.pow43
    for band in 0..<layout.count {
      let scale = tables.step(stepExponent(info, band: band))
      var sum: Float = 0
      for i in layout.start[band]..<layout.end[band] {
        let d = absXR[i] - pow43[Int(ix[i])] * scale
        sum += d * d
      }
      noise[band] = sum
      let ratio = 10 * log10(max(sum, 1e-30) / max(allowedNoise[band], 1e-30))
      result.maxNoise = max(result.maxNoise, ratio)
      if ratio > 0 {
        result.overCount += 1
        result.overNoise += ratio
      }
    }
    return result
  }

  // MARK: - Scalefactors

  /// Chooses `scalefac_compress` and returns the scalefactor bit count.
  func part2Bits(_ info: inout GranuleInfo) -> Int {
    var max1 = 0
    var max2 = 0
    let split = info.isShort ? 18 : 11
    for band in 0..<layout.count where layout.limit[band] > 0 {
      if band < split { max1 = max(max1, info.scalefactors[band]) } else { max2 = max(max2, info.scalefactors[band]) }
    }
    var bestIndex = 15
    var bestBits = Int.max
    for index in 0..<16 {
      let (slen1, slen2) = MPEG.slen[index]
      guard max1 < (1 << slen1) && max2 < (1 << slen2) else { continue }
      let bits = info.isShort ? 18 * (slen1 + slen2) : 11 * slen1 + 10 * slen2
      if bits < bestBits {
        bestBits = bits
        bestIndex = index
      }
    }
    info.scalefacCompress = bestIndex
    return bestBits == Int.max ? 4096 : bestBits
  }

  /// Moves part of the high-band scalefactors into the pre-emphasis flag when that saves bits.
  func tryPreflag(_ info: inout GranuleInfo) {
    guard !info.isShort, info.preflag == 0 else { return }
    for band in 11..<21 where info.scalefactors[band] < layout.pretab[band] {
      return
    }
    var candidate = info
    for band in 11..<21 {
      candidate.scalefactors[band] -= layout.pretab[band]
    }
    candidate.preflag = 1
    var current = info
    if part2Bits(&candidate) < part2Bits(&current) {
      info = candidate
    }
  }

  // MARK: - Bit counting

  /// Sets the region layout and table selection from `ix`; returns the Huffman bit count.
  func countBits(_ info: inout GranuleInfo) -> Int {
    var i = min(576, (lastLine + 1) & ~1)
    while i > 1 && ix[i - 1] == 0 && ix[i - 2] == 0 { i -= 2 }
    var count1 = 0
    while i > 3 && ix[i - 1] <= 1 && ix[i - 2] <= 1 && ix[i - 3] <= 1 && ix[i - 4] <= 1 {
      i -= 4
      count1 += 1
    }
    info.count1 = count1
    info.bigValues = i / 2

    let quads = huffman.countQuadruples(ix, from: i, count: count1)
    info.count1TableSelect = quads.b < quads.a ? 1 : 0
    var bits = min(quads.a, quads.b)

    let bigEnd = i
    if info.blockType == .normal {
      if bigEnd == 0 {
        info.region0Count = 0
        info.region1Count = 0
      } else {
        var bands = 0
        while longBands[bands] < bigEnd { bands += 1 }
        var r0 = MPEG.subdivision[bands].0
        while r0 > 0 && longBands[r0 + 1] > bigEnd { r0 -= 1 }
        var r1 = MPEG.subdivision[bands].1
        while r1 > 0 && longBands[r0 + r1 + 2] > bigEnd { r1 -= 1 }
        info.region0Count = r0
        info.region1Count = r1
      }
    } else {
      info.region0Count = 7
      info.region1Count = 0
    }
    let (a1, a2) = info.regionBoundaries(longBands: longBands)
    let r1 = min(a1, bigEnd)
    let r2 = min(a2, bigEnd)
    let t0 = huffman.chooseTable(ix, 0, r1)
    let t1 = huffman.chooseTable(ix, r1, r2)
    let t2 = huffman.chooseTable(ix, r2, bigEnd)
    info.tableSelect = [t0.table, t1.table, t2.table]
    bits += t0.bits + t1.bits + t2.bits
    return bits
  }

  /// Searches all region splits for the fewest big-values bits; returns the Huffman bit count.
  func optimizeRegions(_ info: inout GranuleInfo, total: Int) -> Int {
    let bigEnd = info.bigValues * 2
    guard bigEnd > 0 else { return total }
    let (a1, a2) = info.regionBoundaries(longBands: longBands)
    let currentBig = huffman.count(ix, 0, min(a1, bigEnd), table: info.tableSelect[0])
      + huffman.count(ix, min(a1, bigEnd), min(a2, bigEnd), table: info.tableSelect[1])
      + huffman.count(ix, min(a2, bigEnd), bigEnd, table: info.tableSelect[2])
    let count1Bits = total - currentBig

    var bestBits = currentBig
    var best = (info.region0Count, info.region1Count, info.tableSelect)
    var r2Cost = [(table: Int, bits: Int)]()
    r2Cost.reserveCapacity(23)
    for boundary in 0..<23 {
      r2Cost.append(huffman.chooseTable(ix, min(longBands[boundary], bigEnd), bigEnd))
    }
    for r0 in 0..<16 {
      let a1 = min(longBands[r0 + 1], bigEnd)
      let c0 = huffman.chooseTable(ix, 0, a1)
      if c0.bits >= bestBits { break }
      for r1 in 0..<8 where r0 + r1 + 2 <= 22 {
        let a2 = min(longBands[r0 + r1 + 2], bigEnd)
        let c1 = huffman.chooseTable(ix, a1, a2)
        let bits = c0.bits + c1.bits + r2Cost[r0 + r1 + 2].bits
        if bits < bestBits {
          bestBits = bits
          best = (r0, r1, [c0.table, c1.table, r2Cost[r0 + r1 + 2].table])
        }
        if a2 >= bigEnd { break }
      }
      if a1 >= bigEnd { break }
    }
    info.region0Count = best.0
    info.region1Count = best.1
    info.tableSelect = best.2
    return count1Bits + bestBits
  }
}

/// Precomputed quantization tables in permanently allocated memory.
struct QuantTables: @unchecked Sendable {
  /// q^(4/3) for q in 0...8207.
  let pow43: UnsafePointer<Float>
  /// Rounding thresholds that pick the reconstruction level closest in the linear domain:
  /// round `v` up when `v - floor(v)` is at least this value.
  let roundingThreshold: UnsafePointer<Float>
  private let inverseSteps: UnsafePointer<Float>
  private let steps: UnsafePointer<Float>
  private static let exponentOffset = 400

  static let shared = QuantTables()

  private init() {
    let count = Int(Quantizer.maxQuantized) + 2
    let pow43 = UnsafeMutablePointer<Float>.allocate(capacity: count)
    let threshold = UnsafeMutablePointer<Float>.allocate(capacity: count)
    for q in 0..<count {
      let lower = pow(Double(q), 4.0 / 3.0)
      let upper = pow(Double(q + 1), 4.0 / 3.0)
      pow43[q] = Float(lower)
      threshold[q] = Float(pow((lower + upper) / 2, 0.75) - Double(q))
    }
    let inverse = UnsafeMutablePointer<Float>.allocate(capacity: 512)
    let steps = UnsafeMutablePointer<Float>.allocate(capacity: 512)
    for i in 0..<512 {
      inverse[i] = Float(pow(2.0, -0.1875 * Double(i - Self.exponentOffset)))
      steps[i] = Float(pow(2.0, 0.25 * Double(i - Self.exponentOffset)))
    }
    self.pow43 = UnsafePointer(pow43)
    self.roundingThreshold = UnsafePointer(threshold)
    self.inverseSteps = UnsafePointer(inverse)
    self.steps = UnsafePointer(steps)
  }

  /// 2^(-3/16 · e): multiplies |x|^(3/4) to give the quantized value.
  @inline(__always)
  func inverseStep(_ e: Int) -> Float { inverseSteps[min(max(e + Self.exponentOffset, 0), 511)] }

  /// 2^(e/4): multiplies q^(4/3) to give the reconstructed value.
  @inline(__always)
  func step(_ e: Int) -> Float { steps[min(max(e + Self.exponentOffset, 0), 511)] }
}
