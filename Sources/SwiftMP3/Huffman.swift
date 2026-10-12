// SwiftMP3
// Huffman.swift
//
// Huffman table selection, bit counting, and coding for quantized spectra.

/// Largest value representable by each escape table (16-31).
private let escapeMaxValues: [Int32] = (0..<32).map { t in
  let linbits = HuffmanTables.tables[t].linbits
  return linbits > 0 ? Int32(15 + (1 << linbits) - 1) : 15
}

extension FlatHuffmanTables {
  /// Picks the cheapest table for the pairs in `[begin, end)` and returns its bit count.
  @inline(__always)
  func chooseTable(_ ix: UnsafePointer<Int32>, _ begin: Int, _ end: Int) -> (table: Int, bits: Int) {
    guard begin < end else { return (0, 0) }
    // Count the pairs in every table at once (at most 288 pairs of at most 19 bits per lane).
    var costs = SIMD16<UInt16>()
    var maxValue: Int32 = 0
    var i = begin
    while i < end {
      let x = ix[i]
      let y = ix[i + 1]
      maxValue = max(maxValue, x, y)
      costs &+= pairCosts[Int(min(x, 15)) * 16 + Int(min(y, 15))]
      i += 2
    }
    var escapes = 0
    if maxValue >= 15 {
      for i in begin..<end where ix[i] >= 15 { escapes += 1 }
    }
    return cheapest(costs, maxValue: maxValue, escapes: escapes)
  }

  /// Bits needed to code the pairs in `[begin, end)` with `table`, including sign and escape bits.
  @inline(__always)
  func count(_ ix: UnsafePointer<Int32>, _ begin: Int, _ end: Int, table: Int) -> Int {
    guard begin < end, table != 0 else { return 0 }
    let l = lengths + table * 256
    let linbits = self.linbits[table]
    var bits = 0
    var i = begin
    if linbits == 0 {
      while i < end {
        let x = Int(ix[i])
        let y = Int(ix[i + 1])
        bits += Int(l[x * 16 + y]) + (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0)
        i += 2
      }
    } else {
      while i < end {
        var x = Int(ix[i])
        var y = Int(ix[i + 1])
        bits += (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0)
        if x >= 15 { x = 15; bits += linbits }
        if y >= 15 { y = 15; bits += linbits }
        bits += Int(l[x * 16 + y])
        i += 2
      }
    }
    return bits
  }

  /// Bits for the count1 region with tables A and B.
  @inline(__always)
  func countQuadruples(_ ix: UnsafePointer<Int32>, from start: Int, count: Int) -> (a: Int, b: Int) {
    let a = lengths + 32 * 256
    var bitsA = 0
    var signs = 0
    var i = start
    for _ in 0..<count {
      let index = Int(ix[i]) * 8 + Int(ix[i + 1]) * 4 + Int(ix[i + 2]) * 2 + Int(ix[i + 3])
      bitsA += Int(a[index])
      signs += Int(ix[i] + ix[i + 1] + ix[i + 2] + ix[i + 3])
      i += 4
    }
    return (bitsA + signs, count * 4 + signs)
  }
}

/// Huffman costs of a big-values region, summed per scalefactor band, so that the cost of
/// any run of bands with any table is a difference of prefix sums.
struct RegionCosts {
  /// Largest value in each band.
  private var bandMax: [Int32]
  /// Line index of each band boundary, clipped to the end of the region.
  private var points: [Int]
  /// Running sums of ``FlatHuffmanTables/pairCosts`` at each boundary. A region has at
  /// most 288 pairs of at most 19 bits, so the lanes can't overflow.
  private var prefix: [SIMD16<UInt16>]
  /// Running count of values that need escape bits (15 or more) at each boundary.
  private var escapes: [Int]
  private let tables: FlatHuffmanTables

  init(_ ix: UnsafePointer<Int32>, end: Int, bands: [Int], tables: FlatHuffmanTables) {
    self.tables = tables
    let count = bands.count
    points = bands.map { min($0, end) }
    bandMax = [Int32](repeating: 0, count: count)
    prefix = [SIMD16<UInt16>](repeating: .zero, count: count)
    escapes = [Int](repeating: 0, count: count)
    var sums = SIMD16<UInt16>()
    var escapeSum = 0
    for k in 0..<(count - 1) {
      var maxValue: Int32 = 0
      var i = points[k]
      while i < points[k + 1] {
        let x = ix[i]
        let y = ix[i + 1]
        maxValue = max(maxValue, x, y)
        sums &+= tables.pairCosts[Int(min(x, 15)) * 16 + Int(min(y, 15))]
        escapeSum += (x >= 15 ? 1 : 0) + (y >= 15 ? 1 : 0)
        i += 2
      }
      bandMax[k] = maxValue
      prefix[k + 1] = sums
      escapes[k + 1] = escapeSum
    }
  }

  /// The same choice as ``FlatHuffmanTables/chooseTable(_:_:_:)`` for the lines between
  /// band boundaries `from` and `to`.
  func chooseTable(from: Int, to: Int) -> (table: Int, bits: Int) {
    guard points[from] < points[to] else { return (0, 0) }
    var maxValue: Int32 = 0
    for k in from..<to { maxValue = max(maxValue, bandMax[k]) }
    return tables.cheapest(prefix[to] &- prefix[from], maxValue: maxValue, escapes: escapes[to] - escapes[from])
  }
}

extension FlatHuffmanTables {
  /// The cheapest table for pairs whose costs sum to `costs` (lanes of ``pairCosts``),
  /// with the same choice and tie-breaking as ``chooseTable(_:_:_:)``.
  @inline(__always)
  func cheapest(_ costs: SIMD16<UInt16>, maxValue: Int32, escapes: Int) -> (table: Int, bits: Int) {
    @inline(__always)
    func bits(_ table: Int) -> Int { Int(costs[Self.lane(of: table)]) }
    @inline(__always)
    func best(_ t1: Int, _ t2: Int, _ t3: Int = -1) -> (Int, Int) {
      var result = (t1, bits(t1))
      if bits(t2) < result.1 { result = (t2, bits(t2)) }
      if t3 >= 0 && bits(t3) < result.1 { result = (t3, bits(t3)) }
      return (result.0, result.1 + Int(costs[Self.signLane]))
    }

    switch maxValue {
    case 0: return (0, 0)
    case 1: return (1, bits(1) + Int(costs[Self.signLane]))
    case 2: return best(2, 3)
    case 3: return best(5, 6)
    case 4...5: return best(7, 8, 9)
    case 6...7: return best(10, 11, 12)
    case 8...15: return best(13, 15)
    default:
      var a = 16
      while a < 23 && escapeMaxValues[a] < maxValue { a += 1 }
      var b = 24
      while b < 31 && escapeMaxValues[b] < maxValue { b += 1 }
      let signs = Int(costs[Self.signLane])
      let bitsA = bits(16) + signs + escapes * linbits[a]
      let bitsB = bits(24) + signs + escapes * linbits[b]
      return bitsA <= bitsB ? (a, bitsA) : (b, bitsB)
    }
  }
}

enum HuffmanWriter {
  /// Writes the big-values and count1 regions of a granule.
  static func write(
    _ info: GranuleInfo,
    spectrum: UnsafePointer<Float>,
    regions: RegionLayout,
    into writer: inout BitWriter
  ) {
    let tables = HuffmanTables.flat
    info.quantized.withUnsafeBufferPointer { q in
      let ix = q.baseAddress!
      let bigEnd = info.bigValues * 2
      let (a1, a2) = info.regionBoundaries(regions)
      let r1 = min(a1, bigEnd)
      let r2 = min(a2, bigEnd)
      writePairs(ix, spectrum, 0, r1, table: info.tableSelect[0], tables, &writer)
      writePairs(ix, spectrum, r1, r2, table: info.tableSelect[1], tables, &writer)
      writePairs(ix, spectrum, r2, bigEnd, table: info.tableSelect[2], tables, &writer)

      let quadTable = (32 + info.count1TableSelect) * 256
      var i = bigEnd
      for _ in 0..<info.count1 {
        let index = quadTable + Int(ix[i]) * 8 + Int(ix[i + 1]) * 4 + Int(ix[i + 2]) * 2 + Int(ix[i + 3])
        writer.write(tables.codes[index], Int(tables.lengths[index]))
        for k in 0..<4 where ix[i + k] != 0 {
          writer.write(spectrum[i + k] < 0 ? 1 : 0, 1)
        }
        i += 4
      }
    }
  }

  private static func writePairs(
    _ ix: UnsafePointer<Int32>,
    _ spectrum: UnsafePointer<Float>,
    _ begin: Int,
    _ end: Int,
    table: Int,
    _ tables: FlatHuffmanTables,
    _ writer: inout BitWriter
  ) {
    guard begin < end, table != 0 else { return }
    let linbits = tables.linbits[table]
    let base = table * 256
    var i = begin
    while i < end {
      let x = Int(ix[i])
      let y = Int(ix[i + 1])
      let cx = linbits > 0 ? min(x, 15) : x
      let cy = linbits > 0 ? min(y, 15) : y
      let entry = base + cx * 16 + cy
      writer.write(tables.codes[entry], Int(tables.lengths[entry]))
      if linbits > 0 && x >= 15 { writer.write(x - 15, linbits) }
      if x != 0 { writer.write(spectrum[i] < 0 ? 1 : 0, 1) }
      if linbits > 0 && y >= 15 { writer.write(y - 15, linbits) }
      if y != 0 { writer.write(spectrum[i + 1] < 0 ? 1 : 0, 1) }
      i += 2
    }
  }
}
