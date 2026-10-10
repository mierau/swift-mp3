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
    var maxValue: Int32 = 0
    for i in begin..<end where ix[i] > maxValue { maxValue = ix[i] }

    switch maxValue {
    case 0: return (0, 0)
    case 1: return (1, count(ix, begin, end, table: 1))
    case 2: return best(ix, begin, end, 2, 3)
    case 3: return best(ix, begin, end, 5, 6)
    case 4...5: return best(ix, begin, end, 7, 8, 9)
    case 6...7: return best(ix, begin, end, 10, 11, 12)
    case 8...15: return best(ix, begin, end, 13, 15)
    default:
      // Escape tables: the smallest of each family whose linbits cover the value.
      var a = 16
      while a < 23 && escapeMaxValues[a] < maxValue { a += 1 }
      var b = 24
      while b < 31 && escapeMaxValues[b] < maxValue { b += 1 }
      let bitsA = count(ix, begin, end, table: a)
      let bitsB = count(ix, begin, end, table: b)
      return bitsA <= bitsB ? (a, bitsA) : (b, bitsB)
    }
  }

  /// Counts two or three non-escape tables in one pass and returns the cheapest.
  @inline(__always)
  private func best(_ ix: UnsafePointer<Int32>, _ begin: Int, _ end: Int, _ t1: Int, _ t2: Int, _ t3: Int = -1) -> (Int, Int) {
    let l1 = lengths + t1 * 256
    let l2 = lengths + t2 * 256
    let l3 = lengths + max(t3, 0) * 256
    var b1 = 0
    var b2 = 0
    var b3 = 0
    var signs = 0
    var i = begin
    while i < end {
      let x = Int(ix[i])
      let y = Int(ix[i + 1])
      let index = x * 16 + y
      b1 += Int(l1[index])
      b2 += Int(l2[index])
      b3 += Int(l3[index])
      signs += (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0)
      i += 2
    }
    var result = (t1, b1)
    if b2 < result.1 { result = (t2, b2) }
    if t3 >= 0 && b3 < result.1 { result = (t3, b3) }
    return (result.0, result.1 + signs)
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
  /// Non-escape tables, one cost column each.
  private static let plainTables = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
  private static let escape16 = 13
  private static let escape24 = 14
  private static let signColumn = 15
  private static let escapeColumn = 16
  private static let columns = 17

  /// Largest value in each band.
  private var bandMax: [Int32]
  /// Line index of each band boundary, clipped to the end of the region.
  private var points: [Int]
  /// Running sums at each boundary, `[point * columns + column]`.
  private var prefix: [Int]
  private let tables: FlatHuffmanTables

  init(_ ix: UnsafePointer<Int32>, end: Int, bands: [Int], tables: FlatHuffmanTables) {
    self.tables = tables
    let count = bands.count
    points = bands.map { min($0, end) }
    bandMax = [Int32](repeating: 0, count: count)
    prefix = [Int](repeating: 0, count: count * Self.columns)
    let columns = Self.columns
    var sums = [Int](repeating: 0, count: columns)
    for k in 0..<(count - 1) {
      var maxValue: Int32 = 0
      var i = points[k]
      while i < points[k + 1] {
        let x = ix[i]
        let y = ix[i + 1]
        maxValue = max(maxValue, x, y)
        let index = Int(min(x, 15)) * 16 + Int(min(y, 15))
        for (column, table) in Self.plainTables.enumerated() {
          sums[column] += Int(tables.lengths[table * 256 + index])
        }
        sums[Self.escape16] += Int(tables.lengths[16 * 256 + index])
        sums[Self.escape24] += Int(tables.lengths[24 * 256 + index])
        sums[Self.signColumn] += (x != 0 ? 1 : 0) + (y != 0 ? 1 : 0)
        sums[Self.escapeColumn] += (x >= 15 ? 1 : 0) + (y >= 15 ? 1 : 0)
        i += 2
      }
      bandMax[k] = maxValue
      for column in 0..<columns {
        prefix[(k + 1) * columns + column] = sums[column]
      }
    }
  }

  @inline(__always)
  private func sum(_ column: Int, _ from: Int, _ to: Int) -> Int {
    prefix[to * Self.columns + column] - prefix[from * Self.columns + column]
  }

  @inline(__always)
  private func plain(_ table: Int, _ from: Int, _ to: Int) -> Int {
    sum(Self.plainTables.firstIndex(of: table)!, from, to)
  }

  /// The same choice as ``FlatHuffmanTables/chooseTable(_:_:_:)`` for the lines between
  /// band boundaries `from` and `to`.
  func chooseTable(from: Int, to: Int) -> (table: Int, bits: Int) {
    guard points[from] < points[to] else { return (0, 0) }
    var maxValue: Int32 = 0
    for k in from..<to { maxValue = max(maxValue, bandMax[k]) }
    let signs = sum(Self.signColumn, from, to)

    func best(_ candidates: [Int]) -> (Int, Int) {
      var result = (candidates[0], plain(candidates[0], from, to))
      for table in candidates.dropFirst() {
        let bits = plain(table, from, to)
        if bits < result.1 { result = (table, bits) }
      }
      return (result.0, result.1 + signs)
    }

    switch maxValue {
    case 0: return (0, 0)
    case 1: return (1, plain(1, from, to) + signs)
    case 2: return best([2, 3])
    case 3: return best([5, 6])
    case 4...5: return best([7, 8, 9])
    case 6...7: return best([10, 11, 12])
    case 8...15: return best([13, 15])
    default:
      var a = 16
      while a < 23 && escapeMaxValues[a] < maxValue { a += 1 }
      var b = 24
      while b < 31 && escapeMaxValues[b] < maxValue { b += 1 }
      let escapes = sum(Self.escapeColumn, from, to)
      let bitsA = sum(Self.escape16, from, to) + signs + escapes * tables.linbits[a]
      let bitsB = sum(Self.escape24, from, to) + signs + escapes * tables.linbits[b]
      return bitsA <= bitsB ? (a, bitsA) : (b, bitsB)
    }
  }
}

enum HuffmanWriter {
  /// Writes the big-values and count1 regions of a granule.
  static func write(
    _ info: GranuleInfo,
    spectrum: UnsafePointer<Float>,
    longBands: [Int],
    into writer: inout BitWriter
  ) {
    let tables = HuffmanTables.flat
    info.quantized.withUnsafeBufferPointer { q in
      let ix = q.baseAddress!
      let bigEnd = info.bigValues * 2
      let (a1, a2) = info.regionBoundaries(longBands: longBands)
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
