// SwiftMP3
// HuffmanTables.swift
//
// Huffman code tables from ISO/IEC 11172-3 Annex B, Table B.7.

/// A Layer III Huffman code table.
struct HuffmanTable: Sendable {
  /// Number of values per dimension (the table covers `0..<xlen` for x and y).
  let xlen: Int
  /// Number of escape bits appended for values of 15 or more (tables 16-31).
  let linbits: Int
  /// Codewords indexed by `x * xlen + y`.
  let codes: [UInt32]
  /// Codeword lengths indexed by `x * xlen + y`.
  let lengths: [UInt8]

  /// The largest absolute value this table can represent.
  var maxValue: Int { linbits > 0 ? 15 + (1 << linbits) - 1 : xlen - 1 }
}

enum HuffmanTables {
  private static let codes1: [UInt32] = [
    1, 1, 1, 0,
  ]
  private static let lengths1: [UInt8] = [
    1, 3, 2, 3,
  ]
  private static let codes2: [UInt32] = [
    1, 2, 1, 3, 1, 1, 3, 2, 0,
  ]
  private static let lengths2: [UInt8] = [
    1, 3, 6, 3, 3, 5, 5, 5, 6,
  ]
  private static let codes3: [UInt32] = [
    3, 2, 1, 1, 1, 1, 3, 2, 0,
  ]
  private static let lengths3: [UInt8] = [
    2, 2, 6, 3, 2, 5, 5, 5, 6,
  ]
  private static let codes5: [UInt32] = [
    1, 2, 6, 5, 3, 1, 4, 4, 7, 5, 7, 1, 6, 1, 1, 0,
  ]
  private static let lengths5: [UInt8] = [
    1, 3, 6, 7, 3, 3, 6, 7, 6, 6, 7, 8, 7, 6, 7, 8,
  ]
  private static let codes6: [UInt32] = [
    7, 3, 5, 1, 6, 2, 3, 2, 5, 4, 4, 1, 3, 3, 2, 0,
  ]
  private static let lengths6: [UInt8] = [
    3, 3, 5, 7, 3, 2, 4, 5, 4, 4, 5, 6, 6, 5, 6, 7,
  ]
  private static let codes7: [UInt32] = [
    1, 2, 10, 19, 16, 10, 3, 3, 7, 10, 5, 3, 11, 4, 13, 17,
    8, 4, 12, 11, 18, 15, 11, 2, 7, 6, 9, 14, 3, 1, 6, 4,
    5, 3, 2, 0,
  ]
  private static let lengths7: [UInt8] = [
    1, 3, 6, 8, 8, 9, 3, 4, 6, 7, 7, 8, 6, 5, 7, 8,
    8, 9, 7, 7, 8, 9, 9, 9, 7, 7, 8, 9, 9, 10, 8, 8,
    9, 10, 10, 10,
  ]
  private static let codes8: [UInt32] = [
    3, 4, 6, 18, 12, 5, 5, 1, 2, 16, 9, 3, 7, 3, 5, 14,
    7, 3, 19, 17, 15, 13, 10, 4, 13, 5, 8, 11, 5, 1, 12, 4,
    4, 1, 1, 0,
  ]
  private static let lengths8: [UInt8] = [
    2, 3, 6, 8, 8, 9, 3, 2, 4, 8, 8, 8, 6, 4, 6, 8,
    8, 9, 8, 8, 8, 9, 9, 10, 8, 7, 8, 9, 10, 10, 9, 8,
    9, 9, 11, 11,
  ]
  private static let codes9: [UInt32] = [
    7, 5, 9, 14, 15, 7, 6, 4, 5, 5, 6, 7, 7, 6, 8, 8,
    8, 5, 15, 6, 9, 10, 5, 1, 11, 7, 9, 6, 4, 1, 14, 4,
    6, 2, 6, 0,
  ]
  private static let lengths9: [UInt8] = [
    3, 3, 5, 6, 8, 9, 3, 3, 4, 5, 6, 8, 4, 4, 5, 6,
    7, 8, 6, 5, 6, 7, 7, 8, 7, 6, 7, 7, 8, 9, 8, 7,
    8, 8, 9, 9,
  ]
  private static let codes10: [UInt32] = [
    1, 2, 10, 23, 35, 30, 12, 17, 3, 3, 8, 12, 18, 21, 12, 7,
    11, 9, 15, 21, 32, 40, 19, 6, 14, 13, 22, 34, 46, 23, 18, 7,
    20, 19, 33, 47, 27, 22, 9, 3, 31, 22, 41, 26, 21, 20, 5, 3,
    14, 13, 10, 11, 16, 6, 5, 1, 9, 8, 7, 8, 4, 4, 2, 0,
  ]
  private static let lengths10: [UInt8] = [
    1, 3, 6, 8, 9, 9, 9, 10, 3, 4, 6, 7, 8, 9, 8, 8,
    6, 6, 7, 8, 9, 10, 9, 9, 7, 7, 8, 9, 10, 10, 9, 10,
    8, 8, 9, 10, 10, 10, 10, 10, 9, 9, 10, 10, 11, 11, 10, 11,
    8, 8, 9, 10, 10, 10, 11, 11, 9, 8, 9, 10, 10, 11, 11, 11,
  ]
  private static let codes11: [UInt32] = [
    3, 4, 10, 24, 34, 33, 21, 15, 5, 3, 4, 10, 32, 17, 11, 10,
    11, 7, 13, 18, 30, 31, 20, 5, 25, 11, 19, 59, 27, 18, 12, 5,
    35, 33, 31, 58, 30, 16, 7, 5, 28, 26, 32, 19, 17, 15, 8, 14,
    14, 12, 9, 13, 14, 9, 4, 1, 11, 4, 6, 6, 6, 3, 2, 0,
  ]
  private static let lengths11: [UInt8] = [
    2, 3, 5, 7, 8, 9, 8, 9, 3, 3, 4, 6, 8, 8, 7, 8,
    5, 5, 6, 7, 8, 9, 8, 8, 7, 6, 7, 9, 8, 10, 8, 9,
    8, 8, 8, 9, 9, 10, 9, 10, 8, 8, 9, 10, 10, 11, 10, 11,
    8, 7, 7, 8, 9, 10, 10, 10, 8, 7, 8, 9, 10, 10, 10, 10,
  ]
  private static let codes12: [UInt32] = [
    9, 6, 16, 33, 41, 39, 38, 26, 7, 5, 6, 9, 23, 16, 26, 11,
    17, 7, 11, 14, 21, 30, 10, 7, 17, 10, 15, 12, 18, 28, 14, 5,
    32, 13, 22, 19, 18, 16, 9, 5, 40, 17, 31, 29, 17, 13, 4, 2,
    27, 12, 11, 15, 10, 7, 4, 1, 27, 12, 8, 12, 6, 3, 1, 0,
  ]
  private static let lengths12: [UInt8] = [
    4, 3, 5, 7, 8, 9, 9, 9, 3, 3, 4, 5, 7, 7, 8, 8,
    5, 4, 5, 6, 7, 8, 7, 8, 6, 5, 6, 6, 7, 8, 8, 8,
    7, 6, 7, 7, 8, 8, 8, 9, 8, 7, 8, 8, 8, 9, 8, 9,
    8, 7, 7, 8, 8, 9, 9, 10, 9, 8, 8, 9, 9, 9, 9, 10,
  ]
  private static let codes13: [UInt32] = [
    1, 5, 14, 21, 34, 51, 46, 71, 42, 52, 68, 52, 67, 44, 43, 19,
    3, 4, 12, 19, 31, 26, 44, 33, 31, 24, 32, 24, 31, 35, 22, 14,
    15, 13, 23, 36, 59, 49, 77, 65, 29, 40, 30, 40, 27, 33, 42, 16,
    22, 20, 37, 61, 56, 79, 73, 64, 43, 76, 56, 37, 26, 31, 25, 14,
    35, 16, 60, 57, 97, 75, 114, 91, 54, 73, 55, 41, 48, 53, 23, 24,
    58, 27, 50, 96, 76, 70, 93, 84, 77, 58, 79, 29, 74, 49, 41, 17,
    47, 45, 78, 74, 115, 94, 90, 79, 69, 83, 71, 50, 59, 38, 36, 15,
    72, 34, 56, 95, 92, 85, 91, 90, 86, 73, 77, 65, 51, 44, 43, 42,
    43, 20, 30, 44, 55, 78, 72, 87, 78, 61, 46, 54, 37, 30, 20, 16,
    53, 25, 41, 37, 44, 59, 54, 81, 66, 76, 57, 54, 37, 18, 39, 11,
    35, 33, 31, 57, 42, 82, 72, 80, 47, 58, 55, 21, 22, 26, 38, 22,
    53, 25, 23, 38, 70, 60, 51, 36, 55, 26, 34, 23, 27, 14, 9, 7,
    34, 32, 28, 39, 49, 75, 30, 52, 48, 40, 52, 28, 18, 17, 9, 5,
    45, 21, 34, 64, 56, 50, 49, 45, 31, 19, 12, 15, 10, 7, 6, 3,
    48, 23, 20, 39, 36, 35, 53, 21, 16, 23, 13, 10, 6, 1, 4, 2,
    16, 15, 17, 27, 25, 20, 29, 11, 17, 12, 16, 8, 1, 1, 0, 1,
  ]
  private static let lengths13: [UInt8] = [
    1, 4, 6, 7, 8, 9, 9, 10, 9, 10, 11, 11, 12, 12, 13, 13,
    3, 4, 6, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 12, 12, 12,
    6, 6, 7, 8, 9, 9, 10, 10, 9, 10, 10, 11, 11, 12, 13, 13,
    7, 7, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 13,
    8, 7, 9, 9, 10, 10, 11, 11, 10, 11, 11, 12, 12, 13, 13, 14,
    9, 8, 9, 10, 10, 10, 11, 11, 11, 11, 12, 11, 13, 13, 14, 14,
    9, 9, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14,
    10, 9, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 16, 16,
    9, 8, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 15, 15,
    10, 9, 10, 10, 11, 11, 11, 13, 12, 13, 13, 14, 14, 14, 16, 15,
    10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 14, 13, 14, 15, 16, 17,
    11, 10, 10, 11, 12, 12, 12, 12, 13, 13, 13, 14, 15, 15, 15, 16,
    11, 11, 11, 12, 12, 13, 12, 13, 14, 14, 15, 15, 15, 16, 16, 16,
    12, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 16, 15, 16, 16,
    13, 12, 12, 13, 13, 13, 15, 14, 14, 17, 15, 15, 15, 17, 16, 16,
    12, 12, 13, 14, 14, 14, 15, 14, 15, 15, 16, 16, 19, 18, 19, 16,
  ]
  private static let codes15: [UInt32] = [
    7, 12, 18, 53, 47, 76, 124, 108, 89, 123, 108, 119, 107, 81, 122, 63,
    13, 5, 16, 27, 46, 36, 61, 51, 42, 70, 52, 83, 65, 41, 59, 36,
    19, 17, 15, 24, 41, 34, 59, 48, 40, 64, 50, 78, 62, 80, 56, 33,
    29, 28, 25, 43, 39, 63, 55, 93, 76, 59, 93, 72, 54, 75, 50, 29,
    52, 22, 42, 40, 67, 57, 95, 79, 72, 57, 89, 69, 49, 66, 46, 27,
    77, 37, 35, 66, 58, 52, 91, 74, 62, 48, 79, 63, 90, 62, 40, 38,
    125, 32, 60, 56, 50, 92, 78, 65, 55, 87, 71, 51, 73, 51, 70, 30,
    109, 53, 49, 94, 88, 75, 66, 122, 91, 73, 56, 42, 64, 44, 21, 25,
    90, 43, 41, 77, 73, 63, 56, 92, 77, 66, 47, 67, 48, 53, 36, 20,
    71, 34, 67, 60, 58, 49, 88, 76, 67, 106, 71, 54, 38, 39, 23, 15,
    109, 53, 51, 47, 90, 82, 58, 57, 48, 72, 57, 41, 23, 27, 62, 9,
    86, 42, 40, 37, 70, 64, 52, 43, 70, 55, 42, 25, 29, 18, 11, 11,
    118, 68, 30, 55, 50, 46, 74, 65, 49, 39, 24, 16, 22, 13, 14, 7,
    91, 44, 39, 38, 34, 63, 52, 45, 31, 52, 28, 19, 14, 8, 9, 3,
    123, 60, 58, 53, 47, 43, 32, 22, 37, 24, 17, 12, 15, 10, 2, 1,
    71, 37, 34, 30, 28, 20, 17, 26, 21, 16, 10, 6, 8, 6, 2, 0,
  ]
  private static let lengths15: [UInt8] = [
    3, 4, 5, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 13,
    4, 3, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11,
    5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11,
    6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
    7, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11,
    8, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 12,
    9, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12,
    9, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12,
    9, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 12,
    9, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12,
    10, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 12,
    10, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13,
    11, 10, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13,
    11, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13,
    12, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 12, 13,
    12, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13,
  ]
  private static let codes16: [UInt32] = [
    1, 5, 14, 44, 74, 63, 110, 93, 172, 149, 138, 242, 225, 195, 376, 17,
    3, 4, 12, 20, 35, 62, 53, 47, 83, 75, 68, 119, 201, 107, 207, 9,
    15, 13, 23, 38, 67, 58, 103, 90, 161, 72, 127, 117, 110, 209, 206, 16,
    45, 21, 39, 69, 64, 114, 99, 87, 158, 140, 252, 212, 199, 387, 365, 26,
    75, 36, 68, 65, 115, 101, 179, 164, 155, 264, 246, 226, 395, 382, 362, 9,
    66, 30, 59, 56, 102, 185, 173, 265, 142, 253, 232, 400, 388, 378, 445, 16,
    111, 54, 52, 100, 184, 178, 160, 133, 257, 244, 228, 217, 385, 366, 715, 10,
    98, 48, 91, 88, 165, 157, 148, 261, 248, 407, 397, 372, 380, 889, 884, 8,
    85, 84, 81, 159, 156, 143, 260, 249, 427, 401, 392, 383, 727, 713, 708, 7,
    154, 76, 73, 141, 131, 256, 245, 426, 406, 394, 384, 735, 359, 710, 352, 11,
    139, 129, 67, 125, 247, 233, 229, 219, 393, 743, 737, 720, 885, 882, 439, 4,
    243, 120, 118, 115, 227, 223, 396, 746, 742, 736, 721, 712, 706, 223, 436, 6,
    202, 224, 222, 218, 216, 389, 386, 381, 364, 888, 443, 707, 440, 437, 1728, 4,
    747, 211, 210, 208, 370, 379, 734, 723, 714, 1735, 883, 877, 876, 3459, 865, 2,
    377, 369, 102, 187, 726, 722, 358, 711, 709, 866, 1734, 871, 3458, 870, 434, 0,
    12, 10, 7, 11, 10, 17, 11, 9, 13, 12, 10, 7, 5, 3, 1, 3,
  ]
  private static let lengths16: [UInt8] = [
    1, 4, 6, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 9,
    3, 4, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11, 12, 11, 12, 8,
    6, 6, 7, 8, 9, 9, 10, 10, 11, 10, 11, 11, 11, 12, 12, 9,
    8, 7, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 10,
    9, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 9,
    9, 8, 9, 9, 10, 11, 11, 12, 11, 12, 12, 13, 13, 13, 14, 10,
    10, 9, 9, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 14, 10,
    10, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 13, 15, 15, 10,
    10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 14, 10,
    11, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 14, 13, 14, 13, 11,
    11, 11, 10, 11, 12, 12, 12, 12, 13, 14, 14, 14, 15, 15, 14, 10,
    12, 11, 11, 11, 12, 12, 13, 14, 14, 14, 14, 14, 14, 13, 14, 11,
    12, 12, 12, 12, 12, 13, 13, 13, 13, 15, 14, 14, 14, 14, 16, 11,
    14, 12, 12, 12, 13, 13, 14, 14, 14, 16, 15, 15, 15, 17, 15, 11,
    13, 13, 11, 12, 14, 14, 13, 14, 14, 15, 16, 15, 17, 15, 14, 11,
    9, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 8,
  ]
  private static let codes24: [UInt32] = [
    15, 13, 46, 80, 146, 262, 248, 434, 426, 669, 653, 649, 621, 517, 1032, 88,
    14, 12, 21, 38, 71, 130, 122, 216, 209, 198, 327, 345, 319, 297, 279, 42,
    47, 22, 41, 74, 68, 128, 120, 221, 207, 194, 182, 340, 315, 295, 541, 18,
    81, 39, 75, 70, 134, 125, 116, 220, 204, 190, 178, 325, 311, 293, 271, 16,
    147, 72, 69, 135, 127, 118, 112, 210, 200, 188, 352, 323, 306, 285, 540, 14,
    263, 66, 129, 126, 119, 114, 214, 202, 192, 180, 341, 317, 301, 281, 262, 12,
    249, 123, 121, 117, 113, 215, 206, 195, 185, 347, 330, 308, 291, 272, 520, 10,
    435, 115, 111, 109, 211, 203, 196, 187, 353, 332, 313, 298, 283, 531, 381, 17,
    427, 212, 208, 205, 201, 193, 186, 177, 169, 320, 303, 286, 268, 514, 377, 16,
    335, 199, 197, 191, 189, 181, 174, 333, 321, 305, 289, 275, 521, 379, 371, 11,
    668, 184, 183, 179, 175, 344, 331, 314, 304, 290, 277, 530, 383, 373, 366, 10,
    652, 346, 171, 168, 164, 318, 309, 299, 287, 276, 263, 513, 375, 368, 362, 6,
    648, 322, 316, 312, 307, 302, 292, 284, 269, 261, 512, 376, 370, 364, 359, 4,
    620, 300, 296, 294, 288, 282, 273, 266, 515, 380, 374, 369, 365, 361, 357, 2,
    1033, 280, 278, 274, 267, 264, 259, 382, 378, 372, 367, 363, 360, 358, 356, 0,
    43, 20, 19, 17, 15, 13, 11, 9, 7, 6, 4, 7, 5, 3, 1, 3,
  ]
  private static let lengths24: [UInt8] = [
    4, 4, 6, 7, 8, 9, 9, 10, 10, 11, 11, 11, 11, 11, 12, 9,
    4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 8,
    6, 5, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 7,
    7, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 7,
    8, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 7,
    9, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 7,
    9, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 7,
    10, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 8,
    10, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 8,
    10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 8,
    11, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 8,
    11, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 8,
    11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 8,
    11, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 8,
    12, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 8,
    8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 4,
  ]

  /// Big-value tables 0-31, indexed by `table_select`. Tables 4 and 14 are unused.
  static let tables: [HuffmanTable] = {
    let empty = HuffmanTable(xlen: 0, linbits: 0, codes: [], lengths: [])
    var t = [HuffmanTable](repeating: empty, count: 32)
    t[1] = HuffmanTable(xlen: 2, linbits: 0, codes: codes1, lengths: lengths1)
    t[2] = HuffmanTable(xlen: 3, linbits: 0, codes: codes2, lengths: lengths2)
    t[3] = HuffmanTable(xlen: 3, linbits: 0, codes: codes3, lengths: lengths3)
    t[5] = HuffmanTable(xlen: 4, linbits: 0, codes: codes5, lengths: lengths5)
    t[6] = HuffmanTable(xlen: 4, linbits: 0, codes: codes6, lengths: lengths6)
    t[7] = HuffmanTable(xlen: 6, linbits: 0, codes: codes7, lengths: lengths7)
    t[8] = HuffmanTable(xlen: 6, linbits: 0, codes: codes8, lengths: lengths8)
    t[9] = HuffmanTable(xlen: 6, linbits: 0, codes: codes9, lengths: lengths9)
    t[10] = HuffmanTable(xlen: 8, linbits: 0, codes: codes10, lengths: lengths10)
    t[11] = HuffmanTable(xlen: 8, linbits: 0, codes: codes11, lengths: lengths11)
    t[12] = HuffmanTable(xlen: 8, linbits: 0, codes: codes12, lengths: lengths12)
    t[13] = HuffmanTable(xlen: 16, linbits: 0, codes: codes13, lengths: lengths13)
    t[15] = HuffmanTable(xlen: 16, linbits: 0, codes: codes15, lengths: lengths15)
    let linbits16 = [1, 2, 3, 4, 6, 8, 10, 13]
    let linbits24 = [4, 5, 6, 7, 8, 9, 11, 13]
    for i in 0..<8 {
      t[16 + i] = HuffmanTable(xlen: 16, linbits: linbits16[i], codes: codes16, lengths: lengths16)
      t[24 + i] = HuffmanTable(xlen: 16, linbits: linbits24[i], codes: codes24, lengths: lengths24)
    }
    return t
  }()

  /// Count1 table A (table 32), indexed by `v * 8 + w * 4 + x * 2 + y`.
  static let count1CodesA: [UInt32] = [1, 5, 4, 5, 6, 5, 4, 4, 7, 3, 6, 0, 7, 2, 3, 1]
  static let count1LengthsA: [UInt8] = [1, 4, 4, 5, 4, 6, 5, 6, 4, 5, 5, 6, 5, 6, 6, 6]
  /// Count1 table B (table 33): fixed 4-bit codes.
  static let count1CodesB: [UInt32] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  static let count1LengthsB: [UInt8] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

  /// All tables in one flat, padded layout for fast lookup: table `t` (0-31) occupies
  /// entries `t * 256 + x * 16 + y`; count1 tables A and B are tables 32 and 33,
  /// indexed by `v * 8 + w * 4 + x * 2 + y`.
  static let flat: FlatHuffmanTables = {
    let lengths = UnsafeMutablePointer<UInt8>.allocate(capacity: 34 * 256)
    let codes = UnsafeMutablePointer<UInt32>.allocate(capacity: 34 * 256)
    lengths.initialize(repeating: 0, count: 34 * 256)
    codes.initialize(repeating: 0, count: 34 * 256)
    for (t, table) in tables.enumerated() where table.xlen > 0 {
      for x in 0..<table.xlen {
        for y in 0..<table.xlen {
          lengths[t * 256 + x * 16 + y] = table.lengths[x * table.xlen + y]
          codes[t * 256 + x * 16 + y] = table.codes[x * table.xlen + y]
        }
      }
    }
    for i in 0..<16 {
      lengths[32 * 256 + i] = count1LengthsA[i]
      codes[32 * 256 + i] = count1CodesA[i]
      lengths[33 * 256 + i] = count1LengthsB[i]
      codes[33 * 256 + i] = count1CodesB[i]
    }
    let linbits = UnsafeMutablePointer<Int>.allocate(capacity: 32)
    for t in 0..<32 { linbits[t] = tables[t].linbits }
    let pairCosts = UnsafeMutablePointer<SIMD16<UInt16>>.allocate(capacity: 256)
    for index in 0..<256 {
      var costs = SIMD16<UInt16>()
      for (lane, table) in FlatHuffmanTables.pairCostTables.enumerated() {
        costs[lane] = UInt16(lengths[table * 256 + index])
      }
      costs[FlatHuffmanTables.signLane] = (index >> 4 != 0 ? 1 : 0) + (index & 15 != 0 ? 1 : 0)
      pairCosts[index] = costs
    }
    return FlatHuffmanTables(lengths: lengths, codes: codes, linbits: linbits, pairCosts: pairCosts)
  }()
}

/// Pointers into permanently allocated Huffman tables (see ``HuffmanTables/flat``).
struct FlatHuffmanTables: @unchecked Sendable {
  let lengths: UnsafePointer<UInt8>
  let codes: UnsafePointer<UInt32>
  let linbits: UnsafePointer<Int>
  /// For each pair index `min(x, 15) * 16 + min(y, 15)`, its code length in every table
  /// of ``pairCostTables`` and, in ``signLane``, its sign bits, so that one vector add
  /// counts a pair in all tables at once.
  let pairCosts: UnsafePointer<SIMD16<UInt16>>

  /// The tables with a lane in ``pairCosts``: every table that codes pairs without
  /// escapes, then 16 and 24, whose codes the other escape tables share.
  static let pairCostTables = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24]
  static let signLane = 15

  /// The lane of ``pairCosts`` holding `table`'s code lengths (tables 16-23 share 16's,
  /// and 24-31 share 24's).
  @inline(__always)
  static func lane(of table: Int) -> Int {
    switch table {
    case 1...3: return table - 1
    case 5...13: return table - 2
    case 15: return 12
    case 16...23: return 13
    default: return 14
    }
  }

  init(
    lengths: UnsafeMutablePointer<UInt8>,
    codes: UnsafeMutablePointer<UInt32>,
    linbits: UnsafeMutablePointer<Int>,
    pairCosts: UnsafeMutablePointer<SIMD16<UInt16>>
  ) {
    self.lengths = UnsafePointer(lengths)
    self.codes = UnsafePointer(codes)
    self.linbits = UnsafePointer(linbits)
    self.pairCosts = UnsafePointer(pairCosts)
  }
}
